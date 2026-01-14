import json
import re
import uuid
import asyncio
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import time


from google.adk.agents import LlmAgent, BaseAgent
from google.adk.models.google_llm import Gemini
from google.adk.agents.callback_context import CallbackContext
from google.adk.runners import InMemoryRunner
from google.genai import types
from google.genai.types import Content, Part, GenerateContentResponse
from google.adk.events import Event

import config
import tools

# ==============================================================================
# 1. Configuration & Model Initialization
# ==============================================================================

retry_config = types.HttpRetryOptions(
    attempts=config.RETRY_ATTEMPTS,
    exp_base=2,
    initial_delay=config.RETRY_DELAY,
    http_status_codes=[429, 500, 503, 504],
)


# ==============================================================================
# 2. Data Schemas(Pydantic)
# ==============================================================================

class Replicates(BaseModel):
    biological: int = Field(..., description="Number of biological replicates")
    technical: int = Field(default=0, description="Number of technical replicates")

class SamplePrep(BaseModel):
    used: bool = Field(..., description="Whether enrichment or specific prep was used")
    method: str = Field(..., description="e.g., TiO2, IMAC, TMT Labeling, High-pH Fractionation")

class CurrentDataStatus(BaseModel):
    identification_count: Optional[int] = Field(0, description="Count of proteins/phosphosites")
    localization_threshold: float = Field(0.75, description="Localization probability threshold")
    qc_issues: List[str] = Field(default_factory=list, description="e.g., low MS2 coverage")

class ExperimentAnalysisContract(BaseModel):
    role: str = Field(..., description="'experimenter' (Wet Lab) or 'analyst' (Bioinformatics)")
    biological_goal: str = Field(..., description="Main biological objective")
    assay_type: str = Field(..., description="e.g., Phospho-proteomics, Label-free DDA")
    organism: str = Field(..., description="e.g., Human, Mouse")
    sample_type: str = Field(..., description="e.g., HeLa cells")
    comparison: List[str] = Field(..., description="Conditions to compare")
    instrument: str = Field(..., description="MS Instrument used")
    replicates: Replicates
    sample_prep: SamplePrep
    current_data_status: CurrentDataStatus
    analyst_wants: List[str] = Field(default_factory=list, description="Requirements from Analyst")
    experimenter_wants: List[str] = Field(default_factory=list, description="Requirements from Experimenter")
    notes: Optional[str] = None,

class summarized_content(BaseModel):
    pmcid: str = Field(..., description="pmcid for given paper")
    contents: str = Field(... , description="summarized contents")


# ==============================================================================
# 3. Prompt Templates (System Instructions & Tasks)
# ==============================================================================

PROMPTS = {
    "router_system": """
    You are a Router Agent. Classify input as 'experimenter' or 'analyst'.
    Output ONLY one word.
    """,
    "contract_builder_system": f"""
    You are a Contract Builder Agent. Extract experiment details to fill the JSON schema.
    Target Schema: {ExperimentAnalysisContract.model_json_schema()}
    Output ONLY valid JSON.
    """,
    "literature_system": """
    You are a Search Query Generator.
    Your task is to convert the user's intent into a single, effective keyword search query for a vector database.
    
    RULES:
    1. Output ONLY the query string.
    2. Do NOT output code blocks, tool calls, or explanations.
    3. Focus on technical keywords and methods
    """,
    
    "filter_system": """
    You are a Research Relevance Filter.
    Task: Select exactly 5 papers from the input list that are most relevant to the query.
    Output: Return ONLY a JSON list of strings (e.g., ["PMC123", "PMC456"]).
    Constraint: Do NOT output anything else. Use exact IDs from the input.
    """,
    "exp_advisor_system": "You are an Experimental Advisor (Wet Lab). Provide advice with citations. Create a template for what data/metadata should be passed to the analyst. USE GIVEN SOURCE DATA AND REFERENCE ID",
    "analyst_advisor_system": "You are an Analyst Advisor (Bioinformatics). Provide analysis plans with citations.Formulate questions to ask the experimenter if critical metadata is missing. USE GIVEN SOURCE DATA AND REFERENCE ID",
    "explainer_system": "You are a Contract Explainer Agent. Translate technical plans while preserving references. show references at the bottom. make sure you make same language as user",
    "qa_system": "You are a QA Agent. Critique outputs for quality and citations. If you think It's good enough just say 'PASS' and do not make any text",
    
    "summarize_system": f"""
    You are a Summarizer. Summarize the given paper to fill the JSON schema. You MUST KEEP citations
    Target Schema: {summarized_content.model_json_schema()}
    Output ONLY valid JSON.
    """,

    # --- [Task Templates] ---
    "exp_advisor_task": """

    [USER QUERY]
    {user_query}

    [CONTEXT: CONTRACT]
    {contract_json}
    
    [PROVIDED SOURCES]
    The following text is enclosed in <Source ID="PMC..."> tags.
    {literature_summary}
    
    [FEEDBACK]
    {feedback}
    
    Task:
    You are an Experimental Advisor.
    1. Suggest optimizations based **ONLY** on the content inside the <Source> tags above.
    2. **STRICT CITATION RULE**:
       - When you use information from a <Source ID="X"> tag, you MUST append [X] at the end of the sentence.
       - Example: If the text inside <Source ID="PMC12345"> says "Use TiO2", you write "We recommend TiO2 enrichment [PMC12345]".
    3. answer in same language as user query
    Output Format:
    ## Protocol Recommendation
    (Step-by-step guide with citations)
    ## Rationale
    ## Key References
    """,

    "analyst_advisor_task": """
    [USER QUERY]
    {user_query}

    [CONTEXT: CONTRACT]
    {contract_json}
    
    [PROVIDED SOURCES]
    The following text is enclosed in <Source ID="PMC..."> tags.
    {literature_summary}
    
    [FEEDBACK]
    {feedback}
    
    Task:
    You are a Bioinformatics Analyst Advisor.
    1. Design a pipeline based **ONLY** on the content inside the <Source> tags above.
    2. **STRICT CITATION RULE**:
       - When you use information from a <Source ID="X"> tag, you MUST append [X] at the end of the sentence.
    3. answer in same language as user query
    Output Format:
    ## Pipeline Design
    ## QC Checklist
    ## Key References
    """,

    "filter_task": """
    [USER QUERY]
    {query}
    
    [CANDIDATE PAPERS]
    The following is a list of candidate papers found in the database.
    {candidates_json}
    
    [TASK]
    1. Analyze the candidate papers above.
    2. Select 5 papers that are MOST relevant to the query.
    3. **STRICT CONSTRAINT**: 
       - You MUST select ONLY from the provided [CANDIDATE PAPERS] list.
       - You MUST use the exact 'pmcid' string as it appears in the list.
    
    [OUTPUT FORMAT]
    Return ONLY a JSON list of strings.
    Example: ["PMC12345", "PMC67890", ...]
    """,

    "summarize_task": """
    [GOAL]
    {goal}

    [PAPER]
    {paper}

    [TASK]
    1. extract methods, result and conclusion related to GOAL
    2. You MUST KEEP citations
    3. OUTPUT MUST BE JSON
    
    """
    ,

    "explainer_task": """
    Translate this technical plan for a: {target_role}.

    [USER QUERY]
    {user_query}

    [TECHNICAL PLAN]
    {advisor_output}
    
    Instructions:
    1. Simplify language but keep parameters.
    2. Keep inline citations [Ref X].
    3. MANDATORY: Include a "Key References" section at the bottom.
    4. answer in same language as user query

    """,

    "qa_task": """
    Review the output.
    [OUTPUT TO REVIEW]
    {final_output}
    
    Check for:
    1. Presence of Citations/References (Critical).
    
    If citations are missing, output "FAIL: Missing References".
    Otherwise, output "PASS".
    """
}

# ==============================================================================
# 4. Create Agents
# ==============================================================================
def save_log(message):
    #응답시간 저장 함수
    with open("response_time.txt", "a", encoding="utf-8") as f:
        f.write(message + "\n")

def create_agents() -> Dict[str, LlmAgent]:
    # Initialize models here to ensure they are bound to the current event loop
    model_fast = Gemini(model=config.MODEL_FAST, retry_options=retry_config)
    model_high = Gemini(model=config.MODEL_HIGH_REASONING, retry_options=retry_config)

    # # 1. Filter용 엄격한 설정 (JSON 강제)
    # filter_config = types.HttpRetryOptions(attempts=3)
    # model_filter = Gemini(
    #     model=config.MODEL_FAST, 
    #     retry_options=filter_config,
    #     generation_config=types.GenerateContentConfig(
    #         temperature=0.0, 
    #         response_mime_type="application/json"
    #     )
    # )

    return {
        "router": LlmAgent(model=model_fast, name="router", instruction=PROMPTS["router_system"]),
        "contract_builder": LlmAgent(model=model_fast, name="contract_builder", instruction=PROMPTS["contract_builder_system"]),
        "literature": LlmAgent(model=model_fast, name="literature", instruction=PROMPTS["literature_system"]),
        "filter": LlmAgent(model=model_fast, name="filter", instruction=PROMPTS["filter_system"]),
        "exp_advisor": LlmAgent(model=model_high, name="exp_advisor", instruction=PROMPTS["exp_advisor_system"]),
        "analyst_advisor": LlmAgent(model=model_high, name="analyst_advisor", instruction=PROMPTS["analyst_advisor_system"]),
        "explainer": LlmAgent(model=model_fast, name="explainer", instruction=PROMPTS["explainer_system"]),
        "qa": LlmAgent(model=model_fast, name="qa", instruction=PROMPTS["qa_system"]),
        "summarize": LlmAgent(model=model_fast, name="summarize", instruction=PROMPTS["summarize_system"]),
    }

# ==============================================================================
# 5. Pipeline
# ==============================================================================

class BioinformaticsPipeline(BaseAgent):
    agents: Dict[str, Any] = Field(..., description="Dictionary of sub-agents")
    plugins: List[Any] = Field(default_factory=list, description="List of plugins")

    def __init__(self, agents: Dict[str, Any]):
        super().__init__(name="bioinformatics_pipeline", agents=agents)
        self.description = "A comprehensive bioinformatics pipeline."

    def _extract_json(self, text: str) -> Any:
        """JSON 파싱 및 에러 처리"""
        if not text: return {}
        try:
            text = text.strip()
            if "```" in text:
                pattern = r"```(?:json)?\s*([\[\{].*?[\]\}])\s*```"
                match = re.search(pattern, text, re.DOTALL)
                if match: return json.loads(match.group(1))
            if '[' in text and ']' in text:
                try:
                    start = text.find('[')
                    end = text.rfind(']') + 1
                    return json.loads(text[start:end])
                except: pass
            
            pmc_ids = re.findall(r'(PMC\d+)', text)
            if pmc_ids: return list(set(pmc_ids))[:5]
            return {}
        except Exception:
            return {}

    def _parse_response_text(self, response) -> str:
        """Runner의 결과(Events List)에서 최종 텍스트 추출"""
        try:
            if isinstance(response, list) and len(response) > 0:
                last_event = response[-1]
                if hasattr(last_event, 'content') and last_event.content.parts:
                    return last_event.content.parts[0].text
            
            if hasattr(response, 'content') and response.content.parts:
                return response.content.parts[0].text
                
        except Exception as e:
            print(f"⚠️ Text Parsing Warning: {e}")
        return ""

    async def _invoke_sub_agent(self, agent_name: str, prompt: str) -> str:
        """
        InMemoryRunner를 사용하여 서브 에이전트에게 프롬프트를 전달
        """
        # 1. 해당 에이전트를 위한 Runner 즉석 생성 (플러그인 전달)
        runner = InMemoryRunner(agent=self.agents[agent_name], plugins=self.plugins)
        
        response_events = await runner.run_debug(prompt)
        
        # 3. 결과 텍스트 추출
        return self._parse_response_text(response_events)

    async def _summarize_single_paper(self, paper_content: str, goal: str) -> str:
        """논문 하나를 받아서 요약 결과를 JSON String으로 반환"""
        summarize_prompt = PROMPTS["summarize_task"].format(
            paper=paper_content, 
            goal=goal
        )
        try:
            # Summarize Agent 호출
            summarized_text = await self._invoke_sub_agent("summarize", summarize_prompt)
            summarized_data = self._extract_json(summarized_text)
            return json.dumps(summarized_data, indent=2)
        except Exception as e:
            print(f"⚠️ Summarization failed for a paper: {e}")
            return "{}" 


    async def _run_async_impl(self, context: CallbackContext):
        save_log("--------------new attemps--------------")
        # 입력 데이터 추출
        input_data = context.user_content
        if hasattr(input_data, 'parts'): input_data = input_data.parts[0].text
        elif not isinstance(input_data, str): input_data = str(input_data)
        t_start = time.perf_counter()

        print(f"Pipeline Started: {input_data}")
        
        # ------------------------------------------------------------------
        # 1. Router Agent
        # ------------------------------------------------------------------
        role_text = await self._invoke_sub_agent("router", input_data)
        role = "experimenter" if "experimenter" in role_text.lower() else "analyst"
        print(f"Role Identified: {role}")
        save_log(f"⏱️ [Time] Router: {time.perf_counter() - t_start:.2f}s\n")

        # ------------------------------------------------------------------
        # 2. Contract Builder
        # ------------------------------------------------------------------
        t_start = time.perf_counter()

        contract_text = await self._invoke_sub_agent("contract_builder", f"Analyze: {input_data}")
        contract_data = self._extract_json(contract_text)
        if not contract_data: 
            contract_data = {"assay_type": "proteomics", "biological_goal": "analysis", "role": role}
        contract_str = json.dumps(contract_data, indent=2)
        save_log(f"⏱️ [Time] Contract Builder: {time.perf_counter() - t_start:.2f}s\n")

        # ------------------------------------------------------------------
        # 3. Reasoning Loop
        # ------------------------------------------------------------------
        MAX_RETRIES = 3
        current_retry = 0
        qa_status = "FAIL"
        feedback = ""
        final_output_text = ""

        while current_retry < MAX_RETRIES:
            print(f"Loop Attempt: {current_retry + 1}")
            
            # A. Literature Agent (Query Generation)
            q_prompt = f"Create a search query for: {contract_data.get('assay_type')} {contract_data.get('biological_goal')}"
            if feedback: q_prompt += f" considering feedback: {feedback}"
            
            t_start = time.perf_counter()
            search_query = await self._invoke_sub_agent("literature", q_prompt)
            print(f"Generated Query: {search_query}")
            save_log(f"⏱️ [Time] Literature (Query Gen): {time.perf_counter() - t_start:.2f}s\n")
            
            # B. Vector Search (Python Tool Direct Call)
            t_start = time.perf_counter()
            raw_candidates_json = tools.search_vectors(search_query, k=20)
            
            if len(json.loads(raw_candidates_json)) == 0:
                print("No papers found. Retrying...")
                feedback = "Search query returned no results. Make it broader."
                current_retry += 1
                continue
            save_log(f"⏱️ [Time] Vector Search: {time.perf_counter() - t_start:.2f}s\n")

            # C. Filter Agent (Selection)
            filter_prompt = PROMPTS["filter_task"].format(
                query=search_query,
                candidates_json=raw_candidates_json
            )

            with open("./temp_full_text.txt", "w") as f:
                f.write(filter_prompt)
            t_start = time.perf_counter()
            filter_output = await self._invoke_sub_agent("filter", filter_prompt)

            with open("./temp_full_text.txt", "a") as f:
                f.write(filter_output)
            save_log(f"⏱️ [Time] Filter: {time.perf_counter() - t_start:.2f}s\n")

            try:
                extracted_data = self._extract_json(filter_output)
                if isinstance(extracted_data, list) and len(extracted_data) > 0:
                    selected_ids = extracted_data
                    print(f"Filter Selected: {len(selected_ids)} papers")
                else:
                    raise ValueError("Output is not a valid list")
            except:
                print(f"Filter parsing failed. Using FAISS top 5 fallback.")
                all_c = json.loads(raw_candidates_json)
                selected_ids = [c['pmcid'] for c in all_c[:5]]
            
            # ID 검증
            all_candidates = json.loads(raw_candidates_json)
            candidate_map = {}
            for c in all_candidates:
                norm_id = re.sub(r'[^0-9]', '', str(c.get('pmcid', '')))
                if norm_id: candidate_map[norm_id] = c.get('pmcid')

            validated_ids = []
            for sel_id in selected_ids:
                sel_norm = re.sub(r'[^0-9]', '', str(sel_id))
                if sel_norm in candidate_map:
                    validated_ids.append(candidate_map[sel_norm])
            
            if not validated_ids:
                validated_ids = [c['pmcid'] for c in all_candidates[:5]]
            
            selected_ids = validated_ids
            print(f"Final IDs: {selected_ids}")
            
            #C+ Summarize Agent
            #논문 하나하나 확인하면서 병렬처리. summarize agent 사용
            #input= 논문, goal

            # t_start = time.perf_counter()
            # selected_papers = []
            # for pmcid in selected_ids:
            #     selected_papers.append(tools.get_paper_by_id(pmcid, raw_candidates_json))
            
            # summarize_outputs = []

            # for selected_paper in selected_papers:
            #     summarize_prompt = PROMPTS["summarize_task"].format(
            #         paper=selected_paper, 
            #         goal=search_query
            #     )
            #     #summarize output안에 paper json들이 들어가야함
            #     sumamarized_text = await self._invoke_sub_agent("summarize", summarize_prompt)
            #     sumamarized_data = self._extract_json(sumamarized_text)
            #     summarized_str = json.dumps(sumamarized_data, indent=2)
            #     summarize_outputs.append(summarized_str)
            # save_log(f"⏱️ [Time] summarize: {time.perf_counter() - t_start:.2f}s\n")

            # #full_text_builder using summarize_outputs
            # full_text_context = tools.full_text_builder(summarize_outputs)

            selected_papers = []
            for pmcid in selected_ids:
                selected_papers.append(tools.get_paper_by_id(pmcid, raw_candidates_json))
            
      
            summarize_tasks = [
                self._summarize_single_paper(paper, search_query) 
                for paper in selected_papers if paper
            ]

            t_start = time.perf_counter()
            print(f"Starting parallel summarization for {len(summarize_tasks)} papers...")
            
            # 3. asyncio.gather로 비동기 실행
            summarize_outputs = await asyncio.gather(*summarize_tasks)
            
            save_log(f"⏱️ [Time] Parallel Summarization: {time.perf_counter() - t_start:.2f}s")

            # D. Full Text Loading
            full_text_context = tools.full_text_builder(summarize_outputs)

            with open("./temp_full_text.txt", "a") as f:
                f.write("****************full text context*********")
                f.write(full_text_context)

            # E. Advisor Agent
            target_agent = "exp_advisor" if role == 'experimenter' else "analyst_advisor"
            target_key = "exp_advisor_task" if role == 'experimenter' else "analyst_advisor_task"
            t_start = time.perf_counter()
            advisor_prompt = PROMPTS[target_key].format(
                contract_json=contract_str, 
                literature_summary=full_text_context,
                feedback=feedback or "None",
                user_query=input_data
            )
            
            raw_advice = await self._invoke_sub_agent(target_agent, advisor_prompt)
            save_log(f"⏱️ [Time] Advisor: {time.perf_counter() - t_start:.2f}s\n")
            
            # F. Explainer Agent
            t_start = time.perf_counter()
            exp_prompt = PROMPTS["explainer_task"].format(target_role=role, advisor_output=raw_advice, user_query=input_data)
            final_output_text = await self._invoke_sub_agent("explainer", exp_prompt)
            save_log(f"⏱️ [Time] Explainer: {time.perf_counter() - t_start:.2f}s\n")

            # G. QA Agent
            t_start = time.perf_counter()
            qa_prompt = PROMPTS["qa_task"].format(final_output=final_output_text)
            qa_result = await self._invoke_sub_agent("qa", qa_prompt)
            
            if "PASS" in qa_result.upper():
                qa_status = "PASS"
                break
            else:
                feedback = qa_result
                current_retry += 1
            save_log(f"⏱️ [Time] QA: {time.perf_counter() - t_start:.2f}s\n")
            
        if qa_status == "FAIL": 
            final_output_text = f"[Max Retries Reached]\n{final_output_text}"
            save_log("QA Failed")

        final_event = Event(
            author="model",
            content=Content(parts=[Part(text=final_output_text)])
        )
        

        yield final_event


def get_pipeline(api_key: str = None):
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    agents = create_agents()
    return BioinformaticsPipeline(agents)

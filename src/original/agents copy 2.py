import json
import re
import uuid
import asyncio
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Google ADK & GenAI Imports
from google.adk.agents import LlmAgent, BaseAgent
from google.adk.models.google_llm import Gemini
from google.adk.agents.callback_context import CallbackContext
from google.genai import types
from google.genai.types import Content, Part, GenerateContentResponse

# User Modules
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

model_fast = Gemini(model=config.MODEL_FAST, retry_options=retry_config)
model_high = Gemini(model=config.MODEL_HIGH_REASONING, retry_options=retry_config)

# ==============================================================================
# 2. Data Schemas
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
    notes: Optional[str] = None

# ==============================================================================
# [cite_start]3. Prompt Templates (System vs Task 분리) [cite: 76-84]
# ==============================================================================

PROMPTS = {
    # --- [System Prompts] 에이전트 생성 시 사용 (변수 없음) ---
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

    # --- [Task Templates] 실행 시 데이터 주입 ({variable} 포함) ---

    "exp_advisor_task": """
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
       - **DO NOT** use any ID that is not listed in the <Source> tags.
       - **DO NOT** make up IDs like [Ref 1] or [PMC99999]. Use the exact string provided in ID="...".
    
    Output Format:
    ## Protocol Recommendation
    (Step-by-step guide with citations)
    
    ## Rationale
    (Why this method? Cite sources)
    
    ## Key References
    (List the IDs used, e.g., [PMC12345]:  summary)
    """,

    "analyst_advisor_task": """
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
       - Example: If the text inside <Source ID="PMC67890"> mentions "MaxQuant", you write "Use MaxQuant software [PMC67890]".
       - **DO NOT** use any ID that is not listed in the <Source> tags.
    
    Output Format:
    ## Pipeline Design
    (Steps and Tools with citations)
    
    ## QC Checklist
    (Based on the literature)
    
    ## Key References
    (List the IDs used, e.g., [PMC12345]:  summary)
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
       - **DO NOT** invent or hallucinate new IDs.
       - **DO NOT** use IDs from your internal knowledge.
       - If the list contains "Unknown" IDs, do not select them if possible.
    
    [OUTPUT FORMAT]
    Return ONLY a JSON list of strings.
    Example: ["PMC12345", "PMC67890", ...]
    """,

    "explainer_task": """
    Translate this technical plan for a: {target_role}.
    [TECHNICAL PLAN]
    {advisor_output}
    
    Instructions:
    1. Simplify language but keep parameters.
    2. Keep inline citations [Ref X].
    3. MANDATORY: Include a "Key References" section at the bottom.
     
    """,

    "qa_task": """
    Review the output.
    [OUTPUT TO REVIEW]
    {final_output}
    
    Check for:
    1. Presence of Citations/References (Critical).
    
    If citations are missing, output "FAIL: Missing References".
    Otherwise, output "PASS".
    
    **CRITICAL**: Be extremely lenient. If there is ANY output with citations, output "PASS".
    Do not critique style, tone, or minor missing details.
    
    """
}
#Output ONLY the word "PASS".
# ==============================================================================
# 4. Agent Factory (수정됨: _system 프롬프트 사용)
# ==============================================================================

def create_agents() -> Dict[str, LlmAgent]:
    # 1. Filter용 엄격한 설정 생성 (JSON 강제)
    filter_config = types.HttpRetryOptions(attempts=3)
    # 모델 생성 시 generation_config 주입
    model_filter = Gemini(
        model=config.MODEL_FAST, 
        retry_options=filter_config,
        generation_config=types.GenerateContentConfig(
            temperature=0.0, 
            response_mime_type="application/json" # ★ 핵심: JSON 외엔 말 못하게 함
        )
    )

    return {
        "router": LlmAgent(model=model_fast, name="router", instruction=PROMPTS["router_system"]),
        "contract_builder": LlmAgent(model=model_fast, name="contract_builder", instruction=PROMPTS["contract_builder_system"]),
        "literature": LlmAgent(model=model_fast, name="literature", instruction=PROMPTS["literature_system"]),
        
        # 2. Filter Agent에만 strict model 적용
        "filter": LlmAgent(model=model_filter, name="filter", instruction=PROMPTS["filter_system"]),
        
        "exp_advisor": LlmAgent(model=model_high, name="exp_advisor", instruction=PROMPTS["exp_advisor_system"]),
        "analyst_advisor": LlmAgent(model=model_high, name="analyst_advisor", instruction=PROMPTS["analyst_advisor_system"]),
        "explainer": LlmAgent(model=model_fast, name="explainer", instruction=PROMPTS["explainer_system"]),
        "qa": LlmAgent(model=model_fast, name="qa", instruction=PROMPTS["qa_system"]),
    }

# ==============================================================================
# 5. Pipeline Logic (수정됨: _task 프롬프트 사용)
# ==============================================================================

class BioinformaticsPipeline(BaseAgent):
    agents: Dict[str, Any] = Field(..., description="Dictionary of sub-agents")
    # ... (__init__, _extract_json 등 기존 메서드 유지) ...
    def __init__(self, agents: Dict[str, Any]):
        super().__init__(name="bioinformatics_pipeline", agents=agents)
        self.description = "A comprehensive bioinformatics pipeline."

    # agents.py 내 _extract_json 메서드 교체

    def _extract_json(self, text: str) -> Any:
        """
        JSON 파싱을 시도하고, 실패하면 텍스트에서 'PMC숫자' 패턴을 강제로 추출합니다.
        """
        if not text: return {}
        try:
            text = text.strip()
            
            # 1. 마크다운 코드 블록 처리
            if "```" in text:
                pattern = r"```(?:json)?\s*([\[\{].*?[\]\}])\s*```"
                match = re.search(pattern, text, re.DOTALL)
                if match: return json.loads(match.group(1))

            # 2. 대괄호/중괄호 탐색
            if '[' in text and ']' in text:
                try:
                    start = text.find('[')
                    end = text.rfind(']') + 1
                    return json.loads(text[start:end])
                except: pass
            
            # 3. [핵심] 정규식으로 PMC ID 강제 추출 (불렛 포인트 대응)
            # 텍스트가 깨져서 와도 PMC ID만 있으면 리스트로 복구합니다.
            pmc_ids = re.findall(r'(PMC\d+)', text)
            if pmc_ids:
                return list(set(pmc_ids))[:5] # 중복 제거 후 5개 반환

            return {}
        except Exception as e:
            print(f"⚠️ JSON Parsing Logic Failed: {e}")
            return {}

    def _parse_response_text(self, response) -> str:
        """응답 객체에서 텍스트만 추출하는 헬퍼 함수"""
        try:
            if isinstance(response, str): return response
            if hasattr(response, 'text') and response.text: return response.text
            if hasattr(response, 'content') and response.content.parts: return response.content.parts[0].text
            if hasattr(response, 'candidates') and response.candidates: return response.candidates[0].content.parts[0].text
        except:
            pass
        # 수정: 이어붙이기를 위해 실패 시 "No content"가 아닌 빈 문자열 반환
        return ""
        
    async def run_step_helper(self, agent_name, prompt, capture_list, context):
        step_ctx = context.model_copy(update={'user_content': prompt})
        
        # [수정 1] 전체 텍스트를 담을 빈 문자열 준비
        full_response_text = ""
        
        async for event in self.agents[agent_name].run_async(step_ctx):
            yield event
            
            # [수정 2] 매 이벤트(Chunk)마다 텍스트를 추출하여 이어 붙임
            chunk_text = self._parse_response_text(event)
            if chunk_text:
                full_response_text += chunk_text
        
        # (디버깅) 전체 텍스트가 잘 모였는지 확인
        # print(f"📝 [{agent_name} Full Output]: {full_response_text}") 

        # [수정 3] 누적된 전체 텍스트를 저장
        capture_list[0] = full_response_text

    async def _run_async_impl(self, context: CallbackContext):
        input_data = context.user_content
        if hasattr(input_data, 'parts'): input_data = input_data.parts[0].text
        elif not isinstance(input_data, str): input_data = str(input_data)
        
        print(f"🚀 Pipeline Started: {input_data}")
        
        # 1. Router & 2. Contract Builder (기존과 동일)
        capture = [None]
        async for e in self.run_step_helper("router", input_data, capture, context): yield e
        role = "experimenter" if "experimenter" in capture[0].lower() else "analyst"
        print(f"📍 Role: {role}")

        capture = [None]
        async for e in self.run_step_helper("contract_builder", f"Analyze: {input_data}", capture, context): yield e
        contract_data = self._extract_json(capture[0])
        if not contract_data: contract_data = {"assay_type": "proteomics", "biological_goal": "analysis", "role": role}
        contract_str = json.dumps(contract_data, indent=2)

        # 3. Loop Logic (수정됨)
        MAX_RETRIES = 3
        current_retry = 0
        qa_status = "FAIL"
        feedback = ""
        final_output_text = ""

        while current_retry < MAX_RETRIES:
            print(f"🔄 Attempt: {current_retry + 1}")
            
            # 1. Literature Agent: 검색 쿼리 생성
            q_prompt = f"Create a search query for: {contract_data.get('assay_type')} {contract_data.get('biological_goal')}"
            if feedback: q_prompt += f" considering feedback: {feedback}"
            
            capture = [None]
            async for e in self.run_step_helper("literature", q_prompt, capture, context): yield e
            search_query = capture[0]
            print(f"🔍 Generated Query: {search_query}")
            
            # 2. Tool Execution (Python Direct Call): 40개 후보 검색
            # Agent가 아니라 Python 함수를 직접 부릅니다. (속도/안정성)
            raw_candidates_json = tools.search_vectors(search_query, k=20)
            
            # Debug: Print first few candidates to verify content
            try:
                debug_candidates = json.loads(raw_candidates_json)
            except:
                print("⚠️ Could not parse candidates for debugging.")
            
            if len(json.loads(raw_candidates_json)) == 0:
                print("⚠️ No papers found. Retrying with broader scope.")
                feedback = "Search query returned no results. Make it broader."
                current_retry += 1
                continue

            # 3. Filter Agent (Gemini Flash): 5개 선정
            filter_prompt = PROMPTS["filter_task"].format(
                query=search_query,
                candidates_json=raw_candidates_json
            )
            with open("./temp_full_text.txt", "w") as f:
                
                f.write(filter_prompt)
                f.write("\n\n\n\n\n\n")
            capture = [None]
            async for e in self.run_step_helper("filter", filter_prompt, capture, context): yield e
            
            with open("./temp_full_text.txt", "a") as f:
                f.write("filter answer")
                f.write(capture[0])
                f.write("\n\n\n\n\n\n")
            # 결과 파싱 (JSON 리스트 추출)
            try:
                extracted_data = self._extract_json(capture[0])
                
                # 리스트인지 확인 (Filter Agent는 리스트를 줘야 함)
                if isinstance(extracted_data, list) and len(extracted_data) > 0:
                    selected_ids = extracted_data
                    print(f"✅ Filter Agent Selection applied: {len(selected_ids)} papers")
                else:
                    raise ValueError("Output is not a valid list")
                    
            except Exception as e:
                print(f"⚠️ Filter parsing failed ({e}). Using FAISS top 5 fallback.")
                all_c = json.loads(raw_candidates_json)
                selected_ids = [c['pmcid'] for c in all_c[:5]]
                
            print(f"⚡ Final Target IDs: {selected_ids}")
                
            print(f"⚡ Filtered Top 5 IDs: {selected_ids}")

            all_candidates = json.loads(raw_candidates_json)

            # 비교를 위해 { '12345': 'PMC12345' } 형태의 맵 생성
            candidate_map = {}
            for c in all_candidates:
                # tools.py에서 쓴 normalize 로직과 동일하게 숫자만 추출
                norm_id = re.sub(r'[^0-9]', '', str(c.get('pmcid', '')))
                if norm_id:
                    candidate_map[norm_id] = c.get('pmcid')

            # 2. 선택된 ID 검증 및 필터링
            validated_ids = []
            for sel_id in selected_ids:
                sel_norm = re.sub(r'[^0-9]', '', str(sel_id))
                if sel_norm in candidate_map:
                    # 실제 후보군에 있는 올바른 포맷의 ID로 저장
                    validated_ids.append(candidate_map[sel_norm])
                else:
                    print(f"⚠️ Filter Agent hallucinated ID '{sel_id}' (Not in local DB). Removing.")

            # 3. 만약 유효한 ID가 하나도 없으면 Fallback
            if not validated_ids:
                print("⚠️ No valid IDs selected by Filter. Falling back to Top 5 FAISS results.")
                validated_ids = [c['pmcid'] for c in all_candidates[:5]]

            selected_ids = validated_ids
            print(f"✅ Final Validated IDs: {selected_ids}")
            # [수정된 로직 끝] --------------------------------------------------

            # 4. Tool Execution: 선정된 5개의 Full Text 로드
            full_text_context = tools.get_full_text_by_ids(selected_ids, raw_candidates_json)

            # 5. Advisor Agent (Gemini Pro): 심층 분석
            target_agent = "exp_advisor" if role == 'experimenter' else "analyst_advisor"
            target_key = "exp_advisor_task" if role == 'experimenter' else "analyst_advisor_task"
            
            full_prompt = PROMPTS[target_key].format(
                contract_json=contract_str, 
                literature_summary=full_text_context, # 여기에 Full Text가 들어감
                feedback=feedback or "None"
            )
            
            with open("./temp_full_text.txt", "a") as f:
                f.write(full_prompt)

            capture = [None]
            async for e in self.run_step_helper(target_agent, full_prompt, capture, context): yield e
            raw_advice = capture[0]

            with open("./temp_full_text.txt", "a") as f:
                f.write(raw_advice)

            # 6. Explainer & QA (기존 동일)
            capture = [None]
            exp_prompt = PROMPTS["explainer_task"].format(target_role=role, advisor_output=raw_advice)
            async for e in self.run_step_helper("explainer", exp_prompt, capture, context): yield e
            final_output_text = capture[0]

            capture = [None]
            qa_prompt = PROMPTS["qa_task"].format(final_output=final_output_text)
            async for e in self.run_step_helper("qa", qa_prompt, capture, context): yield e
            
            if capture[0] and ("PASS" in capture[0].upper()):
                qa_status = "PASS"
                break
            else:
                feedback = capture[0]
                current_retry += 1

        if qa_status == "FAIL": final_output_text = f"⚠️ [Max Retries]\n{final_output_text}"
        
        final_response = GenerateContentResponse(candidates=[types.Candidate(content=Content(parts=[Part(text=final_output_text)]), finish_reason="STOP", index=0)])
        try: 
            object.__setattr__(final_response, 'usage_metadata', None)
            object.__setattr__(final_response, 'partial', None)
            object.__setattr__(final_response, 'actions', None)
            object.__setattr__(final_response, 'timestamp', datetime.now(timezone.utc))
            object.__setattr__(final_response, 'id', str(uuid.uuid4()))
            object.__setattr__(final_response, 'author', 'pipeline')
            object.__setattr__(final_response, 'content', final_response.candidates[0].content)
        except: pass
        yield final_response

# ==============================================================================
# 6. Exportable Instance
# ==============================================================================

# --- 4. Factory for Pipeline ---

def get_pipeline(api_key: str = None):
    """
    Creates and returns the BioinformaticsPipeline instance.
    If api_key is provided, it sets the environment variable.
    """
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        # Also configure genai directly if needed, but ADK usually reads env
        # import google.genai
        # google.genai.configure(api_key=api_key)
    
    # Re-create agents with the new key (if set)
    agents = create_agents()
    return BioinformaticsPipeline(agents)

# Global instance for backward compatibility (uses default env var)
# bioinformatics_pipeline = get_pipeline()

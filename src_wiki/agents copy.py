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
    You are a Literature Retrieval Agent. Your job is to call the 'search_literature' tool. 
    Focus on methods and similar experimental setups.
    For each paper you find, output one line in the exact format:
    paper_id: summary
    """,
    "evidence_qa_system": """
    You are a Paper Evidence QA Agent. 
    Evaluate search results and format bibliography.
    If more than 5 papers are found, output only top relevant 5 papers.
    """,
    "exp_advisor_system": "You are an Experimental Advisor (Wet Lab). Provide advice with citations. Create a template for what data/metadata should be passed to the analyst.",
    "analyst_advisor_system": "You are an Analyst Advisor (Bioinformatics). Provide analysis plans with citations.Formulate questions to ask the experimenter if critical metadata is missing.",
    "explainer_system": "You are a Contract Explainer Agent. Translate technical plans while preserving references. show references at the bottom. make sure you make same language as user",
    "qa_system": "You are a QA Agent. Critique outputs for quality and citations. If you think It's good enough just say 'PASS' and do not make any text",

    # --- [Task Templates] 실행 시 데이터 주입 ({variable} 포함) ---
    "evidence_qa_task": """
    Evaluate the relevance and quality of the following search results.
    
    [SEARCH RESULTS]
    {search_results}
    
    Task:
    1. **Filter**: Check if the results are relevant to the user's query. 
       - If NO relevant papers are found, output ONLY: "LOW_QUALITY".
    
    2. **Summarize (Section 1)**: Extract key experimental conditions or computational methods.
       - You MUST cite the source for every fact using [Ref X].
       - Example: "Use 80% ACN for elution [Ref 1]."
    
    3. **Bibliography (Section 2)**: Create a structured list of references.
       - Format: [Ref X] Title/Filename (Year if available), Journal/Source.
       - Ensure every [Ref X] used in the summary is listed here.
    
    **CRITICAL OUTPUT RULE**:
    - Do NOT output "PASS".
    - Output the **Content** (Summary + Bibliography) directly.
    - Only output "LOW_QUALITY" if the search failed completely.
    """,

    "exp_advisor_task": """
    [CONTEXT: CONTRACT]
    {contract_json}
    
    [SELECTED RESEARCH PAPERS (FULL TEXT)]
    {literature_summary}
    
    [FEEDBACK]
    {feedback}
    
    Task:
    1. Using the <FullBody> and <Methods> of the provided papers, suggest specific experimental optimizations.
    2. Cite specific papers using their ID (e.g., [PMC12345]).
    3. If methods conflict, explain the pros/cons based on the paper's context.
    """,

    "analyst_advisor_task": """
    [CONTEXT: CONTRACT]
    {contract_json}
    
    [SELECTED RESEARCH PAPERS (FULL TEXT)]
    {literature_summary}
    
    [FEEDBACK]
    {feedback}
    
    Task:
    1. Using the provided papers, design a bioinformatics pipeline.
    2. Focus on specific software/parameters mentioned in the <FullBody>.
    3. Create a QC checklist based on the literature.
    """,

    "explainer_task": """
    Translate this technical plan for a: {target_role}.
    [TECHNICAL PLAN]
    {advisor_output}
    
    Instructions:
    1. Simplify language but keep parameters.
    2. Keep inline citations [Ref X].
    3. MANDATORY: Include a "## Key References" section at the bottom.
     
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
    Output ONLY the word "PASS".
    """
}

# ==============================================================================
# 4. Agent Factory (수정됨: _system 프롬프트 사용)
# ==============================================================================

def create_agents() -> Dict[str, LlmAgent]:
    return {
        "router": LlmAgent(model=model_fast, name="router", instruction=PROMPTS["router_system"]),
        "contract_builder": LlmAgent(model=model_fast, name="contract_builder", instruction=PROMPTS["contract_builder_system"]),
        "literature": LlmAgent(model=model_fast, name="literature", tools=[tools.search_literature], instruction=PROMPTS["literature_system"]),
        "evidence_qa": LlmAgent(model=model_fast, name="evidence_qa", instruction=PROMPTS["evidence_qa_system"]),
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
    
    def __init__(self, agents: Dict[str, Any]):
        super().__init__(name="bioinformatics_pipeline", agents=agents)
        self.description = "A comprehensive bioinformatics pipeline."

    def _extract_json(self, text: str) -> dict:
        try:
            if "```" in text:
                pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
                match = re.search(pattern, text, re.DOTALL)
                if match: return json.loads(match.group(1))
            start, end = text.find('{'), text.rfind('}') + 1
            return json.loads(text[start:end]) if start != -1 and end != 0 else {}
        except: return {}

    def _parse_response_text(self, response) -> str:
        if isinstance(response, str): return response
        if hasattr(response, 'text') and response.text: return response.text
        if hasattr(response, 'content') and response.content.parts: return response.content.parts[0].text
        if hasattr(response, 'candidates') and response.candidates: return response.candidates[0].content.parts[0].text
        return "No content"

    async def _run_async_impl(self, context: CallbackContext):
        input_data = context.user_content
        if hasattr(input_data, 'parts'): input_data = input_data.parts[0].text
        elif not isinstance(input_data, str): input_data = str(input_data)
        
        print(f"🚀 Pipeline Started: {input_data}")

        async def run_step(agent_name, prompt, capture_list):
            step_ctx = context.model_copy(update={'user_content': prompt})
            response = None
            async for event in self.agents[agent_name].run_async(step_ctx):
                yield event
                if isinstance(event, (str, GenerateContentResponse)) or hasattr(event, 'content'):
                    response = event
            capture_list[0] = self._parse_response_text(response)

        # 1. Router
        capture = [None]
        async for e in run_step("router", input_data, capture): yield e
        role = "experimenter" if "experimenter" in capture[0].lower() else "analyst"
        print(f"📍 Role: {role}")

        # 2. Contract Builder
        capture = [None]
        async for e in run_step("contract_builder", f"Analyze: {input_data}", capture): yield e
        contract_data = self._extract_json(capture[0])
        if not contract_data: contract_data = {"assay_type": "proteomics", "biological_goal": "analysis", "role": role}
        contract_str = json.dumps(contract_data, indent=2)

        # 3. Loop Logic
        MAX_RETRIES, current_retry, qa_status, feedback = 3, 0, "FAIL", ""
        final_output_text = ""

        while current_retry < MAX_RETRIES:
            print(f"🔄 Attempt: {current_retry + 1}")
            
            # Literature
            q = f"{contract_data.get('assay_type')} {contract_data.get('biological_goal')}"
            if feedback: q += f" refined by: {feedback}"
            
            capture = [None]
            async for e in run_step("literature", f"Search: {q}", capture): yield e
            literature_output = capture[0]
            print("WWWWWWWWWWWWWWWW")
            print(literature_output)
            print("WWWWWWWWWWWWWWWW")
            # Evidence QA (Task Prompt 사용)
            capture = [None]
            ev_prompt = PROMPTS["evidence_qa_task"].format(search_results=literature_output)
            async for e in run_step("evidence_qa", ev_prompt, capture): yield e
            evidence = capture[0]

            if "LOW_QUALITY" in evidence or "no relevant" in evidence.lower():
                print("⚠️ Low quality evidence. Retrying...")
                feedback = "Broaden keywords."
                current_retry += 1
                continue

            # Advisor (Task Prompt 사용)
            print(f"🧠 Advisor for {role}")
            target_agent = "exp_advisor" if role == 'experimenter' else "analyst_advisor"
            target_key = "exp_advisor_task" if role == 'experimenter' else "analyst_advisor_task"
            
            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@start")
            # print(f"Contract JSON: {contract_str}")
            # print(f"Literature Summary: {evidence}")
            # print(f"Feedback: {feedback}")
            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@End")

            full_prompt = PROMPTS[target_key].format(
                contract_json=contract_str, 
                literature_summary=evidence, 
                feedback=feedback or "None"
            )
            
            capture = [None]
            async for e in run_step(target_agent, full_prompt, capture): yield e
            raw_advice = capture[0]

            # Explainer (Task Prompt 사용)
            capture = [None]
            exp_prompt = PROMPTS["explainer_task"].format(target_role=role, advisor_output=raw_advice)
            async for e in run_step("explainer", exp_prompt, capture): yield e
            final_output_text = capture[0]

            # QA (Task Prompt 사용)
            capture = [None]
            qa_prompt = PROMPTS["qa_task"].format(final_output=final_output_text)
            async for e in run_step("qa", qa_prompt, capture): yield e
            # print(f"QA디버깅@@@@ 본체@@@@@@@@@@: {type(capture)}, {capture}")
            # print(f"QA디버깅@@@@: {type(capture[0])}, {capture[0]}")
            if capture[0] is None: 
                # print("QA디버깅@@@@의답 None@@@")
                continue

            elif "PASS" in capture[0].upper() or "VALID" in capture[0].upper() or "APPROVED" in capture[0].upper():
                qa_status = "PASS"
                # print(f"QA디버깅@@@@의답 pass@@@: {capture[0]}")
                break
            else:
                print(f"❌ QA Failed")
                
                feedback = capture[0].replace("FAIL:", "").strip()
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

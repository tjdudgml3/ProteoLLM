import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

import config
import tools
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.genai import types

# --- 1. Experiment-Analysis Contract Schema (Pydantic) ---
# PDF  내용을 바탕으로 구조화

class Replicates(BaseModel):
    biological: int = Field(..., description="Number of biological replicates [cite: 41]")
    technical: int = Field(default=0, description="Number of technical replicates [cite: 42]")

class Enrichment(BaseModel):
    used: bool = Field(..., description="Whether enrichment was used [cite: 46]")
    method: Optional[str] = Field(None, description="e.g., TiO2, IMAC [cite: 47]")

class CurrentDataStatus(BaseModel):
    phosphosite_count: Optional[int] = Field(0, description="Current count of phosphosites [cite: 50]")
    localization_threshold: float = Field(0.75, description="Localization probability threshold [cite: 53]")
    qc_issues: List[str] = Field(default_factory=list, description="e.g., low MS2 coverage [cite: 56]")

class ExperimentAnalysisContract(BaseModel):
    role: str = Field(..., description="'experimenter' (Wet Lab) or 'analyst' (Bioinformatics) [cite: 24]")
    biological_goal: str = Field(..., description="Biological objective (e.g., EGF signaling) [cite: 25]")
    assay_type: str = Field(..., description="e.g., Label-free DDA, TMT [cite: 26]")
    organism: str = Field(..., description="e.g., Human, Mouse [cite: 27]")
    sample_type: str = Field(..., description="e.g., HeLa cells [cite: 28]")
    comparison: List[str] = Field(..., description="Conditions to compare [cite: 29]")
    instrument: str = Field(..., description="MS Instrument used [cite: 39]")
    replicates: Replicates
    enrichment: Enrichment
    current_data_status: CurrentDataStatus
    analyst_wants: List[str] = Field(default_factory=list, description="Requirements from Analyst [cite: 60]")
    experimenter_wants: List[str] = Field(default_factory=list, description="Requirements from Experimenter [cite: 65]")
    notes: Optional[str] = Field(None, description="Free text notes [cite: 72]")

# --- 2. Dynamic Prompt Templates ---
# {} Placeholder를 사용하여 실행 시점에 데이터를 주입합니다.

PROMPTS = {
    "router": """
    You are a Router Agent.
    Determine if the user is an 'experimenter' (Wet Lab) or an 'analyst' (Bioinformatics)[cite: 77].
    Output ONLY one word: 'experimenter' or 'analyst'.
    """,

    "contract_builder": f"""
    You are a Contract Builder Agent[cite: 78].
    Extract information to fill the `ExperimentAnalysisContract`.
    
    Target Schema:
    {ExperimentAnalysisContract.model_json_schema()}
    
    If critical info is missing, use null/empty defaults. Output ONLY JSON.
    """,

    "literature": """
    You are a Literature Retrieval Agent[cite: 79].
    Search for papers relevant to:
    - Biological Goal: {biological_goal}
    - Assay Type: {assay_type}
    
    Focus on methods and experimental setups.
    """,

    "evidence_qa": """
    You are a Paper Evidence QA Agent[cite: 80].
    Evaluate these search results for relevance and evidence quality:
    
    [SEARCH RESULTS]
    {search_results}
    
    Output a list of recommended papers with rationales.
    """,

    "exp_advisor": """
    You are an Experimental Advisor Agent[cite: 82].
    
    [CONTEXT: CONTRACT]
    {contract_json}
    
    [CONTEXT: LITERATURE EVIDENCE]
    {literature_summary}
    
    Task:
    1. Suggest experimental optimizations (enrichment, lysis, instrument settings) to increase signals[cite: 16].
    2. Create a data hand-over template for the analyst[cite: 17].
    """,

    "analyst_advisor": """
    You are an Analyst Advisor Agent[cite: 81].
    
    [CONTEXT: CONTRACT]
    {contract_json}
    
    [CONTEXT: LITERATURE EVIDENCE]
    {literature_summary}
    
    Task:
    1. Propose an analysis plan (software, DBs, statistical tests)[cite: 13].
    2. Suggest QC checks (missed cleavage, localization scores).
    3. List questions to ask the experimenter regarding metadata.
    """,

    "explainer": """
    You are a Contract Explainer Agent[cite: 83].
    Translate the technical plan for a: {target_role}.
    
    [TECHNICAL PLAN]
    {advisor_output}
    
    Explain this clearly in natural language suited for the role.
    """,

    "qa": """
    You are a QA Agent[cite: 84].
    Critique the following output for Consistency, Feasibility, and Clarity.
    
    [OUTPUT TO REVIEW]
    {final_output}
    
    Return "PASS" or "FAIL: [Reason]".
    """
}

# --- 3. Agent Factory ---

def create_agents():
    """
    Initializes agents with appropriate models (High reasoning vs Fast) as per spec.
    """
    retry_config = types.HttpRetryOptions(
        attempts=config.RETRY_ATTEMPTS,
        exp_base=2,
        initial_delay=config.RETRY_DELAY,
        http_status_codes=[429, 500, 503, 504],
    )

    # Gemini 3 (High Reasoning) [cite: 2]
    model_high = Gemini(model=config.MODEL_HIGH_REASONING, retry_options=retry_config)
    # Gemini Flash (Fast/Lightweight) 
    model_fast = Gemini(model=config.MODEL_FAST, retry_options=retry_config)

    agents = {}

    # Static Agents (Fixed Instructions)
    agents["router"] = LlmAgent(model=model_fast, name="router", instruction=PROMPTS["router"])
    agents["contract_builder"] = LlmAgent(model=model_high, name="contract_builder", instruction=PROMPTS["contract_builder"])
    
    # Dynamic Agents (Instruction templates to be formatted at runtime)
    # Note: LlmAgent를 생성할 때 instruction에 placeholder가 있으면 invoke 전에 포맷팅하거나,
    # 여기서는 모델 인스턴스와 템플릿만 저장해두고 Orchestrator에서 조립하는 방식을 권장합니다.
    
    # For demonstration, we initialize them with base templates. 
    # Actual injection happens in the execution flow.
    agents["literature_model"] = model_fast # Uses tools
    agents["evidence_qa_model"] = model_high
    agents["exp_advisor_model"] = model_high
    agents["analyst_advisor_model"] = model_high
    agents["explainer_model"] = model_fast
    agents["qa_model"] = model_high

    return agents

# --- 4. Orchestrator Example (How to inject {}) ---

class BioinformaticsPipeline:
    def __init__(self, agents: Dict[str, Any]):
        """
        PDF의 '제어 흐름'을 담당하는 오케스트레이터입니다.
        단순 순차 실행이 아니라, Role에 따른 분기(Branching)와
        Context 주입(Data Injection)을 관리합니다.
        """
        self.router = agents['router']
        self.contract_builder = agents['contract_builder']
        self.literature = agents['literature'] # tools 포함
        self.evidence_qa = agents['evidence_qa']
        
        # Advisors (Model만 가지고 있거나, 동적 프롬프트가 필요한 에이전트들)
        self.exp_advisor_model = agents['exp_advisor_model']
        self.analyst_advisor_model = agents['analyst_advisor_model']
        self.explainer = agents['explainer'] # Model or Agent
        self.qa = agents['qa']

    def run(self, user_query: str) -> Dict[str, Any]:
        print(f"🚀 Pipeline Started: {user_query}")
        
        # --- Step 1: Router (Role 판단) ---
        # [cite: 77, 87, 105] 실험자/분석가 구분
        role = self.router.invoke(user_query).text.strip()
        print(f"📍 Role Identified: {role}")

        # --- Step 2: Contract Builder (계약서 작성) ---
        # [cite: 78, 89, 107] 정보 추출 및 JSON 생성
        contract_json = self.contract_builder.invoke(f"Extract info from: {user_query}").text
        # Pydantic을 통한 검증 (실제 구현시 try-except 권장)
        contract_data = json.loads(contract_json) 
        contract_str = json.dumps(contract_data, indent=2)

        # --- Step 3: Literature & Evidence (논문 검색 및 검증) ---
        # [cite: 79, 93, 110]
        # Contract 정보를 바탕으로 검색 쿼리 최적화
        search_query = f"{contract_data.get('assay_type', 'proteomics')} {contract_data.get('biological_goal', '')}"
        search_results = self.literature.invoke(search_query).text
        
        # [cite: 80, 94, 111] 검색 결과 평가
        evidence_summary = self.evidence_qa.invoke(
            f"Review these papers: {search_results}"
        ).text

        # --- Step 4: Advisor (Branching Logic) ---
        # [cite: 81, 82, 99, 112]
        # 여기가 단순 SequentialAgent로는 불가능한 '분기' 및 '다중 입력 주입' 구간입니다.
        
        print(f"🧠 Running Advisor for {role}...")
        
        if role == 'experimenter':
            # 실험자용 프롬프트: Contract + Literature 주입
            prompt = PROMPTS["experimental_advisor"].format(
                contract_context=contract_str,
                literature_context=evidence_summary
            )
            raw_advice = self.exp_advisor_model.generate_content(prompt).text
            
        else: # analyst
            # 분석가용 프롬프트: Contract + Literature 주입
            prompt = PROMPTS["analyst_advisor"].format(
                contract_context=contract_str,
                literature_context=evidence_summary
            )
            raw_advice = self.analyst_advisor_model.generate_content(prompt).text

        # --- Step 5: Explainer ---
        # [cite: 83, 101, 115] 기술적 조언을 사용자 언어로 변환
        final_explanation = self.explainer.generate_content(
            PROMPTS["explainer"].format(
                target_role=role,
                advisor_output=raw_advice
            )
        ).text

        # --- Step 6: QA ---
        # [cite: 84, 103, 116] 최종 검수
        qa_result = self.qa.generate_content(
            PROMPTS["qa"].format(final_output=final_explanation)
        ).text

        return {
            "role": role,
            "contract": contract_data,
            "evidence": evidence_summary,
            "final_output": final_explanation,
            "qa_status": qa_result
        }

# --- 사용 예시 (Usage) ---

if __name__ == "__main__":
    # 1. 에이전트 생성 (팩토리 함수)
    agents_dict = create_proteomics_agents()
    
    # 2. 파이프라인 초기화 (원하시는 깔끔한 스타일)
    pipeline = BioinformaticsPipeline(agents=agents_dict)
    
    # 3. 실행
    result = pipeline.run("EGF treated HeLa cells phosphoproteomics signal too low")
    
    print("\n=== Final Report ===")
    print(result["final_output"])
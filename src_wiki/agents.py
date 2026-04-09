"""
agents.py - Wiki 기반 bioinformatics multi-agent pipeline
original_code/agents.py 구조 완전 유지, VectorDB/Summarize → LLM Wiki 검색으로 교체
"""

import json
import re
import asyncio
import os
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent, BaseAgent
from google.adk.models.google_llm import Gemini
from google.adk.agents.callback_context import CallbackContext
from google.adk.runners import InMemoryRunner
from google.genai import types
from google.genai.types import Content, Part
from google.adk.events import Event

import config
import tools

# ==============================================================================
# 1. Retry Config
# ==============================================================================

retry_config = types.HttpRetryOptions(
    attempts=config.RETRY_ATTEMPTS,
    exp_base=2,
    initial_delay=config.RETRY_DELAY,
    http_status_codes=[429, 500, 503, 504],
)

# ==============================================================================
# 2. Data Schemas (Pydantic)
# ==============================================================================

class Replicates(BaseModel):
    biological: int = Field(..., description="Number of biological replicates")
    technical: int = Field(default=0, description="Number of technical replicates")

class SamplePrep(BaseModel):
    used: bool = Field(..., description="Whether enrichment or specific prep was used")
    method: str = Field(..., description="e.g., TiO2, IMAC, TMT Labeling, High-pH Fractionation")

class CurrentDataStatus(BaseModel):
    identification_count: Optional[int] = Field(0)
    localization_threshold: float = Field(0.75)
    qc_issues: List[str] = Field(default_factory=list)

class ExperimentAnalysisContract(BaseModel):
    role: str = Field(..., description="'experimenter' (Wet Lab) or 'analyst' (Bioinformatics)")
    biological_goal: str = Field(..., description="Main biological objective")
    assay_type: str = Field(..., description="e.g., Phospho-proteomics, Label-free DDA")
    organism: str = Field(default="Human")
    sample_type: str = Field(default="Unknown")
    comparison: List[str] = Field(default_factory=list)
    instrument: str = Field(default="Unknown")
    replicates: Replicates = Field(default_factory=lambda: Replicates(biological=3))
    sample_prep: SamplePrep = Field(default_factory=lambda: SamplePrep(used=False, method="None"))
    current_data_status: CurrentDataStatus = Field(default_factory=CurrentDataStatus)
    analyst_wants: List[str] = Field(default_factory=list)
    experimenter_wants: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

# ==============================================================================
# 3. Prompt Templates
# ==============================================================================

PROMPTS = {
    "router_system": """
You are a Router Agent. Classify the user's input.
Output ONLY one word: experimenter or analyst.
- experimenter: wet lab, protocol, sample prep
- analyst: bioinformatics, data analysis, pipeline
""",

    "contract_builder_system": f"""
You are a Contract Builder Agent. Extract experiment details.
Target schema: {ExperimentAnalysisContract.model_json_schema()}
Output ONLY valid JSON. Use sensible defaults if info is missing.
""",

    "literature_system": """
You are a Search Query Generator for a scientific knowledge base.
Convert the user's intent into 3-5 precise English keywords.
Output ONLY the keyword string. No explanation, no code blocks.
""",

    "filter_system": """
You are a Research Relevance Filter.
Select up to 5 file_paths from the candidate list most relevant to the query.
Output ONLY a JSON list of file_path strings.
Use EXACT file_path strings from the input.
Example: ["/path/to/file1.md", "/path/to/file2.md"]
""",

    "exp_advisor_system": """
You are an Experimental Advisor (Wet Lab Specialist).
Provide protocol recommendations with citations from the provided wiki pages.
Cite sources using [SourceFilename] format at the end of relevant sentences.
""",

    "analyst_advisor_system": """
You are a Bioinformatics Analyst Advisor.
Design analysis pipelines with citations from the provided wiki pages.
Cite sources using [SourceFilename] format at the end of relevant sentences.
""",

    "explainer_system": """
You are a Science Communicator.
Translate technical content into clear language while preserving citations.
Match the language of the user's original query (Korean query = Korean response).
Always include a 'Key References' section at the bottom.
""",

    "qa_system": """
You are a QA Agent. Check if the output contains citations/references.
If citations present: output only "PASS"
If citations missing: output "FAIL: Missing References"
""",

    # ---------- Task Templates ----------
    "filter_task": """
[USER QUERY]
{query}

[CANDIDATE WIKI PAGES]
{candidates_json}

Select up to 5 file_path strings most relevant to the query.
Return ONLY a JSON list. Example: ["/path/a.md", "/path/b.md"]
""",

    "exp_advisor_task": """
[USER QUERY]
{user_query}

[EXPERIMENT CONTRACT]
{contract_json}

[WIKI KNOWLEDGE BASE]
{literature_summary}

[QA FEEDBACK]
{feedback}

Provide step-by-step protocol recommendations using ONLY the <WikiPage> content above.
Cite each fact with [SourceFilename]. Answer in same language as user query.

## Protocol Recommendation
## Rationale
## Key References
""",

    "analyst_advisor_task": """
[USER QUERY]
{user_query}

[EXPERIMENT CONTRACT]
{contract_json}

[WIKI KNOWLEDGE BASE]
{literature_summary}

[QA FEEDBACK]
{feedback}

Design an analysis pipeline using ONLY the <WikiPage> content above.
Cite each fact with [SourceFilename]. Answer in same language as user query.

## Pipeline Design
## QC Checklist
## Key References
""",

    "explainer_task": """
[USER QUERY]
{user_query}

[TECHNICAL PLAN]
{advisor_output}

Explain for a {target_role} audience.
Keep parameters but simplify jargon. Preserve citations [SourceX].
Include a 'Key References' section. Answer in same language as user query.
""",

    "qa_task": """
Check if this output contains citations/references like [SourceX].

[OUTPUT]
{final_output}

Output "PASS" if citations exist, "FAIL: Missing References" if not.
"""
}

# ==============================================================================
# 4. Logging
# ==============================================================================

def save_log(message: str):
    with open("response_time.txt", "a", encoding="utf-8") as f:
        f.write(message + "\n")

# ==============================================================================
# 5. Agent Factory
# ==============================================================================

def create_agents() -> Dict[str, LlmAgent]:
    model_fast = Gemini(model=config.MODEL_FAST, retry_options=retry_config)
    model_high = Gemini(model=config.MODEL_HIGH_REASONING, retry_options=retry_config)

    return {
        "router":           LlmAgent(model=model_fast, name="router",           instruction=PROMPTS["router_system"]),
        "contract_builder": LlmAgent(model=model_fast, name="contract_builder", instruction=PROMPTS["contract_builder_system"]),
        "literature":       LlmAgent(model=model_fast, name="literature",       instruction=PROMPTS["literature_system"]),
        "filter":           LlmAgent(model=model_fast, name="filter",           instruction=PROMPTS["filter_system"]),
        "exp_advisor":      LlmAgent(model=model_high, name="exp_advisor",      instruction=PROMPTS["exp_advisor_system"]),
        "analyst_advisor":  LlmAgent(model=model_high, name="analyst_advisor",  instruction=PROMPTS["analyst_advisor_system"]),
        "explainer":        LlmAgent(model=model_fast, name="explainer",        instruction=PROMPTS["explainer_system"]),
        "qa":               LlmAgent(model=model_fast, name="qa",               instruction=PROMPTS["qa_system"]),
    }

# ==============================================================================
# 6. Pipeline (BaseAgent) — 원본 original_code/agents.py 구조 완전 유지
# ==============================================================================

class BioinformaticsPipeline(BaseAgent):
    agents:  Dict[str, Any] = Field(..., description="Dictionary of sub-agents")
    plugins: List[Any]       = Field(default_factory=list, description="List of plugins")

    def __init__(self, agents: Dict[str, Any]):
        super().__init__(name="bioinformatics_pipeline", agents=agents)
        self.description = "A comprehensive bioinformatics pipeline using LLM Wiki."

    def _extract_json(self, text: str) -> Any:
        """JSON 파싱 및 에러 처리 (원본 로직 유지)"""
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
            # filepath 패턴 fallback (Wiki 버전 추가)
            paths = re.findall(r'(/[^\s\'"`,\]]+\.md)', text)
            if paths: return list(dict.fromkeys(paths))[:5]
            return {}
        except Exception:
            return {}

    def _parse_response_text(self, response) -> str:
        """Runner의 결과(Events List)에서 최종 텍스트 추출 (원본 로직 유지)"""
        try:
            if isinstance(response, list) and len(response) > 0:
                last_event = response[-1]
                if hasattr(last_event, 'content') and last_event.content and last_event.content.parts:
                    return last_event.content.parts[0].text or ""
            if hasattr(response, 'content') and response.content and response.content.parts:
                return response.content.parts[0].text or ""
        except Exception as e:
            print(f"⚠️ Text Parsing Warning: {e}")
        return ""

    async def _invoke_sub_agent(self, agent_name: str, prompt: str) -> str:
        """InMemoryRunner를 사용하여 서브 에이전트에게 프롬프트를 전달 (원본 로직 유지)"""
        runner = InMemoryRunner(agent=self.agents[agent_name], plugins=self.plugins)
        response_events = await runner.run_debug(prompt)
        return self._parse_response_text(response_events)

    async def _run_async_impl(self, context: CallbackContext):
        save_log("--------------new attempt (Wiki Mode)--------------")

        # 입력 데이터 추출 (원본과 동일)
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
        save_log(f"⏱️ [Time] Router: {time.perf_counter() - t_start:.2f}s")

        # ------------------------------------------------------------------
        # 2. Contract Builder
        # ------------------------------------------------------------------
        t_start = time.perf_counter()
        contract_text = await self._invoke_sub_agent("contract_builder", f"Analyze: {input_data}")
        contract_data = self._extract_json(contract_text)
        if not contract_data:
            contract_data = {"assay_type": "proteomics", "biological_goal": input_data, "role": role}
        contract_str = json.dumps(contract_data, indent=2, ensure_ascii=False)
        save_log(f"⏱️ [Time] Contract Builder: {time.perf_counter() - t_start:.2f}s")

        # ------------------------------------------------------------------
        # 3. Reasoning Loop (원본 구조 그대로)
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
            save_log(f"⏱️ [Time] Literature (Query Gen): {time.perf_counter() - t_start:.2f}s")

            # B. Wiki Search (FAISS 대신 LLM 기반 — tools.py에서 처리)
            t_start = time.perf_counter()
            raw_candidates_json = tools.search_wiki_with_llm(search_query, k=20)
            candidates = json.loads(raw_candidates_json)

            if not candidates:
                print("No wiki pages found. Retrying...")
                feedback = "Search query returned no results. Make it broader."
                current_retry += 1
                continue
            save_log(f"⏱️ [Time] Wiki Search: {time.perf_counter() - t_start:.2f}s")

            # C. Filter Agent (원본과 동일한 구조)
            filter_prompt = PROMPTS["filter_task"].format(
                query=search_query,
                candidates_json=raw_candidates_json
            )
            with open("./temp_full_text.txt", "w", encoding="utf-8") as f:
                f.write(filter_prompt)

            t_start = time.perf_counter()
            filter_output = await self._invoke_sub_agent("filter", filter_prompt)
            with open("./temp_full_text.txt", "a", encoding="utf-8") as f:
                f.write(filter_output)
            save_log(f"⏱️ [Time] Filter: {time.perf_counter() - t_start:.2f}s")

            try:
                extracted_data = self._extract_json(filter_output)
                if isinstance(extracted_data, list) and len(extracted_data) > 0:
                    selected_paths = extracted_data
                    print(f"Filter Selected: {len(selected_paths)} pages")
                else:
                    raise ValueError("Output is not a valid list")
            except:
                print("Filter parsing failed. Using top-5 fallback.")
                selected_paths = [c['file_path'] for c in candidates[:5]]

            # ID 검증 (원본의 ID 검증 로직과 동일한 역할)
            valid_path_set = {c['file_path'] for c in candidates}
            validated_paths = [p for p in selected_paths if p in valid_path_set]
            if not validated_paths:
                validated_paths = [c['file_path'] for c in candidates[:5]]
            selected_paths = validated_paths
            print(f"Final Paths: {selected_paths}")

            # D. Full Text Loading (원본의 full_text_builder 역할)
            t_start = time.perf_counter()
            full_text_context = tools.get_full_wiki_pages(selected_paths)
            with open("./temp_full_text.txt", "a", encoding="utf-8") as f:
                f.write("****************full text context*********\n")
                f.write(full_text_context)
            save_log(f"⏱️ [Time] Wiki Full Load: {time.perf_counter() - t_start:.2f}s")

            # E. Advisor Agent
            target_agent = "exp_advisor" if role == 'experimenter' else "analyst_advisor"
            target_key   = "exp_advisor_task" if role == 'experimenter' else "analyst_advisor_task"
            t_start = time.perf_counter()
            advisor_prompt = PROMPTS[target_key].format(
                contract_json=contract_str,
                literature_summary=full_text_context,
                feedback=feedback or "None",
                user_query=input_data
            )
            raw_advice = await self._invoke_sub_agent(target_agent, advisor_prompt)
            save_log(f"⏱️ [Time] Advisor: {time.perf_counter() - t_start:.2f}s")

            # F. Explainer Agent
            t_start = time.perf_counter()
            exp_prompt = PROMPTS["explainer_task"].format(
                target_role=role,
                advisor_output=raw_advice,
                user_query=input_data
            )
            final_output_text = await self._invoke_sub_agent("explainer", exp_prompt)
            save_log(f"⏱️ [Time] Explainer: {time.perf_counter() - t_start:.2f}s")

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
            save_log(f"⏱️ [Time] QA: {time.perf_counter() - t_start:.2f}s")

        if qa_status == "FAIL":
            final_output_text = f"[Max Retries Reached]\n{final_output_text}"
            save_log("QA Failed")

        final_event = Event(
            author="model",
            content=Content(parts=[Part(text=final_output_text)])
        )

        yield final_event


def get_pipeline(api_key: str = None) -> BioinformaticsPipeline:
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    agents = create_agents()
    return BioinformaticsPipeline(agents)

"""
tools.py - LLM Wiki 기반 검색 도구
FAISS/VectorDB 없이, frontmatter 기반 경량 JSON 인덱스를 LLM이 읽고 선택합니다.
"""

import os
import json
import re
import pickle
import google.genai as genai

WIKI_DIR = "/home/07seoy/biollm/llm_wiki_h200"
SOURCES_DIR = os.path.join(WIKI_DIR, "Sources")
LITE_INDEX_PATH = os.path.join(WIKI_DIR, "wiki_lite_index.pkl")


# ==============================================================================
# 1. 경량 인덱스 빌드 (frontmatter만 파싱)
# ==============================================================================

def _parse_frontmatter(filepath: str) -> dict:
    """마크다운 파일의 YAML frontmatter를 파싱하여 딕셔너리로 반환합니다."""
    meta = {"file_path": filepath, "title": "", "year": "", "entities": []}
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(1500)  # 첫 1500자만 읽어 frontmatter 파싱

        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                fm_block = content[3:end].strip()
                # title 추출
                t = re.search(r'^title:\s*["\']?(.*?)["\']?\s*$', fm_block, re.MULTILINE)
                if t:
                    meta["title"] = t.group(1).strip()[:200]  # 최대 200자
                # year 추출
                y = re.search(r'^year:\s*(\d{4})', fm_block, re.MULTILINE)
                if y:
                    meta["year"] = y.group(1)
                # entities 추출 (간단히 단어 리스트로)
                e = re.search(r'^entities:\s*\[(.+?)\]', fm_block, re.MULTILINE | re.DOTALL)
                if e:
                    raw = e.group(1)
                    ents = re.findall(r'\[\[\s*(\w+)\s*\]\]', raw)
                    meta["entities"] = ents[:10]  # 최대 10개
    except Exception:
        pass
    return meta


def build_lite_index(force_rebuild: bool = False) -> list:
    """
    Sources/ 폴더의 모든 .md 파일에서 frontmatter만 파싱해
    경량 인덱스(리스트)를 만들고 pkl로 캐시합니다.
    """
    if not force_rebuild and os.path.exists(LITE_INDEX_PATH):
        print("Loading cached lite index...")
        with open(LITE_INDEX_PATH, "rb") as f:
            return pickle.load(f)

    print("Building lite index from Sources/ frontmatter...")
    index = []
    for fname in os.listdir(SOURCES_DIR):
        if not fname.endswith(".md"):
            continue
        fpath = os.path.join(SOURCES_DIR, fname)
        meta = _parse_frontmatter(fpath)
        index.append(meta)

    print(f"Indexed {len(index)} sources.")
    with open(LITE_INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    return index


# ==============================================================================
# 2. LLM 기반 Wiki 검색 (Gemini Flash)
# ==============================================================================

def search_wiki_with_llm(query: str, k: int = 20) -> str:
    """
    경량 인덱스를 Gemini Flash에 넘겨 쿼리와 관련된 파일 경로를 선택받습니다.
    Returns: JSON string - list of {source, file_path, title} dicts
    """
    lite_index = build_lite_index()

    # 인덱스가 너무 크면 LLM 컨텍스트에 모두 넣을 수 없으므로
    # title/entities 기반 빠른 키워드 pre-filter 수행 (Python, 빠름)
    query_words = set(re.findall(r'\w+', query.lower()))
    
    scored = []
    for item in lite_index:
        text = (item["title"] + " " + " ".join(item["entities"])).lower()
        score = sum(1 for w in query_words if w in text)
        if score > 0:
            scored.append((score, item))
    
    # 상위 100개만 LLM에 전달
    scored.sort(key=lambda x: -x[0])
    top_candidates = [item for _, item in scored[:100]]

    if not top_candidates:
        # 키워드 매칭 없으면 아무거나 100개 보냄
        top_candidates = lite_index[:100]

    # LLM 선택용 간결한 표현으로 변환
    compact_list = []
    for i, item in enumerate(top_candidates):
        compact_list.append({
            "idx": i,
            "file_path": item["file_path"],
            "title": item["title"][:180],
            "year": item["year"],
            "entities": item["entities"]
        })

    prompt = f"""You are a research document selector.

Query: "{query}"

Below is a numbered list of research paper summaries.
Select the {k} most relevant papers to the query.

CANDIDATES:
{json.dumps(compact_list, ensure_ascii=False, indent=1)}

INSTRUCTIONS:
- Return ONLY a JSON list of file_path strings.
- Pick exactly {k} (or fewer if not enough candidates).
- Prefer papers directly related to the query topic.
- Example output: ["/path/a.md", "/path/b.md"]
"""

    try:
        import config
        client = genai.Client(api_key=config.GOOGLE_API_KEY)
        response = client.models.generate_content(
            model=config.MODEL_FAST,
            contents=prompt,
        )
        raw = response.text.strip()

        # JSON 파싱
        # ```json 블록 제거
        if "```" in raw:
            raw = re.sub(r'```(?:json)?', '', raw).replace('```', '').strip()

        paths = json.loads(raw)
        if not isinstance(paths, list):
            raise ValueError("Not a list")

        results = []
        path_set = {item["file_path"] for item in compact_list}
        for p in paths:
            if p in path_set:
                results.append({
                    "source": os.path.basename(p),
                    "file_path": p,
                    "snippet": ""  # 실제 내용은 get_full_wiki_pages에서 로드
                })
        return json.dumps(results, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"LLM search failed: {e}. Falling back to keyword top-{k}.")
        fallback = []
        for item in top_candidates[:k]:
            fallback.append({
                "source": os.path.basename(item["file_path"]),
                "file_path": item["file_path"],
                "snippet": item["title"]
            })
        return json.dumps(fallback, ensure_ascii=False, indent=2)


# ==============================================================================
# 3. 선택된 Wiki 페이지 전체 텍스트 로드
# ==============================================================================

def get_full_wiki_pages(selected_filepaths: list) -> str:
    """선택된 마크다운 파일들의 전체 내용을 읽어 하나의 문자열로 합칩니다."""
    final_text = ""
    for path in selected_filepaths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                final_text += f"\n<WikiPage source='{os.path.basename(path)}'>\n"
                final_text += content
                final_text += "\n</WikiPage>\n"
            except Exception as e:
                print(f"Error reading {path}: {e}")

    return final_text if final_text else "No wiki text available."

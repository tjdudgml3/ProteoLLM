import json
import os
from vector_db import VectorDB  # 위에서 작성한 VectorDB 클래스

# 전역 DB 인스턴스 (한 번만 로드)
import json
import os
import re
from vector_db import VectorDB

# 전역 DB 인스턴스
db = VectorDB()

# ID 정규화 함수 (PMC 접두어 제거 및 공백 제거)
def normalize_id(pmc_id):
    """
    PMC12345 -> 12345
    12345 -> 12345
    """
    return re.sub(r'[^0-9]', '', str(pmc_id))

def search_vectors(query: str, k: int = 40) -> str:
    results = db.search(query, k=k)
    candidates = []
    for meta in results:
        try:
            with open(meta['file_path'], 'r', encoding='utf-8') as f:
                paper = json.load(f)
                candidates.append({
                    "pmcid": paper.get("pmcid", "Unknown"),
                    "year": paper.get("year", "Unknown"),
                    "abstract": paper.get("abstract", "")[:1000],
                    "methods_snippet": paper.get("methods", "")[:1000],
                    "file_path": meta['file_path']
                })
        except Exception as e:
            continue
    return json.dumps(candidates, indent=2, ensure_ascii=False)

def get_full_text_by_ids(selected_pmcids: list, all_candidates_json: str) -> str:
    """
    선택된 PMCID 리스트를 받아, 전체 본문(Full Body)을 로드하여 반환합니다.
    [수정사항]
    1. ID 매칭 로직 개선 (숫자 기반 비교)
    2. 디버깅 로그 추가 (왜 파일을 못 찾았는지 출력)
    """
    print(f"Loading full text for IDs: {selected_pmcids}")
    
    candidates = json.loads(all_candidates_json)
    final_text = ""
    
    id_to_path = {}
    for c in candidates:
        raw_id = c.get('pmcid', '')
        norm_id = normalize_id(raw_id)
        if norm_id:
            id_to_path[norm_id] = c['file_path']
            
    success_count = 0
    
    for target_pmcid in selected_pmcids:
        target_norm = normalize_id(target_pmcid)
        
        path = id_to_path.get(target_norm)
        
        if not path:
            print(f"Warning: Could not find path for ID '{target_pmcid}' (Norm: {target_norm}) in candidates.")
            continue
            
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    paper = json.load(f)
                    
                    # 텍스트 포맷팅 (XML 구조)
                    final_text += f"""
                    <Source ID="{target_pmcid}">
                        <Meta>Title: {paper.get('pmcid')} (Year: {paper.get('year')})</Meta>
                        <Content_Methods>
                        {paper.get('methods', 'No methods section')[:5000]}
                        </Content_Methods>
                        <Content_Body>
                        {paper.get('body', 'No body text')[:10000]}
                        </Content_Body>
                    </Source>
                    """
                    success_count += 1
            except Exception as e:
                print(f"Error reading file {path}: {e}")
        else:
            print(f"File not found at path: {path}")

    if success_count == 0:
        print("CRITICAL: No full text loaded! Check ID format matching.")
        # 비상용 메시지 리턴 (프롬프트가 비어있지 않게)
        return "No full text available. The retrieval system failed to load document content."
        
    print(f"Successfully loaded {success_count} documents.")
    return final_text



def get_paper_by_id(pmcid: str, all_candidates_json: str) -> str:
    candidates = json.loads(all_candidates_json)
    final_text = ""
    id_to_path = {}
    for c in candidates:
        raw_id = c.get('pmcid', '')
        norm_id = normalize_id(raw_id)
        if norm_id:
            id_to_path[norm_id] = c['file_path']
    
    target_norm = normalize_id(pmcid)
    path = id_to_path.get(target_norm)
    
    if not path:
        print(f"Warning: Could not find path for ID '{pmcid}' (Norm: {target_norm}) in candidates.")
        return ""
    
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                paper = json.load(f)
                final_text += f"""
                <Source ID="{pmcid}">
                    <Meta>Title: {paper.get('pmcid')} (Year: {paper.get('year')})</Meta>
                    <Content_Methods>
                    {paper.get('methods', 'No methods section')[:5000]}
                    </Content_Methods>
                    <Content_Body>
                    {paper.get('body', 'No body text')[:10000]}
                    </Content_Body>
                </Source>
                """
                return final_text
        except Exception as e:
            print(f"Error reading file {path}: {e}")
    else:
        print(f"File not found at path: {path}")
    
    return ""
    


def full_text_builder(papers: list)-> str:
    final_text = ""
    # print("papers--------json----------")
    # print(papers)
    for paper1 in papers:
        paper = json.loads(paper1)
        # print("paper--------json----------")
        # print(paper)
        paper_content = paper.get('contents')
        if paper_content:
            paper_content = paper_content[:5000]
        else:
            paper_content = ""
        final_text += f"""
                <Source ID="{paper.get('pmcid')}">
                    <Meta>Title: {paper.get('pmcid')}</Meta>
                    <summarized_content>
                    
                    {paper_content}

                    </summarized_content>
                </Source>\n
                """
    return final_text
        

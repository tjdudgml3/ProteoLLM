import config
import os
import glob
import json
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class VectorDB:
    def __init__(self, data_dir=config.DATA_DIR):
        self.data_dir = data_dir
        self.index_path = os.path.join(data_dir, "faiss_index.bin")
        self.docs_path = os.path.join(data_dir, "documents.pkl")
        
        # 모델 로드 
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "all-MiniLM-L6-v2")
        if os.path.exists(model_path):
            self.model = SentenceTransformer(model_path)
        else:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
        self.index = None
        self.documents = [] 
        self.metadata = [] 
        
        if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
            self._load_index()
        else:
            self._build_index()

    def _load_index(self):
        print("Loading existing vector index...")
        self.index = faiss.read_index(self.index_path)
        with open(self.docs_path, "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
        print(f"Loaded {len(self.documents)} chunks.")

    def _build_index(self):
        print("Building new vector index from JSON files...")
        # 2020년부터 2025년
        years = [str(y) for y in range(2020, 2026)]
        
        all_chunks = []
        all_meta = []
        
        for year in years:
            # json 파일만 타겟팅
            path = os.path.join(self.data_dir, year, "*.json")
            files = glob.glob(path)
            print(f"Processing {year}: {len(files)} files found.")
            
            for filepath in tqdm(files, desc=f"Indexing {year}"):
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        data = json.load(f)
                        
                    abstract = data.get("abstract", "")
                    methods = data.get("methods", "")
                    
                    # 내용이 너무 짧으면 스킵
                    if len(abstract) < 50 and len(methods) < 50:
                        continue

                    # 검색용 텍스트 생성 (Abstract + Methods)
                    combined_text = f"Abstract: {abstract}\nMethods: {methods}"
                    
                    # Chunking
                    chunks = self._chunk_text(combined_text)
                    
                    for chunk in chunks:
                        all_chunks.append(chunk)
                        all_meta.append({
                            "source": os.path.basename(filepath),
                            "file_path": filepath, 
                            "pmcid": data.get("pmcid", "Unknown"),
                            "year": year
                        })
                        
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

        if not all_chunks:
            all_chunks = ["No data."]
            all_meta = [{"source": "dummy", "file_path": "", "pmcid": "0", "year": "0"}]


        print("Encoding documents...")
        batch_size = 32
        embeddings = self.model.encode(all_chunks, batch_size=batch_size, show_progress_bar=True)
        embeddings = np.array(embeddings)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        self.documents = all_chunks
        self.metadata = all_meta
        
        # 저장
        print("Saving index to disk...")
        faiss.write_index(self.index, self.index_path)
        with open(self.docs_path, "wb") as f:
            pickle.dump({"documents": self.documents, "metadata": self.metadata}, f)

    def _chunk_text(self, text, chunk_size=1000, overlap=100):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += (chunk_size - overlap)
        return chunks

    def search(self, query, k=40):
        """
        Top-k 개의 유니크한 '파일'을 찾기 위해 검색 개수를 넉넉히 잡습니다.
        청크 단위 검색이므로 한 논문에서 여러 청크가 나올 수 있기 때문입니다.
        """
        if not self.index: return []
        
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), k * 3)
        
        unique_papers = {}
        results = []
        
        for idx in indices[0]:
            if idx < len(self.documents) and idx >= 0:
                meta = self.metadata[idx]
                file_path = meta['file_path']
                
                # 이미 찾은 논문이면 스킵
                if file_path in unique_papers:
                    continue
                
                unique_papers[file_path] = meta
                results.append(meta)
                
                if len(results) >= k:
                    break
        
        return results
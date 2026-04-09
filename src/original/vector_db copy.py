import config
import os
import glob
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
        
        # Load Model
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "all-MiniLM-L6-v2")
        if os.path.exists(model_path):
            print(f"Loading local model from {model_path}")
            self.model = SentenceTransformer(model_path)
        else:
            print("Loading model from Hub (fallback)")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
        self.index = None
        self.documents = [] # List of text chunks
        self.metadata = [] # List of dicts with source info
        
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
        print(f"Loaded {len(self.documents)} documents.")

    def _build_index(self):
        print("Building new vector index from data files...")
        years = ['2022', '2023', '2024', '2025']
        
        all_chunks = []
        all_meta = []
        
        for year in years:
            path = os.path.join(self.data_dir, year, "*.txt")
            files = glob.glob(path)
            print(f"Processing {year}: {len(files)} files found.")
            
            for filepath in tqdm(files, desc=f"Indexing {year}"):
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                        
                    # Simple chunking
                    chunks = self._chunk_text(text)
                    for chunk in chunks:
                        all_chunks.append(chunk)
                        all_meta.append({"source": os.path.basename(filepath), "year": year})
                        
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

        if not all_chunks:
            print("No documents found! Creating dummy index.")
            all_chunks = ["No data found."]
            all_meta = [{"source": "dummy", "year": "none"}]

        # Batch encode
        print("Encoding documents...")
        batch_size = 32
        embeddings = []
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            emb = self.model.encode(batch)
            embeddings.append(emb)
        
        embeddings = np.vstack(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        self.documents = all_chunks
        self.metadata = all_meta
        
        # Save
        print("Saving index to disk...")
        faiss.write_index(self.index, self.index_path)
        with open(self.docs_path, "wb") as f:
            pickle.dump({"documents": self.documents, "metadata": self.metadata}, f)
            
        print(f"Indexed {len(self.documents)} chunks.")

    def _chunk_text(self, text, chunk_size=1000, overlap=100):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += (chunk_size - overlap)
        return chunks

    def search(self, query, k=10):
        if not self.index:
            return []
        
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx >= 0:
                doc = self.documents[idx]
                meta = self.metadata[idx]
                source_name = meta['source'].replace('.txt', '')
                results.append(f"[Source: {source_name}]\n{doc}")
        
        return results

import config
from vector_db import VectorDB
import time

def main():
    print("Initializing VectorDB...")
    start_time = time.time()
    db = VectorDB()
    end_time = time.time()
    print(f"VectorDB initialized in {end_time - start_time:.2f} seconds.")
    
    query = "phosphoproteomics breast cancer"
    print(f"\nSearching for: '{query}'")
    results = db.search(query, k=3)
    
    print(f"\nFound {len(results)} results:")
    for i, res in enumerate(results):
        print(f"\nResult {i+1}:")
        print(res[:500] + "..." if len(res) > 500 else res)

if __name__ == "__main__":
    main()

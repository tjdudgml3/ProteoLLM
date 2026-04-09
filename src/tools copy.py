from typing import List
from vector_db import VectorDB

# Global instance to avoid reloading
_vector_db = None

def get_vector_db():
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDB()
    return _vector_db

def search_literature(query: str) -> List[str]:
    """
    Searches for relevant scientific literature based on the query.
    
    Args:
        query: The search query string (e.g., "phosphoproteomics targets in breast cancer").
        
    Returns:
        A list of strings, where each string is a relevant document or paper summary.
    """
    db = get_vector_db()
    results = db.search(query)
    return results

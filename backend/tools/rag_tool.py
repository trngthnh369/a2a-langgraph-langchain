from langchain_core.tools import tool
from typing import List, Dict, Any
from backend.data.vector_store import VectorStore

# Global vector store instance
vector_store = VectorStore("products")

@tool
def rag_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
     RAG tool with true vector search capabilities.
    
    Args:
        query: Search query for products
        max_results: Maximum number of results to return
    
    Returns:
        List of relevant products with detailed information
    """
    try:
        # Perform vector search (sync)
        results = vector_store.search(query, k=max_results)
        
        # Format results for agent consumption
        formatted_results = []
        for result in results:
            metadata = result.get('metadata', {})
            
            formatted_result = {
                "content": result.get("content", ""),
                "title": metadata.get("title", ""),
                "price": metadata.get("current_price", ""),
                "specs": metadata.get("product_specs", ""),
                "promotion": metadata.get("product_promotion", ""),
                "colors": metadata.get("color_options", ""),
                "relevance_score": result.get("relevance_score", 0.0),
                "source": "vector_database"
            }
            formatted_results.append(formatted_result)
        
        print(f"🔍 RAG Search: Found {len(formatted_results)} results for '{query}'")
        return formatted_results
        
    except Exception as e:
        print(f"❌ RAG search failed: {e}")
        return []
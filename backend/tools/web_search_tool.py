from langchain_core.tools import tool
from typing import List, Dict, Any
from backend.core.config import settings
import httpx

@tool
def web_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Search the web for current information using Serper API.
    
    Args:
        query: Search query
        max_results: Maximum number of results
    
    Returns:
        List of web search results
    """
    if not settings.serper_api_key or not settings.enable_web_search:
        return [{
            "title": "Web Search Unavailable",
            "snippet": "Web search is currently disabled or API key not configured.",
            "link": "",
            "source": "system"
        }]
    
    try:
        url = "https://google.serper.dev/search"
        headers = {
            'X-API-KEY': settings.serper_api_key,
            'Content-Type': 'application/json'
        }
        payload = {
            'q': query,
            'num': max_results,
            'gl': 'vn',
            'hl': 'vi'
        }
        with httpx.Client(timeout=20.0) as client:
            response = client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                results = data.get('organic', [])
                formatted_results = []
                for result in results[:max_results]:
                    formatted_results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "link": result.get("link", ""),
                        "source": "web_search"
                    })
                print(f"🌐 Web Search: Found {len(formatted_results)} results")
                return formatted_results
            else:
                print(f"❌ Web search API error: {response.status_code}")
                # Return a fallback informative result so the agent can inform the user
                reason = "API forbidden (403). Check SERPER_API_KEY or quota." if response.status_code == 403 else f"HTTP {response.status_code} from search API."
                return [{
                    "title": "Web Search Unavailable",
                    "snippet": f"{reason}",
                    "link": "",
                    "source": "system"
                }]
                    
    except Exception as e:
        print(f"❌ Web search failed: {e}")
        return []
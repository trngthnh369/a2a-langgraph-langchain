from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from collections.abc import AsyncIterable
from typing import Any, Literal, Dict, List
from pydantic import BaseModel
import time

from backend.core.config import settings
from backend.tools.rag_tool import rag_search
from backend.tools.web_search_tool import web_search
from backend.data.cache_manager import CacheManager

import logging

# Initialize logger
logger = logging.getLogger(__name__)

load_dotenv()
memory = MemorySaver()

class ResponseFormat(BaseModel):
    """Response format with more metadata"""
    status: Literal['input_required', 'completed', 'error'] = 'completed'
    message: str
    confidence: float = 1.0
    sources: List[str] = []
    processing_time: float = 0.0

# Initialize cache manager
cache_manager = CacheManager() if settings.enable_caching else None

@tool
def shop_information_rag():
    """
     Shop information tool with structured data.
    """
    shops = [
        {
            "id": 1,
            "name": "Hoàng Hà Mobile - Tam Trinh",
            "address": "89 Đ. Tam Trinh, Mai Động, Hoàng Mai, Hà Nội 100000, Vietnam",
            "maps_url": "https://maps.app.goo.gl/SitTbiYwUpu8jpeRA",
            "opening_hours": "8:30 AM–9:30 PM",
            "phone": "024 3868 7777",
            "services": ["Product consultation", "Warranty repair", "Technical support", "Home delivery"],
            "district": "Hoàng Mai"
        },
        {
            "id": 2,
            "name": "Hoàng Hà Mobile - Nguyễn Công Trứ", 
            "address": "27A Nguyễn Công Trứ, Phạm Đình Hổ, Hai Bà Trưng, Hà Nội 100000, Vietnam",
            "maps_url": "https://maps.app.goo.gl/3L7iSHpbHawsEaTx9",
            "opening_hours": "8:30 AM–9:30 PM",
            "phone": "024 3974 7777",
            "services": ["Product consultation", "Warranty repair", "Express delivery", "Trade-in service"],
            "district": "Hai Bà Trưng"
        },
        {
            "id": 3,
            "name": "Hoàng Hà Mobile - Trương Định",
            "address": "392 Đ. Trương Định, Tương Mai, Hoàng Mai, Hà Nội, Vietnam", 
            "maps_url": "https://maps.app.goo.gl/torAE2bHddW6nMPq9",
            "opening_hours": "8:30 AM–9:30 PM",
            "phone": "024 3636 7777",
            "services": ["Product consultation", "Warranty repair", "Technical support", "Pickup service"],
            "district": "Hoàng Mai"
        }
    ]
    return shops

class LangGraphAgent:
    """LangGraph Agent with RAG, Web Search, and Performance Monitoring"""
    
    SYSTEM_INSTRUCTION = (
        'You are an AI assistant specializing in mobile phones and technology products with advanced capabilities.\n\n'

        ' AVAILABLE TOOLS:\n'
        '1. rag_search: Search product information using vector similarity (TRUE RAG)\n'
        '2. shop_information_rag: Get detailed shop locations and services\n'
        '3. web_search: Search current information from the internet\n\n'
        
        ' TOOL USAGE STRATEGY:\n'
        '- Use rag_search for product information (prices, specs, features).\n'
        '- If RAG yields no results or insufficient confidence, IMMEDIATELY use web_search to answer, without asking for permission.\n'
        '- Use shop_information_rag for store locations, hours, services.\n'
        '- Use web_search for current events, recent news, or information not in knowledge base.\n\n'
        
        ' RESPONSE GUIDELINES:\n'
        '- Always include confidence score based on data quality.\n'
        '- Mention information sources (vector DB, shop data, web search).\n'
        '- Prefer acting autonomously. Only ask follow-up questions when essential (e.g., ambiguous product).\n'
        '- Set status to "input_required" only when absolutely necessary.\n'
        '- Set status to "error" if tools fail or data unavailable.\n'
        '- Set status to "completed" for successful responses.\n\n'
        
        ' FEATURES:\n'
        '- Provide detailed product comparisons when requested\n'
        '- Include pricing and availability information\n'
        '- Suggest related products or alternatives\n'
        '- Give complete shop information with contact details\n'
        '- Use web search for trending or recent information\n\n'
        
        'Always be helpful, accurate, and provide comprehensive information.'
    )
    
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model=settings.chat_model if hasattr(settings, 'chat_model') else 'gemini-2.0-flash',
            google_api_key=settings.google_api_key,
            temperature=0.1
        )
        
        self.tools = [rag_search, shop_information_rag, web_search]
        
        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=ResponseFormat,
        )
        
        print("🚀  LangGraph Agent initialized with RAG + Web Search")
    
    def invoke(self, query: str, sessionId: str) -> Dict[str, Any]:
        """Synchronous invocation with caching"""
        start_time = time.time()
        
        # Check cache first
        if cache_manager:
            cache_key = f"query:{hash(query)}:{sessionId}"
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                cached_result['from_cache'] = True
                cached_result['processing_time'] = time.time() - start_time
                return cached_result
        
        config = {'configurable': {'thread_id': sessionId}}
        self.graph.invoke({'messages': [('user', query)]}, config)
        
        result = self.get_agent_response(config)
        result['processing_time'] = time.time() - start_time
        
        # Cache result
        if cache_manager and result.get('is_task_complete'):
            cache_manager.set(cache_key, result, ttl=settings.cache_ttl)
        
        return result
    
    async def stream(self, query: str, sessionId: str) -> AsyncIterable[Dict[str, Any]]:
        """Streaming with progress indicators"""
        start_time = time.time()
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': sessionId}}
        
        # Check cache first
        if cache_manager:
            cache_key = f"query:{hash(query)}:{sessionId}"
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                cached_result['from_cache'] = True
                cached_result['processing_time'] = time.time() - start_time
                yield cached_result
                return
        
        tool_calls_made = []
        
        for item in self.graph.stream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.get('name', 'unknown')
                    tool_calls_made.append(tool_name)
                    
                    if tool_name == 'rag_search':
                        yield {
                            'is_task_complete': False,
                            'require_user_input': False,
                            'content': '🔍 Searching product database with vector similarity...',
                            'tool_used': tool_name
                        }
                    elif tool_name == 'web_search':
                        yield {
                            'is_task_complete': False,
                            'require_user_input': False, 
                            'content': '🌐 Searching web for current information...',
                            'tool_used': tool_name
                        }
                    elif tool_name == 'shop_information_rag':
                        yield {
                            'is_task_complete': False,
                            'require_user_input': False,
                            'content': '🏪 Retrieving shop information...',
                            'tool_used': tool_name
                        }
        
        # Get final result
        final_result = self.get_agent_response(config)
        final_result['processing_time'] = time.time() - start_time
        final_result['tools_used'] = list(set(tool_calls_made))
        
        # If RAG was insufficient and web_search is enabled, proactively search the web
        try:
            if (
                final_result.get('is_task_complete') is False
                and final_result.get('require_user_input') is True
                and settings.enable_web_search
            ):
                # Perform a direct web search fallback
                tool_calls_made.append('web_search')
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': '🌐 No exact match in database. Searching the web for the latest information...',
                    'tool_used': 'web_search'
                }
                web_results = web_search.invoke({"query": query, "max_results": 3}) if hasattr(web_search, 'invoke') else web_search(query=query, max_results=3)
                if isinstance(web_results, list) and web_results:
                    # Synthesize a concise answer from web results
                    top = web_results[0]
                    snippet = top.get('snippet', '')
                    link = top.get('link', '')
                    synthesized = f"{snippet}\n\nNguồn: {link}" if link else snippet
                    final_result = {
                        'is_task_complete': True,
                        'require_user_input': False,
                        'content': synthesized or 'Không tìm thấy thông tin phù hợp trên web.',
                        'confidence': 0.6,
                        'sources': [r.get('link', '') for r in web_results if r.get('link')][:3],
                        'from_cache': False,
                    }
                else:
                    # Keep the original message but convert to completed with explanation
                    final_result = {
                        'is_task_complete': True,
                        'require_user_input': False,
                        'content': final_result.get('content') or 'Không thể truy cập công cụ tìm kiếm web lúc này.',
                        'confidence': 0.5,
                        'sources': [],
                        'from_cache': False,
                    }
                final_result['processing_time'] = time.time() - start_time
        except Exception as e:
            logger.warning(f"Web search fallback failed: {e}")

        # Cache result
        if cache_manager and final_result.get('is_task_complete'):
            cache_manager.set(cache_key, final_result, ttl=settings.cache_ttl)
        
        yield final_result
    
    def get_agent_response(self, config) -> Dict[str, Any]:
        """Response processing with metadata"""
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')

        logger.info(f"Current State: {current_state}")  # Add this logging
        logger.info(f"Structured Response: {structured_response}") # Add this logging

        base_response = {
            'is_task_complete': True,
            'require_user_input': False,
            'content': 'Response processed successfully.',
            'confidence': 1.0,
            'sources': [],
            'from_cache': False
        }

        if structured_response and isinstance(structured_response, ResponseFormat):
            if structured_response.status == 'input_required':
                base_response.update({
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                    'confidence': structured_response.confidence,
                    'sources': structured_response.sources
                })
            elif structured_response.status == 'error':
                base_response.update({
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                    'confidence': 0.0,
                    'sources': []
                })
            elif structured_response.status == 'completed':
                base_response.update({
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                    'confidence': structured_response.confidence,
                    'sources': structured_response.sources
                })
            else:
                logger.warning(f"Unexpected structured_response.status: {structured_response.status}") # Add this logging
                base_response['content'] = "An unexpected error occurred." # Add a default message
        else:
            logger.warning("structured_response is None or not a ResponseFormat object.") # Add this logging
            base_response['content'] = "The agent failed to generate a valid response." # Add a default message

        return base_response
    
    # Maintain compatibility with original interface
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
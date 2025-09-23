import streamlit as st
import asyncio
import httpx
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any, Optional
from uuid import uuid4
import base64
import os
import warnings

warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

# A2A imports
try:
    from a2a.client import A2ACardResolver, ClientFactory, A2AClient
    from a2a.types import (
        Part,
        TextPart,
        FilePart,
        FileWithBytes,
        Task,
        TaskState,
        Message,
        MessageSendConfiguration,
        SendMessageRequest,
        MessageSendParams,
    )
except ImportError as e:
    st.error(f"❌ A2A SDK import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Enhanced A2A Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stAlert {
        border-radius: 10px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #ddd;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
    }
    .agent-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 2rem;
    }
    .metrics-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .feature-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .error-message {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .success-message {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class RealA2AStreamlitClient:
    """Real A2A client integration for Streamlit"""
    
    def __init__(self, agent_url: str):
        self.agent_url = agent_url.rstrip('/')
        self.httpx_client = None
        self.a2a_client = None
        self.agent_card = None
        self.is_connected = False
        
    async def initialize(self):
        """Initialize the A2A client"""
        try:
            if not self.httpx_client:
                self.httpx_client = httpx.AsyncClient(timeout=60.0)
            
            # If already connected, skip re-initialization
            if self.is_connected and self.a2a_client and self.agent_card:
                return True, None

            # Get agent card
            card_resolver = A2ACardResolver(self.httpx_client, self.agent_url)
            self.agent_card = await card_resolver.get_agent_card()
            
            # Initialize A2A client using ClientFactory if available, otherwise fallback
            try:
                if 'ClientFactory' in globals() and hasattr(ClientFactory, 'create_client'):
                    self.a2a_client = await ClientFactory.create_client(
                        transport_url=self.agent_url,
                        agent_card=self.agent_card
                    )
                else:
                    # Fallback for older SDKs
                    self.a2a_client = A2AClient(self.httpx_client, agent_card=self.agent_card)
            except AttributeError:
                # Fallback when create_client is missing
                self.a2a_client = A2AClient(self.httpx_client, agent_card=self.agent_card)
            
            self.is_connected = True
            return True, None
            
        except Exception as e:
            self.is_connected = False
            return False, str(e)
    
    async def send_message(
        self, 
        message_text: str, 
        session_id: str = "streamlit",
        context_id: str = None,
        task_id: str = None,
        attached_file_path: str = None
    ) -> Dict[str, Any]:
        """Send message to A2A agent"""
        # Ensure client is initialized
        if not self.is_connected or not self.a2a_client:
            await self.initialize()
        if not self.is_connected:
            return {"error": "Failed to connect to A2A agent"}
        
        try:
            start_time = time.time()
            
            # Create message parts
            parts = [TextPart(text=message_text)]
            
            # Add file attachment if provided
            if attached_file_path and os.path.exists(attached_file_path):
                try:
                    with open(attached_file_path, 'rb') as f:
                        file_content = base64.b64encode(f.read()).decode('utf-8')
                        file_name = os.path.basename(attached_file_path)
                    
                    parts.append(
                        Part(
                            root=FilePart(
                                file=FileWithBytes(
                                    name=file_name,
                                    bytes=file_content
                                )
                            )
                        )
                    )
                except Exception as e:
                    pass  # Skip attachment silently
            
            # Create message
            message = Message(
                role='user',
                parts=parts,
                messageId=str(uuid4()),
                # Always start a new task; maintain conversation via contextId only
                taskId=None,
                contextId=context_id,
            )
            
            # Create payload
            payload = MessageSendParams(
                id=str(uuid4()),
                message=message,
                configuration=MessageSendConfiguration(
                    acceptedOutputModes=['text'],
                ),
            )
            
            # Send message and rely on HTTP client's timeout
            try:
                event = await self.a2a_client.send_message(
                    SendMessageRequest(
                        id=str(uuid4()),
                        params=payload,
                    )
                )
            except httpx.TimeoutException as e:
                self.is_connected = False
                return {"error": "HTTP timeout when contacting agent", "details": str(e), "success": False}
            except httpx.ConnectError as e:
                self.is_connected = False
                return {"error": "Failed to connect to agent (connection error)", "details": str(e), "success": False}
            except Exception as e:
                self.is_connected = False
                return {"error": "Unexpected error while contacting agent", "details": str(e), "success": False}
            
            processing_time = time.time() - start_time
            
            # Process response: unwrap event robustly
            if hasattr(event, 'root'):
                try:
                    root_obj = getattr(event, 'root')
                    result_obj = getattr(root_obj, 'result', None)
                    event = result_obj if result_obj is not None else root_obj
                except Exception:
                    pass

            # Extract response content and metadata robustly (duck-typed)
            def _get_attr(obj, *names):
                for n in names:
                    if hasattr(obj, n):
                        return getattr(obj, n)
                return None

            def _extract_text_from_parts(parts):
                texts = []
                for p in parts or []:
                    root = getattr(p, 'root', None) or p
                    text = getattr(root, 'text', None)
                    if text:
                        texts.append(text)
                return "\n".join([t for t in texts if t])

            response_content = ""
            metadata = {
                "processing_time": processing_time,
                "tools_used": [],
                "confidence": 0.0,
                "sources": [],
                "context_id": context_id,
                "task_id": None
            }

            # 1) Prefer structured_response.message (or camelCase) when present
            sr = _get_attr(event, 'structured_response', 'structuredResponse')
            if sr is not None:
                msg = getattr(sr, 'message', None)
                if not msg and isinstance(sr, dict):
                    msg = sr.get('message')
                if msg:
                    response_content = msg
                sr_status = getattr(sr, 'status', None) if not isinstance(sr, dict) else sr.get('status')
                if isinstance(sr_status, str) and sr_status.lower() == 'input_required':
                    metadata["needs_input"] = True

            # 2) If empty, try parts (Message-like)
            if not response_content:
                parts = _get_attr(event, 'parts')
                if parts is None and isinstance(event, dict):
                    parts = event.get('parts')
                if parts:
                    response_content = _extract_text_from_parts(parts)

            # 3) If empty, try artifacts (Task-like)
            if not response_content:
                artifacts = _get_attr(event, 'artifacts')
                if artifacts is None and isinstance(event, dict):
                    artifacts = event.get('artifacts')
                if artifacts:
                    all_text = []
                    for artifact in artifacts or []:
                        all_text.append(_extract_text_from_parts(getattr(artifact, 'parts', None) if not isinstance(artifact, dict) else artifact.get('parts')))
                    response_content = "\n".join([t for t in all_text if t])

            # 4) If empty, try messages list
            if not response_content:
                messages_list = _get_attr(event, 'messages')
                if messages_list is None and isinstance(event, dict):
                    messages_list = event.get('messages')
                if messages_list:
                    for m in messages_list or []:
                        role = getattr(m, 'role', None) if not isinstance(m, dict) else m.get('role')
                        if role in ('assistant', 'tool'):
                            parts = getattr(m, 'parts', None) if not isinstance(m, dict) else m.get('parts')
                            text = _extract_text_from_parts(parts)
                            if text:
                                response_content = text
                                break

            # 5) Deep-inspect as dict to find structured_response.message or any 'message'
            if not response_content:
                def _to_dict(obj):
                    try:
                        if isinstance(obj, dict):
                            return obj
                        if hasattr(obj, 'dict'):
                            return obj.dict()
                        if hasattr(obj, 'model_dump'):
                            return obj.model_dump()
                        if hasattr(obj, '__dict__'):
                            return dict(obj.__dict__)
                    except Exception:
                        return None
                    return None

                def _deep_find_message(d):
                    try:
                        if not isinstance(d, dict):
                            return None
                        # Prefer structured_response.message
                        sr = d.get('structured_response') or d.get('structuredResponse')
                        if isinstance(sr, dict):
                            if isinstance(sr.get('message'), str) and sr.get('message'):
                                return sr.get('message')
                        # General message key
                        if isinstance(d.get('message'), str) and d.get('message'):
                            return d.get('message')
                        # Recurse
                        for v in d.values():
                            sub = _to_dict(v)
                            if isinstance(sub, dict):
                                found = _deep_find_message(sub)
                                if found:
                                    return found
                            elif isinstance(v, list):
                                for item in v:
                                    sub2 = _to_dict(item)
                                    if isinstance(sub2, dict):
                                        found = _deep_find_message(sub2)
                                        if found:
                                            return found
                        return None
                    except Exception:
                        return None

                event_dict = _to_dict(event)
                msg = _deep_find_message(event_dict) if event_dict else None
                if msg:
                    response_content = msg

            # Context propagation
            ctx = _get_attr(event, 'contextId', 'context_id')
            if not ctx and isinstance(event, dict):
                ctx = event.get('contextId') or event.get('context_id')
            if ctx:
                metadata["context_id"] = ctx

            # Status detection for input-required
            if 'needs_input' not in metadata:
                status_obj = _get_attr(event, 'status')
                state_val = getattr(status_obj, 'state', None)
                if isinstance(event, dict):
                    state_val = state_val or ((event.get('status') or {}).get('state') if isinstance(event.get('status'), dict) else None)
                if (isinstance(state_val, str) and state_val == TaskState.input_required.name):
                    metadata["needs_input"] = True

            if not response_content and metadata.get('needs_input'):
                response_content = "The agent needs more input to continue. Please provide additional details."
            
            if not response_content:
                response_content = "Agent processed your request but returned no content."
            
            return {
                "content": response_content,
                "metadata": metadata,
                "success": True
            }
            
        except Exception as e:
            return {
                "error": f"Failed to send message: {str(e)}",
                "success": False
            }
    
    async def get_agent_card_info(self) -> Dict[str, Any]:
        """Get agent card information"""
        if not self.agent_card:
            await self.initialize()
        
        if self.agent_card:
            return {
                "name": self.agent_card.name,
                "description": self.agent_card.description,
                "version": self.agent_card.version,
                "skills": [{"name": skill.name, "description": skill.description} 
                          for skill in self.agent_card.skills]
            }
        return {}
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        if not self.httpx_client:
            return {"error": "Client not initialized"}
        
        try:
            response = await self.httpx_client.get(f"{self.agent_url}/metrics")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def get_health(self) -> Dict[str, Any]:
        """Get health status"""
        if not self.httpx_client:
            return {"error": "Client not initialized"}
        
        try:
            response = await self.httpx_client.get(f"{self.agent_url}/health/detailed")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def close(self):
        """Close the client connection"""
        if self.a2a_client and hasattr(self.a2a_client, 'close'):
            try:
                await self.a2a_client.close()
            except:
                pass
        if self.httpx_client:
            await self.httpx_client.aclose()

# Async utilities for Streamlit
import threading
from concurrent.futures import ThreadPoolExecutor

# Maintain a single background event loop thread for all async operations
_loop_lock = threading.Lock()

def _ensure_background_loop():
    """Ensure a single persistent event loop across Streamlit reruns using session_state."""
    with _loop_lock:
        loop = getattr(st.session_state, 'async_loop', None)
        loop_thread = getattr(st.session_state, 'async_loop_thread', None)
        if loop is not None and loop_thread is not None and loop_thread.is_alive():
            return loop

        loop = asyncio.new_event_loop()

        def _run_loop(l: asyncio.AbstractEventLoop):
            asyncio.set_event_loop(l)
            l.run_forever()

        loop_thread = threading.Thread(target=_run_loop, args=(loop,), name="streamlit_async_loop", daemon=True)
        loop_thread.start()

        st.session_state.async_loop = loop
        st.session_state.async_loop_thread = loop_thread
        return loop

def run_async_in_thread(coro, timeout: float = None):
    """Run coroutine on a persistent background event loop and wait for result with optional timeout"""
    loop = _ensure_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=timeout)
    except Exception as e:
        try:
            future.cancel()
        except Exception:
            pass
        raise e

def main():
    """Main Streamlit application"""
    
    # Sidebar configuration
    st.sidebar.image("https://via.placeholder.com/200x100/4facfe/white?text=A2A+Agent", width=200)
    st.sidebar.title("🤖 Enhanced A2A Agent")
    st.sidebar.markdown("---")
    
    # Connection settings
    st.sidebar.subheader("🔌 Connection")
    agent_url = st.sidebar.text_input(
        "Agent URL", 
        value="http://localhost:10000",
        help="URL of the A2A agent server"
    )
    
    session_id = st.sidebar.text_input(
        "Session ID",
        value="streamlit_user",
        help="Unique session identifier"
    )
    
    # Features toggles
    st.sidebar.subheader("✨ Features")
    show_metrics = st.sidebar.checkbox("📊 Show Metrics", value=True)
    show_debug = st.sidebar.checkbox("🐛 Debug Mode", value=False)
    auto_scroll = st.sidebar.checkbox("📜 Auto Scroll", value=True)
    enable_file_upload = st.sidebar.checkbox("📎 File Upload", value=False)
    
    # Initialize client
    if 'client' not in st.session_state or st.session_state.get('agent_url') != agent_url:
        st.session_state.client = RealA2AStreamlitClient(agent_url)
        st.session_state.agent_url = agent_url
        st.session_state.client_initialized = False
    
    # Initialize chat history and conversation state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.context_id = None
        st.session_state.task_id = None
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "🎉 Chào mừng đến với Enhanced A2A Agent! Tôi sẽ kết nối đến agent thực để trả lời câu hỏi của bạn.\n\nHãy hỏi tôi bất cứ điều gì!",
            "timestamp": datetime.now(),
            "metadata": {}
        })
    
    # Main content
    st.title("🤖 Enhanced A2A Agent Interface")
    st.markdown("**Powered by Real A2A Protocol Integration**")
    
    # Connection status
    connection_status = st.empty()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "⚙️ System"])
    
    with tab1:
        render_chat_interface(
            show_debug, auto_scroll, session_id, 
            enable_file_upload, connection_status
        )
    
    with tab2:
        if show_metrics:
            render_analytics_dashboard()
    
    with tab3:
        render_system_info()

def render_chat_interface(
    show_debug: bool, 
    auto_scroll: bool, 
    session_id: str,
    enable_file_upload: bool,
    connection_status
):
    """Render the main chat interface"""
    
    # Check connection status
    if not st.session_state.get('client_initialized', False):
        with st.spinner("🔄 Initializing A2A connection..."):            
            try:
                init_result = run_async_in_thread(st.session_state.client.initialize())
                success, err = init_result if isinstance(init_result, tuple) else (init_result, None)
                if success:
                    st.session_state.client_initialized = True
                    # Get agent card info
                    card_info = run_async_in_thread(st.session_state.client.get_agent_card_info())
                    if card_info:
                        connection_status.success(
                            f"✅ Connected to {card_info.get('name', 'A2A Agent')} "
                            f"v{card_info.get('version', 'Unknown')}"
                        )
                        
                        # Update welcome message with agent info
                        if st.session_state.messages and len(st.session_state.messages) == 1:
                            welcome_content = f"""🎉 Kết nối thành công đến **{card_info.get('name', 'A2A Agent')}**!

📝 **Mô tả**: {card_info.get('description', 'No description available')}
🔧 **Phiên bản**: {card_info.get('version', 'Unknown')}

🌟 **Kỹ năng có sẵn**:"""
                            
                            for skill in card_info.get('skills', []):
                                welcome_content += f"\n• **{skill['name']}**: {skill['description']}"
                            
                            welcome_content += "\n\nHãy hỏi tôi bất cứ điều gì!"
                            
                            st.session_state.messages[0]['content'] = welcome_content
                else:
                    if err:
                        connection_status.error(f"❌ Failed to connect to A2A agent: {err}")
                    else:
                        connection_status.error("❌ Failed to connect to A2A agent")
            except Exception as e:
                connection_status.error(f"❌ Connection error: {e}")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            with st.container():
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You</strong> <small>({message['timestamp'].strftime('%H:%M:%S')})</small><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    metadata = message.get('metadata', {})
                    confidence = metadata.get('confidence', '')
                    sources = metadata.get('sources', [])
                    processing_time = metadata.get('processing_time', '')
                    tools_used = metadata.get('tools_used', [])
                    
                    # Build metadata string
                    meta_parts = []
                    if confidence and confidence > 0:
                        meta_parts.append(f"🎯 {confidence:.2f}")
                    if processing_time:
                        meta_parts.append(f"⏱️ {processing_time:.2f}s")
                    if tools_used:
                        meta_parts.append(f"🔧 {', '.join(tools_used)}")
                    if metadata.get('from_cache'):
                        meta_parts.append("💾 Cached")
                    
                    meta_str = " | ".join(meta_parts)
                    
                    # Show error or success styling
                    message_class = "agent-message"
                    if message.get('error'):
                        message_class = "error-message"
                    
                    st.markdown(f"""
                    <div class="chat-message {message_class}">
                        <strong>🤖 Agent</strong> <small>({message['timestamp'].strftime('%H:%M:%S')})</small>
                        {f"<br><small>{meta_str}</small>" if meta_str else ""}<br><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show sources
                    if sources and show_debug:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(sources, 1):
                                st.write(f"{i}. {source}")
                    
                    # Show debug info
                    if show_debug and metadata:
                        with st.expander("🔧 Debug Info"):
                            st.json(metadata)
    
    # Chat input
    st.markdown("---")
    
    # File upload section
    uploaded_file = None
    if enable_file_upload:
        uploaded_file = st.file_uploader(
            "📎 Attach a file (optional)",
            type=['txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg'],
            key="file_upload"
        )

    # Unified input: single-line input with Enter-to-send and a Send button
    # Initialize session state keys
    if 'chat_input_text' not in st.session_state:
        st.session_state['chat_input_text'] = ""
    if 'pending_send' not in st.session_state:
        st.session_state['pending_send'] = False
    if 'pending_text' not in st.session_state:
        st.session_state['pending_text'] = ""
    
    # Clear input in a safe way on next rerun
    if st.session_state.get('clear_input_once'):
        st.session_state['chat_input_text'] = ""
        st.session_state['clear_input_once'] = False

    def _on_text_submit():
        text_value = st.session_state.get('chat_input_text', '')
        st.session_state['pending_text'] = text_value.strip()
        st.session_state['pending_send'] = True

    col1, col2 = st.columns([4, 1])
    with col1:
        st.text_input(
            "💭 Your message:",
            placeholder="Hỏi về sản phẩm, cửa hàng, hoặc tin tức công nghệ...",
            key="chat_input_text",
            on_change=_on_text_submit
        )
    with col2:
        st.write("")
        send_button = st.button("📤 Send", use_container_width=True, type="primary")
        clear_button = st.button("🗑️ Clear", use_container_width=True)

    user_input = None
    trigger_send = False
    if send_button:
        user_input = st.session_state.get('chat_input_text', '').strip()
        trigger_send = True
    elif st.session_state.get('pending_send'):
        user_input = st.session_state.get('pending_text', '').strip()
        trigger_send = True
        # Reset pending flags if empty to avoid loops
        if not user_input:
            st.session_state['pending_send'] = False
            st.session_state['pending_text'] = ""
    
    # Handle clear button
    if clear_button:
        st.session_state.messages = []
        st.session_state.context_id = None
        st.session_state.task_id = None
        st.rerun()
    
    # Handle send button
    if trigger_send and user_input:
        if not st.session_state.get('client_initialized', False):
            st.error("❌ Please wait for the A2A connection to initialize")
            return
        
        # Add user message
        user_message = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now(),
            "metadata": {}
        }
        st.session_state.messages.append(user_message)
        
        # Handle file upload
        file_path = None
        if uploaded_file:
            # Save uploaded file temporarily
            temp_dir = "/tmp" if os.path.exists("/tmp") else "."
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Show processing indicator
        with st.spinner("🤔 Agent is processing your request..."):
            try:
                # Send message to real A2A agent
                response = run_async_in_thread(
                    st.session_state.client.send_message(
                        message_text=user_input,
                        session_id=session_id,
                        context_id=st.session_state.context_id,
                        task_id=None,
                        attached_file_path=file_path
                    )
                )
                
                # Clean up temporary file
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
                
                if response.get("success"):
                    # Update conversation state
                    metadata = response.get("metadata", {})
                    if metadata.get("context_id"):
                        st.session_state.context_id = metadata["context_id"]
                    if metadata.get("task_id"):
                        st.session_state.task_id = metadata["task_id"]
                    
                    # If content is empty and processing time is suspiciously fast, retry once
                    content_text = response.get("content") or ""
                    if (not content_text.strip()) and float(metadata.get("processing_time") or 0) < 0.1:
                        try:
                            # Reinitialize client and retry send once
                            run_async_in_thread(st.session_state.client.initialize())
                            retry_resp = run_async_in_thread(
                                st.session_state.client.send_message(
                                    message_text=user_input,
                                    session_id=session_id,
                                    context_id=st.session_state.context_id,
                                    task_id=None,
                                    attached_file_path=file_path
                                )
                            )
                            if retry_resp.get("success") and retry_resp.get("content"):
                                response = retry_resp
                                metadata = response.get("metadata", {})
                                if metadata.get("context_id"):
                                    st.session_state.context_id = metadata["context_id"]
                                if metadata.get("task_id"):
                                    st.session_state.task_id = metadata["task_id"]
                                content_text = response.get("content", "")
                        except Exception:
                            pass

                    # Add agent response
                    agent_message = {
                        "role": "assistant",
                        "content": content_text if content_text.strip() else ("Agent needs more input." if metadata.get("needs_input") else "Agent processed your request but returned no content."),
                        "timestamp": datetime.now(),
                        "metadata": metadata
                    }
                    st.session_state.messages.append(agent_message)
                else:
                    # Add error message with details if present
                    details = response.get('details')
                    content_err = f"❌ Error: {response.get('error', 'Unknown error occurred')}" + (f"\nDetails: {details}" if details else "")
                    error_message = {
                        "role": "assistant",
                        "content": content_err,
                        "timestamp": datetime.now(),
                        "metadata": {},
                        "error": True
                    }
                    st.session_state.messages.append(error_message)
                    # Only force re-init on connection/timeout type errors
                    err_text = (response.get('error') or '') + ' ' + (response.get('details') or '')
                    if any(k in err_text.lower() for k in ['timeout', 'connect', 'connection']):
                        st.session_state.client_initialized = False
                
                # Clear input state and rerun (defer clearing input value to next run)
                st.session_state['pending_send'] = False
                st.session_state['pending_text'] = ""
                st.session_state['clear_input_once'] = True
                st.rerun()
                
            except Exception as e:
                # Show detailed error; re-init only for network/timeout errors
                err_name = type(e).__name__
                err_msg = str(e)
                st.error(f"❌ Error communicating with agent: {err_name}: {err_msg}")
                error_message = {
                    "role": "assistant",
                    "content": f"❌ Connection Error: {err_name}: {err_msg}\n\nPlease check if the A2A agent is running at {st.session_state.client.agent_url}",
                    "timestamp": datetime.now(),
                    "metadata": {
                        "agent_url": st.session_state.client.agent_url,
                        "context_id": st.session_state.get('context_id'),
                    },
                    "error": True
                }
                st.session_state.messages.append(error_message)
                if any(k in (err_name + ' ' + err_msg).lower() for k in ['timeout', 'connect', 'connection']):
                    st.session_state.client_initialized = False
                st.rerun()

def render_analytics_dashboard():
    """Render analytics and metrics dashboard"""
    
    st.subheader("📊 Real-time Analytics")
    
    # Get real metrics from agent
    if st.session_state.get('client_initialized', False):
        with st.spinner("📊 Loading metrics..."):
            try:
                metrics_data = run_async_in_thread(st.session_state.client.get_metrics())
                
                if "error" not in metrics_data:
                    display_real_metrics(metrics_data)
                else:
                    st.error(f"❌ Failed to load metrics: {metrics_data['error']}")
                    display_fallback_metrics()
            except Exception as e:
                st.error(f"❌ Metrics error: {e}")
                display_fallback_metrics()
    else:
        st.warning("⚠️ Connect to agent first to view real metrics")
        display_fallback_metrics()

def display_real_metrics(metrics_data: Dict[str, Any]):
    """Display real metrics from agent"""
    
    agent_metrics = metrics_data.get('agent_metrics', {})
    config = metrics_data.get('config', {})
    
    # Main metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_requests = agent_metrics.get('total_requests', 0)
        st.markdown(f"""
        <div class="metrics-card">
            <h3>🔢 Total Requests</h3>
            <h2>{total_requests:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        success_rate = agent_metrics.get('success_rate', 0)
        st.markdown(f"""
        <div class="metrics-card">
            <h3>✅ Success Rate</h3>
            <h2>{success_rate:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_time = agent_metrics.get('average_response_time', 0)
        st.markdown(f"""
        <div class="metrics-card">
            <h3>⏱️ Avg Response</h3>
            <h2>{avg_time:.2f}s</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        cache_hit = agent_metrics.get('cache_hit_rate', 0)
        st.markdown(f"""
        <div class="metrics-card">
            <h3>💾 Cache Hit</h3>
            <h2>{cache_hit:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Tool usage and performance breakdown
    tool_metrics = metrics_data.get('tool_metrics', {})
    if tool_metrics:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 Tool Usage")
            
            tools = list(tool_metrics.keys())
            usage = [tool_metrics[tool].get('usage_count', 0) for tool in tools]
            
            if tools and any(usage):
                fig = px.pie(values=usage, names=tools, title="Tool Usage Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⚡ Tool Performance")
            
            performance_data = []
            for tool, data in tool_metrics.items():
                performance_data.append({
                    'Tool': tool,
                    'Requests': data.get('usage_count', 0),
                    'Avg Time (s)': data.get('average_time', 0),
                    'Success Rate': f"{data.get('success_rate', 0):.1%}"
                })
            
            if performance_data:
                df = pd.DataFrame(performance_data)
                st.dataframe(df, use_container_width=True)

def display_fallback_metrics():
    """Display fallback/demo metrics when real ones aren't available"""
    
    import random
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metrics-card">
            <h3>🔢 Total Requests</h3>
            <h2>--</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metrics-card">
            <h3>✅ Success Rate</h3>
            <h2>--</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metrics-card">
            <h3>⏱️ Avg Response</h3>
            <h2>--</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metrics-card">
            <h3>💾 Cache Hit</h3>
            <h2>--</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.info("📊 Real metrics will appear once connected to the A2A agent")

def render_system_info():
    """Render system information and controls"""
    
    st.subheader("⚙️ System Information")
    
    # Agent connection status
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.get('client_initialized', False):
            st.markdown("""
            <div class="success-message">
                <h4>🔗 A2A Connection</h4>
                <p>✅ Connected and Ready</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="error-message">
                <h4>🔗 A2A Connection</h4>
                <p>❌ Not Connected</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Agent card information
        if st.session_state.get('client_initialized', False):
            with st.expander("📋 Agent Information"):
                try:
                    card_info = run_async_in_thread(
                        st.session_state.client.get_agent_card_info()
                    )
                    
                    if card_info:
                        st.write(f"**Name**: {card_info.get('name', 'Unknown')}")
                        st.write(f"**Version**: {card_info.get('version', 'Unknown')}")
                        st.write(f"**Description**: {card_info.get('description', 'No description')}")
                        
                        skills = card_info.get('skills', [])
                        if skills:
                            st.write("**Available Skills**:")
                            for skill in skills:
                                st.write(f"• **{skill['name']}**: {skill['description']}")
                except Exception as e:
                    st.error(f"Failed to get agent info: {e}")
    
    with col2:
        # System health
        if st.button("🔄 Check Health", use_container_width=True):
            if st.session_state.get('client_initialized', False):
                with st.spinner("Checking agent health..."):
                    try:
                        health_data = run_async_in_thread(
                            st.session_state.client.get_health()
                        )
                        
                        if "error" not in health_data:
                            st.success("✅ Agent is healthy!")
                            with st.expander("Health Details"):
                                st.json(health_data)
                        else:
                            st.error(f"❌ Health check failed: {health_data['error']}")
                    except Exception as e:
                        st.error(f"❌ Health check error: {e}")
            else:
                st.error("❌ Not connected to agent")
        
        if st.button("🔄 Reconnect", use_container_width=True):
            st.session_state.client_initialized = False
            st.session_state.context_id = None
            st.session_state.task_id = None
            st.rerun()
    
    # Conversation state
    st.subheader("💬 Conversation State")
    
    col1, col2 = st.columns(2)
    
    with col1:
        context_id = st.session_state.get('context_id', 'None')
        st.info(f"**Context ID**: {context_id}")
    
    with col2:
        task_id = st.session_state.get('task_id', 'None')
        st.info(f"**Task ID**: {task_id}")
    
    if st.button("🗑️ Reset Conversation State"):
        st.session_state.context_id = None
        st.session_state.task_id = None
        st.success("✅ Conversation state reset!")
    
    # Configuration and controls
    st.subheader("🎛️ System Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Refresh Metrics", use_container_width=True):
            st.success("✅ Metrics will refresh on next view!")
    
    with col2:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.success("✅ Chat history cleared!")
    
    with col3:
        if st.button("💾 Export Chat", use_container_width=True):
            if st.session_state.messages:
                chat_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "session_id": st.session_state.get('session_id', 'unknown'),
                    "context_id": st.session_state.get('context_id'),
                    "task_id": st.session_state.get('task_id'),
                    "messages": []
                }
                
                for msg in st.session_state.messages:
                    chat_data["messages"].append({
                        "role": msg["role"],
                        "content": msg["content"],
                        "timestamp": msg["timestamp"].isoformat(),
                        "metadata": msg.get("metadata", {})
                    })
                
                st.download_button(
                    "📥 Download Chat Export",
                    data=json.dumps(chat_data, indent=2, ensure_ascii=False),
                    file_name=f"a2a_chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.info("No chat history to export")
    
    # Advanced configuration
    with st.expander("🔧 Advanced Settings"):
        st.markdown("**Connection Settings**")
        
        col1, col2 = st.columns(2)
        with col1:
            timeout_value = st.number_input(
                "Request Timeout (seconds)", 
                min_value=10, 
                max_value=300, 
                value=60,
                help="Timeout for A2A requests"
            )
        
        with col2:
            retry_count = st.number_input(
                "Max Retries", 
                min_value=0, 
                max_value=5, 
                value=1,
                help="Number of retry attempts for failed requests"
            )
        
        st.markdown("**Display Settings**")
        show_timestamps = st.checkbox("Show Message Timestamps", value=True)
        show_metadata = st.checkbox("Show Response Metadata", value=False)
        compact_mode = st.checkbox("Compact Message Display", value=False)
        
        if st.button("💾 Save Advanced Settings"):
            # In a real implementation, you'd save these settings
            st.success("✅ Advanced settings saved!")
    
    # About and documentation
    st.subheader("ℹ️ About")
    st.markdown("""
    **Enhanced A2A Agent Interface v2.0**
    
    🔧 **Real A2A Integration Features:**
    - ✅ Authentic A2A protocol communication
    - 🔄 Real-time message streaming
    - 📎 File attachment support
    - 💾 Conversation state management
    - 📊 Live performance metrics
    - 🔍 Multi-tool orchestration (RAG, Web Search, etc.)
    
    🛠️ **Built with:**
    - A2A Protocol SDK
    - LangGraph Multi-Agent Framework
    - Streamlit Interactive Interface
    - Real-time Metrics Dashboard
    
    📚 **Usage Tips:**
    - Messages maintain conversation context automatically
    - Upload files for document analysis
    - Use debug mode to see tool execution details
    - Monitor performance through the Analytics tab
    
    🆘 **Troubleshooting:**
    - If connection fails, check the agent URL
    - Use "Reconnect" button to refresh connection
    - Clear conversation state to start fresh
    - Check agent health status in System tab
    """)
    
    # Debug information
    if st.checkbox("🐛 Show Debug Info"):
        st.subheader("🔍 Debug Information")
        
        debug_info = {
            "streamlit_session_state": {
                "client_initialized": st.session_state.get('client_initialized', False),
                "agent_url": st.session_state.get('agent_url'),
                "context_id": st.session_state.get('context_id'),
                "task_id": st.session_state.get('task_id'),
                "message_count": len(st.session_state.get('messages', []))
            },
            "client_info": {
                "agent_url": st.session_state.client.agent_url if hasattr(st.session_state, 'client') else None,
                "is_connected": getattr(st.session_state.client, 'is_connected', False) if hasattr(st.session_state, 'client') else False,
            }
        }
        
        st.json(debug_info)

# Utility functions for async operations in Streamlit
def run_async_safely(coro):
    """Safely run async coroutine in Streamlit - DEPRECATED"""
    return run_async_in_thread(coro)

# Error handling and cleanup
@st.cache_resource
def get_client_singleton(agent_url: str):
    """Get or create a singleton client instance"""
    return RealA2AStreamlitClient(agent_url)

# Session cleanup
def cleanup_session():
    """Clean up session resources"""
    if 'client' in st.session_state and hasattr(st.session_state.client, 'close'):
        try:
            run_async_in_thread(st.session_state.client.close())
        except:
            pass
    # Stop background loop
    try:
        loop = getattr(st.session_state, 'async_loop', None)
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
    except:
        pass

# Register cleanup on app exit
import atexit
atexit.register(cleanup_session)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        
        # Show error details in debug mode
        if st.checkbox("Show Error Details"):
            import traceback
            st.code(traceback.format_exc())
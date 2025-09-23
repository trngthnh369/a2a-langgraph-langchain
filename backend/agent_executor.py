import logging
import time
from typing import Dict, Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import Event, EventQueue
from a2a.server.tasks import TaskUpdater

from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    Task,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError
from backend.agents.agent import LangGraphAgent
from backend.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class A2AAgentExecutor(AgentExecutor):
    """ A2A Agent Executor with performance monitoring and error handling"""

    def __init__(self):
        self.agent = LangGraphAgent()
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_response_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        logger.info("🚀A2A Agent Executor initialized")

    def _validate_request(self, context: RequestContext) -> bool:
        """Request validation"""
        try:
            user_input = context.get_user_input()
            if not user_input or len(user_input.strip()) < 2:
                logger.warning("Invalid request: empty or too short input")
                return True  # Return True means there's an error
            
            # Additional validation
            if len(user_input) > settings.max_context_length:
                logger.warning(f"Request too long: {len(user_input)} chars")
                return True
                
            return False  # No error
            
        except Exception as e:
            logger.error(f"Request validation error: {e}")
            return True
    
    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        """Handle task cancellation"""
        logger.info("Task cancellation requested")
        raise ServerError(error=UnsupportedOperationError())
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute method with monitoring and error handling"""
        start_time = time.time()
        self.performance_metrics['total_requests'] += 1
        
        error = self._validate_request(context)
        if error:
            self.performance_metrics['failed_requests'] += 1
            raise ServerError(error=InvalidParamsError())
        
        query = context.get_user_input()
        task = context.current_task
        
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        try:
            logger.info(f"🎯 Processing query: {query[:100]}...")
            
            # Always use stream, even if RAG is disabled
            async for item in self.agent.stream(query, task.context_id):
                await self._process_stream_item(item, updater, task)
            
            # Update performance metrics
            response_time = time.time() - start_time
            self.performance_metrics['total_response_time'] += response_time
            self.performance_metrics['successful_requests'] += 1
                
            if 'from_cache' in locals() and locals().get('from_cache'):
                self.performance_metrics['cache_hits'] += 1
            else:
                self.performance_metrics['cache_misses'] += 1
            
            logger.info(f"✅ Query processed successfully in {response_time:.2f}s")

        except Exception as e:
            response_time = time.time() - start_time
            self.performance_metrics['failed_requests'] += 1
            logger.error(f'❌ Error processing query: {e}', exc_info=True)
            
            # Send error response
            await updater.update_status(
                TaskState.input_required,
                new_agent_text_message(
                    f"Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại sau. (Error: {str(e)[:100]})",
                    task.context_id,
                    task.id,
                ),
                final=True,
            )
            raise ServerError(error=InternalError()) from e
    
    async def _process_stream_item(self, item: Dict[str, Any], updater: TaskUpdater, task: Task):
        """Process individual stream items"""
        is_task_complete = item.get('is_task_complete', False)
        require_user_input = item.get('require_user_input', False)
        content = item.get('content', '')
        
        if not is_task_complete and not require_user_input:
            # Intermediate processing message
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    content,
                    task.context_id,
                    task.id,
                ),
            )
        elif require_user_input:
            # Need more input
            await updater.update_status(
                TaskState.input_required,
                new_agent_text_message(
                    content,
                    task.context_id,
                    task.id,
                ),
                final=True,
            )
            # Also attach the same content as an artifact so clients that read artifacts can render it
            await updater.add_artifact(
                [Part(root=TextPart(text=content))],
                name='agent_response',
            )
            # Ensure the task is explicitly marked complete so clients resolve
            await updater.complete()
        else:
            # Task complete
            await self._process_final_result(item, updater, task)
    
    async def _process_final_result(self, result: Dict[str, Any], updater: TaskUpdater, task: Task):
        """Process final result"""
        content = result.get('content', 'Không có phản hồi')
        
        # Add metadata to response if available
        metadata_parts = []
        
        if result.get('confidence'):
            metadata_parts.append(f"🎯 Confidence: {result['confidence']:.2f}")
        
        if result.get('sources'):
            sources = result['sources'][:3]  # Limit sources
            if sources:
                metadata_parts.append(f"📚 Sources: {', '.join(sources[:2])}")
        
        if result.get('from_cache'):
            metadata_parts.append("💾 From cache")
        
        if result.get('processing_time'):
            metadata_parts.append(f"⏱️ {result['processing_time']:.2f}s")
        
        # Append metadata to content
        if metadata_parts:
            content += f"\n\n---\n{' | '.join(metadata_parts)}"
        
        # Create artifact with the response
        await updater.add_artifact(
            [Part(root=TextPart(text=content))],
            name='agent_response',
        )
        await updater.complete()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        total_requests = max(1, self.performance_metrics['total_requests'])  # Avoid division by zero
        
        return {
            'total_requests': self.performance_metrics['total_requests'],
            'successful_requests': self.performance_metrics['successful_requests'],
            'failed_requests': self.performance_metrics['failed_requests'],
            'success_rate': self.performance_metrics['successful_requests'] / total_requests,
            'average_response_time': self.performance_metrics['total_response_time'] / total_requests,
            'cache_hit_rate': self.performance_metrics['cache_hits'] / max(1, self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'])
        }
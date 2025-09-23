import logging
import asyncio

import os
import sys

import click
import httpx

from starlette.routing import Route
from starlette.responses import JSONResponse

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks.base_push_notification_sender import BasePushNotificationSender
from a2a.server.tasks.inmemory_push_notification_config_store import InMemoryPushNotificationConfigStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from backend import A2AAgentExecutor
from backend import LangGraphAgent
from backend import settings
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MissingAPIKeyError(Exception):
    """Exception for missing API key."""
    pass

@click.command()
@click.option('--host', 'host', default=settings.host)
@click.option('--port', 'port', default=settings.port)
@click.option('--setup-data', 'setup_data', is_flag=True, help='Setup vector database on startup')
def main(host, port, setup_data):
    """A2A Agent Server with RAG capabilities"""
    try:
        # Validate required configuration
        if not settings.google_api_key:
            raise MissingAPIKeyError(
                'GOOGLE_API_KEY environment variable not set. Please set it in .env file.'
            )

        logger.info("🚀 Starting A2A Agent Server...")
        logger.info(f"📍 Server: http://{host}:{port}")
        logger.info(f"🔍 RAG Enabled: {settings.enable_rag}")
        logger.info(f"🌐 Web Search: {settings.enable_web_search}")
        logger.info(f"💾 Caching: {settings.enable_caching}")

        # Setup vector database if requested
        if setup_data:
            logger.info("🔧 Setting up vector database...")
            asyncio.run(setup_vector_database())

        # Define agent capabilities
        capabilities = AgentCapabilities(
            streaming=True, 
            pushNotifications=True
        )
        
        # Skills with RAG and web search
        skills = [
            AgentSkill(
                id='product_search',
                name='Product Search',
                description='Advanced product search with vector similarity and real-time data',
                tags=['RAG', 'vector search', 'product information', 'AI-powered'],
                examples=[
                    'Tìm thông tin chi tiết về iPhone 15 Pro Max',
                    'So sánh Samsung Galaxy S24 và iPhone 15',
                    'Điện thoại nào tốt nhất trong tầm giá 10 triệu?'
                ],
            ),
            AgentSkill(
                id='smart_shop_information',
                name='Smart Shop Information',
                description='Comprehensive shop information with location services',
                tags=['shop locations', 'store hours', 'services', 'contact info'],
                examples=[
                    'Cửa hàng gần nhà tôi ở Hoàng Mai',
                    'Thông tin liên hệ các cửa hàng',
                    'Giờ mở cửa và dịch vụ có sẵn'
                ],
            ),
            AgentSkill(
                id='web_search_integration',
                name='Web Search Integration',
                description='Real-time web search for current information and trends',
                tags=['web search', 'current events', 'latest news', 'trends'],
                examples=[
                    'Tin tức công nghệ mới nhất',
                    'Giá điện thoại trên thị trường hiện tại',
                    'Đánh giá sản phẩm mới ra mắt'
                ],
            )
        ]

        #  agent card
        agent_card = AgentCard(
            name='RAG Agent',
            description='Advanced AI agent with RAG capabilities, vector search, and web integration for comprehensive product and shop information',
            url=f'http://{host}:{port}/',
            version='2.0.0',
            defaultInputModes=LangGraphAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=LangGraphAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=skills,
        )

        # Initialize components
        httpx_client = httpx.AsyncClient()
        push_notifier = BasePushNotificationSender(
            httpx_client,
            config_store=InMemoryPushNotificationConfigStore()
        )
        
        # Executor
        executor = A2AAgentExecutor()
        
        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
            push_sender=push_notifier,
        )
        
        # A2A server application
        server = A2AStarletteApplication(
            agent_card=agent_card, 
            http_handler=request_handler
        )
        
        # Add custom metrics endpoint
        # app = server.build()
        
        # @app.get("/metrics")
        # async def get_metrics():
        #     """Get performance metrics"""
        #     return {
        #         "status": "ok",
        #         "agent_metrics": executor.get_performance_metrics(),
        #         "config": {
        #             "rag_enabled": settings.enable_rag,
        #             "web_search_enabled": settings.enable_web_search,
        #             "caching_enabled": settings.enable_caching
        #         }
        #     }
        
        # @app.get("/health/detailed")
        # async def detailed_health():
        #     """Detailed health check"""
        #     return {
        #         "status": "healthy",
        #         "version": "2.0.0",
        #         "features": {
        #             "rag": settings.enable_rag,
        #             "web_search": settings.enable_web_search,
        #             "caching": settings.enable_caching
        #         },
        #         "metrics": executor.get_performance_metrics()
        #     }
        async def get_metrics(request):  # Add request argument
            """Get performance metrics"""
            return JSONResponse({
                "status": "ok",
                "agent_metrics": executor.get_performance_metrics(),
                "config": {
                    "rag_enabled": settings.enable_rag,
                    "web_search_enabled": settings.enable_web_search,
                    "caching_enabled": settings.enable_caching
                }
            })
        
        async def detailed_health(request):  # Add request argument
            """Detailed health check"""
            return JSONResponse({
                "status": "healthy",
                "version": "2.0.0",
                "features": {
                    "rag": settings.enable_rag,
                    "web_search": settings.enable_web_search,
                    "caching": settings.enable_caching
                },
                "metrics": executor.get_performance_metrics()
            })
        
        # Define routes            
        routes = [
            Route("/metrics", endpoint=get_metrics),
            Route("/health/detailed", endpoint=detailed_health)
        ]
        
        # Mount routes to the Starlette app
        app = server.build(routes=routes)
        
        import uvicorn
        logger.info("✅ Server initialization complete!")
        uvicorn.run(app, host=host, port=port)
        
    except MissingAPIKeyError as e:
        logger.error(f'❌ Configuration Error: {e}')
        logger.error('💡 Please create a .env file with your Google API key')
        exit(1)
    except Exception as e:
        logger.error(f'❌ Server startup failed: {e}', exc_info=True)
        exit(1)

async def setup_vector_database():
    """Setup vector database with sample data"""
    try:
        from backend.data.vector_store import VectorStore
        import pandas as pd
        import os
        
        # Check if data file exists
        data_file = "../data/products.csv"
        if os.path.exists(data_file):
            logger.info(f"📊 Loading data from {data_file}")
            df = pd.read_csv(data_file)
            
            # Process data for vector store
            documents = []
            for _, row in df.head(100).iterrows():  # Limit for demo
                content_parts = []
                
                if 'title' in row and pd.notna(row['title']):
                    content_parts.append(str(row['title']))
                
                if 'product_specs' in row and pd.notna(row['product_specs']):
                    specs = str(row['product_specs']).replace('<br>', ' ')
                    content_parts.append(specs)
                
                if 'current_price' in row and pd.notna(row['current_price']):
                    content_parts.append(f"Giá: {row['current_price']}")
                
                content = " ".join(content_parts)
                
                if content.strip():
                    doc = {
                        "content": content,
                        "title": str(row.get('title', '')),
                        "current_price": str(row.get('current_price', '')),
                        "product_specs": str(row.get('product_specs', ''))[:500],
                    }
                    documents.append(doc)
            
            # Initialize vector store and add documents
            vector_store = VectorStore()
            await vector_store.add_documents(documents, batch_size=20)
            logger.info(f"✅ Vector database setup complete with {len(documents)} documents")
        else:
            logger.warning(f"⚠️ Data file not found: {data_file}")
            logger.info("💡 Place your products.csv file in the ./data/ directory")
            
    except Exception as e:
        logger.error(f"❌ Vector database setup failed: {e}")

if __name__ == '__main__':
    main()
import asyncio
import base64
import os
import httpx
import time
import json

from uuid import uuid4

import asyncclick as click

from a2a.client import A2AClient, A2ACardResolver
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

@click.command()
@click.option('--agent', default='http://localhost:10000')
@click.option('--session', default='session')
@click.option('--history', default=False)
@click.option('--use_push_notifications', default=False)
@click.option('--push_notification_receiver', default='http://localhost:5000')
@click.option('--show_metrics', default=False, help='Show performance metrics')
@click.option('--benchmark', default=False, help='Run benchmark test')
async def cli(
    agent,
    session,
    history,
    use_push_notifications: bool,
    push_notification_receiver: str,
    show_metrics: bool,
    benchmark: bool,
):
    """A2A Client with metrics and benchmarking"""
    async with httpx.AsyncClient(timeout=60) as httpx_client:
        card_resolver = A2ACardResolver(httpx_client, agent)
        
        try:
            card = await card_resolver.get_agent_card()
            
            print('🤖 ======= A2A Agent Card ========')
            print(f"📝 Name: {card.name}")
            print(f"📄 Description: {card.description}")
            print(f"🔧 Version: {card.version}")
            print(f"🌟 Skills: {len(card.skills)} available")
            
            for skill in card.skills:
                print(f"  • {skill.name}: {skill.description}")
            
            print('=' * 45)
            
        except Exception as e:
            print(f"❌ Failed to get agent card: {e}")
            return

        client = A2AClient(httpx_client, agent_card=card)

        # Show metrics if requested
        if show_metrics:
            await show_agent_metrics(httpx_client, agent)

        # Run benchmark if requested
        if benchmark:
            await run_benchmark(client, session)
            return

        continue_loop = True
        conversation_count = 0

        print("\n💬 A2A Chat Interface")
        print("✨ Features: RAG Search, Web Search, Performance Metrics")
        print("📝 Type ':help' for commands, ':q' to quit\n")

        while continue_loop:
            conversation_count += 1
            print(f'🔄 ===== Conversation {conversation_count} =====')
            
            continue_loop, contextId, taskId = await complete_task(
                client,
                use_push_notifications,
                push_notification_receiver,
                None,
                None,
                None
            )

            if history and continue_loop and taskId:
                print('\n📚 ===== Conversation History =====')
                try:
                    task_response = await client.get_task({'id': taskId, 'historyLength': 5})
                    print(json.dumps(task_response.dict(), indent=2, default=str))
                except Exception as e:
                    print(f"❌ Failed to get history: {e}")

async def complete_task(
    client: A2AClient,
    use_push_notifications: bool,
    notification_receiver_host: str,
    notification_receiver_port: int,
    taskId,
    contextId,
):
    """task completion with timing and metrics"""
    
    # Special commands
    prompt = click.prompt(
        '\n💭 What would you like to ask? (:help for commands, :q to quit)'
    )
    
    if prompt == ':q' or prompt == 'quit':
        return False, None, None
    elif prompt == ':help':
        print_help()
        return True, contextId, taskId
    elif prompt == ':metrics':
        await show_agent_metrics(client._client, client.agent_card.url)
        return True, contextId, taskId
    
    start_time = time.time()
    
    message = Message(
        role='user',
        parts=[TextPart(text=prompt)],
        messageId=str(uuid4()),
        taskId=taskId,
        contextId=contextId,
    )

    # File attachment option
    file_path = click.prompt(
        '📎 Attach a file? (press enter to skip)',
        default='',
        show_default=False,
    )
    
    if file_path and file_path.strip() and os.path.exists(file_path.strip()):
        try:
            with open(file_path.strip(), 'rb') as f:
                file_content = base64.b64encode(f.read()).decode('utf-8')
                file_name = os.path.basename(file_path.strip())

            message.parts.append(
                Part(
                    root=FilePart(
                        file=FileWithBytes(
                            name=file_name, 
                            bytes=file_content
                        )
                    )
                )
            )
            print(f"📎 Attached file: {file_name}")
        except Exception as e:
            print(f"❌ Failed to attach file: {e}")

    payload = MessageSendParams(
        id=str(uuid4()),
        message=message,
        configuration=MessageSendConfiguration(
            acceptedOutputModes=['text'],
        ),
    )

    print("🤔 Processing your request...")
    
    try:
        event = await client.send_message(
            SendMessageRequest(
                id=str(uuid4()),
                params=payload,
            )
        )
        
        processing_time = time.time() - start_time
        
        if hasattr(event, 'root') and hasattr(event.root, 'result'):
            event = event.root.result
        else:
            print(f"❌ Unexpected response format: {event}")
            return False, contextId, taskId
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False, contextId, taskId

    if not contextId:
        contextId = event.context_id
    if isinstance(event, Task):
        if not taskId:
            taskId = event.id
        taskResult = event
    elif isinstance(event, Message):
        message = event

    # Display results with formatting
    if message:
        print(f'\n🤖 Agent Response (⏱️ {processing_time:.2f}s):')
        print('─' * 50)
        print(message.parts[0].root.text if message.parts else "No response")
        print('─' * 50)
        print("\nRaw JSON Response:")
        print(json.dumps(message.dict(), indent=2, default=str))  # Dump the raw JSON
        print('─' * 50)
        return True, contextId, taskId
        
    if taskResult:
        print(f'\n📋 Task Result (⏱️ {processing_time:.2f}s):')
        print('─' * 50)
        
        # Extract and display content
        if taskResult.artifacts:
            for artifact in taskResult.artifacts:
                if artifact.parts:
                    for part in artifact.parts:
                        if hasattr(part.root, 'text'):
                            print(part.root.text)
        
        print('─' * 50)
        
        state = TaskState(taskResult.status.state)
        if state.name == TaskState.input_required.name:
            print("🔄 More input required, continuing conversation...")
            return await complete_task(
                client,
                use_push_notifications,
                notification_receiver_host,
                notification_receiver_port,
                taskId,
                contextId,
            )
        
        print("✅ Task completed successfully!")
        return True, contextId, taskId
    
    return True, contextId, taskId

async def show_agent_metrics(httpx_client, agent_url):
    """Show agent performance metrics"""
    try:
        base_url = agent_url.rstrip('/')
        response = await httpx_client.get(f"{base_url}/metrics")
        
        if response.status_code == 200:
            metrics = response.json()
            print("\n📊 ===== Agent Performance Metrics =====")
            
            agent_metrics = metrics.get('agent_metrics', {})
            print(f"📈 Total Requests: {agent_metrics.get('total_requests', 0)}")
            print(f"✅ Success Rate: {agent_metrics.get('success_rate', 0):.2%}")
            print(f"⏱️ Avg Response Time: {agent_metrics.get('average_response_time', 0):.2f}s")
            print(f"💾 Cache Hit Rate: {agent_metrics.get('cache_hit_rate', 0):.2%}")
            
            config = metrics.get('config', {})
            print(f"🔍 RAG Enabled: {'✅' if config.get('rag_enabled') else '❌'}")
            print(f"🌐 Web Search: {'✅' if config.get('web_search_enabled') else '❌'}")
            print(f"💾 Caching: {'✅' if config.get('caching_enabled') else '❌'}")
            print("=" * 40)
        else:
            print(f"❌ Failed to get metrics: {response.status_code}")
    except Exception as e:
        print(f"❌ Metrics request failed: {e}")

async def run_benchmark(client: A2AClient, session: str):
    """Run benchmark test"""
    print("\n🏃 Running benchmark test...")
    
    test_queries = [
        "Thông tin về iPhone 15",
        "Cửa hàng ở Hoàng Mai",
        "Điện thoại Samsung Galaxy S24",
        "Tin tức công nghệ mới nhất"
    ]
    
    total_time = 0
    successful_requests = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"🧪 Test {i}: {query}")
        start_time = time.time()
        
        try:
            message = Message(
                role='user',
                parts=[TextPart(text=query)],
                messageId=str(uuid4()),
                contextId=f"benchmark_{i}"
            )
            
            payload = MessageSendParams(
                id=str(uuid4()),
                message=message,
                configuration=MessageSendConfiguration(acceptedOutputModes=['text'])
            )
            
            await client.send_message(SendMessageRequest(id=str(uuid4()), params=payload))
            
            request_time = time.time() - start_time
            total_time += request_time
            successful_requests += 1
            
            print(f"   ✅ Completed in {request_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        # Small delay between requests
        await asyncio.sleep(0.5)
    
    print(f"\n📊 Benchmark Results:")
    print(f"   📈 Successful: {successful_requests}/{len(test_queries)}")
    print(f"   ⏱️ Total Time: {total_time:.2f}s")
    print(f"   ⚡ Avg Time: {total_time/max(1, successful_requests):.2f}s/request")

def print_help():
    """Print help information"""
    print("\n📚 ===== A2A Client Help =====")
    print("🔤 Commands:")
    print("   :q or quit - Exit the application")
    print("   :help - Show this help message")
    print("   :metrics - Show agent performance metrics")
    print("\n✨ Features:")
    print("   • 🔍 RAG-powered product search")
    print("   • 🏪 Smart shop information")
    print("   • 🌐 Web search integration")
    print("   • 📎 File attachment support")
    print("   • 📊 Performance monitoring")
    print("   • 💾 Intelligent caching")
    print("\n💡 Example queries:")
    print("   • 'Thông tin chi tiết về iPhone 15 Pro Max'")
    print("   • 'Cửa hàng gần nhà tôi ở Hoàng Mai'")
    print("   • 'So sánh Samsung Galaxy S24 và iPhone 15'")
    print("   • 'Tin tức công nghệ mới nhất hôm nay'")
    print("=" * 40)

if __name__ == '__main__':
    asyncio.run(cli())
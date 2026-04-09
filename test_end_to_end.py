import sys
import os
import traceback
import asyncio
import queue
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import config

async def run_direct():
    """직접 파이프라인의 async generator를 실행하여 에러 확인"""
    from agents import get_pipeline
    from google.genai.types import Content, Part

    pipeline = get_pipeline(api_key=config.GOOGLE_API_KEY)
    query = "disulfide bond를 어떻게 연구할방법을 알려줘"

    # Manually create a minimal invocation context to simulate run_async
    from google.adk.runners import InMemoryRunner
    from google.adk.sessions.in_memory_session_service import InMemorySessionService

    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name="test", user_id="u1")

    from google.adk.agents.invocation_context import InvocationContext, new_invocation_context_id
    from google.adk.plugins.plugin_manager import PluginManager

    ctx = InvocationContext(
        invocation_id=new_invocation_context_id(),
        agent=pipeline,
        session=session,
        session_service=session_service,
        plugin_manager=PluginManager(),
        user_content=Content(parts=[Part(text=query)]),
    )

    print("Running pipeline directly via async generator...")
    try:
        async for event in pipeline._run_async_impl(ctx):
            print("\n============= FINAL OUTPUT =============")
            if event.content and event.content.parts:
                print(event.content.parts[0].text[:2000])
    except Exception as e:
        print(f"\n!!! EXCEPTION in pipeline: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_direct())

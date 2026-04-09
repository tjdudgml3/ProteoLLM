import asyncio
import json
import os
import sys
from unittest.mock import MagicMock, AsyncMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.agents import create_agents, PROMPTS

# Mock Context
class MockSession:
    def __init__(self):
        self.id = "mock_session_id"
        self.state = {}

class MockContext:
    def __init__(self, user_content=""):
        self.user_content = user_content
        self.session = MockSession()
        self.history = []
        self.invocation_id = "mock_invocation_id"
        self.end_invocation = False
        self.plugin_manager = MagicMock()
        self.plugin_manager.run_before_agent_callback = AsyncMock(return_value=None)
        self.plugin_manager.run_after_agent_callback = AsyncMock(return_value=None)
        self.plugin_manager.run_before_tool_callback = AsyncMock(return_value=None)
        self.plugin_manager.run_after_tool_callback = AsyncMock(return_value=None)
        self.plugin_manager.run_after_model_callback = AsyncMock(return_value=None)

    def model_copy(self, update=None):
        new_content = update.get('user_content', self.user_content) if update else self.user_content
        new_ctx = MockContext(user_content=new_content)
        new_ctx.session = self.session
        new_ctx.history = self.history
        new_ctx.invocation_id = self.invocation_id
        new_ctx.plugin_manager = self.plugin_manager
        return new_ctx

async def main():
    print("🚀 Starting Filter Agent Test (using agent.run_async)...")
    
    # 1. Initialize Agents
    if not os.environ.get("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not found in environment.")
        return

    agents = create_agents()
    filter_agent = agents["filter"]
    print("✅ Filter Agent initialized.")

    # 2. Load Prompt
    if os.path.exists("temp_full_text.txt"):
        with open("temp_full_text.txt", "r") as f:
            prompt = f.read()
    else:
        print("❌ temp_full_text.txt not found.")
        return

    print(f"📝 Prompt length: {len(prompt)} chars")

    # 3. Run Agent using run_async
    context = MockContext(user_content=prompt)
    
    print("🏃 Running Filter Agent...")
    
    try:
        async for event in filter_agent.run_async(context):
            # Print event for debugging
            # print(f"Event: {event}")
            
            # Check for content
            if hasattr(event, 'content') and hasattr(event.content, 'parts'):
                for part in event.content.parts:
                    if hasattr(part, 'text'):
                        print(part.text)
            elif isinstance(event, str):
                print(event)
            
    except Exception as e:
        print(f"❌ Execution Error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n✅ Test Completed.")

if __name__ == "__main__":
    asyncio.run(main())

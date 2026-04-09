import sys
import os

# Add src to python path so we can import tools and agents
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import tools
import asyncio
from agents import get_pipeline
from google.adk.events import Event
from google.genai.types import Content, Part
from google.adk.agents.callback_context import CallbackContext

async def test_search():
    print("Testing Wiki Retrieval Tool Directly...")
    res = tools.search_wiki("disulfide bond", 2)
    print("Search Result characters:", len(res))
    
    print("\n----------------------")
    print("Testing Full Pipeline...")
    pipeline = get_pipeline()
    
    # Mock context
    class MockContext(CallbackContext):
        def __init__(self, query):
            super().__init__(None)
            self.user_content = Content(parts=[Part(text=query)])
            self.all_events = []
            self._next_call_index = 0
            self.history = []
            
    ctx = MockContext("disulfied bond를 어떻게 연구할방법을 알려줘")
    async for event in pipeline._run_async_impl(ctx):
        print(f"\nFinal Event Output:\n")
        print(event.content.parts[0].text[:500] if event.content.parts else "No output")

if __name__ == "__main__":
    asyncio.run(test_search())

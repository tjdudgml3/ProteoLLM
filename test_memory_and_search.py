import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

import asyncio
from unittest.mock import MagicMock, patch
from agents import BioinformaticsPipeline, create_agents
from pipeline import run_adk_pipeline
import queue

async def test_session_memory():
    print("Testing Session Memory Logic...")
    # Mock agents dictionary
    agents = create_agents()
    pipeline = BioinformaticsPipeline(agents)
    
    # pipeline.py now sends a JSON string
    import json
    input_payload = {
        "query": "Tell me more about it.",
        "chat_history": [
            {"role": "user", "content": "What is breast cancer?"},
            {"role": "assistant", "content": "Breast cancer is a type of cancer..."}
        ]
    }
    payload_str = json.dumps(input_payload)
    context.user_content = payload_str
    
    # We want to check if prompt_with_context is formed correctly in _run_async_impl
    gen = pipeline._run_async_impl(context)
    
    # Mock _invoke_sub_agent to see what prompt it receives
    with patch.object(BioinformaticsPipeline, '_invoke_sub_agent', return_value="analyst") as mock_invoke:
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            pass
        
        # Check if the first call to router has the history
        args, kwargs = mock_invoke.call_args_list[0]
        prompt_received = args[1]
        print(f"Prompt sent to Router: \n{prompt_received}")
        assert "Breast cancer" in prompt_received
        assert "Tell me more about it." in prompt_received
        print("✅ Session Memory Logic Verified.")

async def test_internet_routing():
    print("\nTesting Internet Routing Logic...")
    agents = create_agents()
    pipeline = BioinformaticsPipeline(agents)
    
    context = MagicMock()
    import json
    payload_str = json.dumps({"query": "What are the latest news from 2025?", "chat_history": []})
    context.user_content = payload_str
    
    # Mock router to return 'internet'
    with patch.object(BioinformaticsPipeline, '_invoke_sub_agent') as mock_invoke:
        mock_invoke.side_effect = ["internet", "search query", "internet response"]
        
        with patch('tools.search_internet', return_value="Search results...") as mock_search:
            gen = pipeline._run_async_impl(context)
            response = await gen.__anext__()
            
            print(f"Response: {response.content.parts[0].text}")
            assert response.content.parts[0].text == "internet response"
            mock_search.assert_called_once()
            print("✅ Internet Routing Logic Verified.")

if __name__ == "__main__":
    asyncio.run(test_session_memory())
    asyncio.run(test_internet_routing())

```python
import config
import asyncio
import sys
from google.adk.runners import InMemoryRunner
from google.adk.plugins.base_plugin import BasePlugin
from agents import get_pipeline

# Create pipeline instance
bioinformatics_pipeline = get_pipeline()

# Debug Plugin to inspect response structure
class DebugPlugin(BasePlugin):
    def __init__(self):
        super().__init__(name="debug_plugin")

    async def after_model_callback(self, *, callback_context, llm_response) -> None:
        print(f"\n[DEBUG] Model Response Type: {type(llm_response)}")
        try:
            print(f"[DEBUG] Response Content: {llm_response}")
        except:
            pass

    async def before_tool_callback(self, *, callback_context, tool, tool_args) -> None:
        print(f"\n[DEBUG] Tool Call: {tool.name} with args: {tool_args}")

async def main():
    print("Starting Pipeline Test...")
    runner = InMemoryRunner(agent=bioinformatics_pipeline, plugins=[DebugPlugin()])
    
    query = "phospho id가 생각보다안되네"
    print(f"Query: {query}")
    
    try:
        # run_debug prints internal steps to stdout
        response = await runner.run_debug(query)
        print("\n--- Final Response Object ---")
        print(f"Type: {type(response)}")
        print(f"Dir: {dir(response)}")
        print(response)
        
        # Test parsing logic
        print("\n--- Parsed Text ---")
        text = "No text"
        try:
            llm_response = response
            # Handle list
            if isinstance(llm_response, list):
                if len(llm_response) > 0:
                    llm_response = llm_response[-1]
            
            if hasattr(llm_response, 'content') and llm_response.content and hasattr(llm_response.content, 'parts'):
                parts = llm_response.content.parts
                if parts:
                    texts = []
                    for part in parts:
                        if hasattr(part, 'text') and part.text:
                            texts.append(part.text)
                        elif hasattr(part, 'function_call') and part.function_call:
                            texts.append(f"Function Call: {part.function_call.name}")
                    text = "\n".join(texts) if texts else "No text content"
            elif hasattr(llm_response, 'candidates') and llm_response.candidates:
                 if llm_response.candidates[0].content.parts:
                    text = llm_response.candidates[0].content.parts[0].text or "Function Call"
            elif isinstance(llm_response, str):
                text = llm_response
        except Exception as e:
            text = f"Error parsing: {e}"
        print(f"Parsed: {text}")

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

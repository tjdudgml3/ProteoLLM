import asyncio
import queue
from google.adk.runners import InMemoryRunner
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from agents import get_pipeline

def _parse_response_text(llm_response):
    text = "No text"
    if llm_response:
        try:
            # Handle list of Events (from run_debug)
            if isinstance(llm_response, list):
                # Get the last event
                if len(llm_response) > 0:
                    last_event = llm_response[-1]
                    # Recursively parse the content of the last event
                    return _parse_response_text(last_event)
            
            # Check for content directly (ADK version dependent)
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
            # Fallback for candidates structure
            elif hasattr(llm_response, 'candidates') and llm_response.candidates:
                 if llm_response.candidates[0].content.parts:
                    text = llm_response.candidates[0].content.parts[0].text or "Function Call"
            # Fallback for string
            elif isinstance(llm_response, str):
                text = llm_response
        except Exception as e:
            text = f"Error parsing response: {e}"
    return text

class StreamlitEventPlugin(BasePlugin):
    def __init__(self, event_queue: queue.Queue):
        super().__init__(name="streamlit_event_plugin")
        self.event_queue = event_queue

    async def before_agent_callback(self, **kwargs) -> None:
        agent = kwargs.get('agent')
        if agent:
            self.event_queue.put({
                "step": "AGENT_START",
                "content": f"Starting Agent: {agent.name}",
                "data": {"agent": agent.name}
            })

    async def after_agent_callback(self, **kwargs) -> None:
        agent = kwargs.get('agent')
        if agent:
            self.event_queue.put({
                "step": "AGENT_END",
                "content": f"Completed Agent: {agent.name}",
                "data": {"agent": agent.name}
            })

    async def before_model_callback(self, **kwargs) -> None:
        llm_request = kwargs.get('llm_request')
        if llm_request:
            self.event_queue.put({
                "step": "LLM_REQUEST",
                "content": f"Requesting {llm_request.model}",
                "data": {"request": str(llm_request)}
            })

    async def after_model_callback(self, **kwargs) -> None:
        llm_response = kwargs.get('llm_response')
        text = _parse_response_text(llm_response)

        self.event_queue.put({
            "step": "LLM_RESPONSE",
            "content": f"Model Response received",
            "data": {"response": text}
        })
        
    async def before_tool_callback(self, **kwargs) -> None:
         tool = kwargs.get('tool')
         tool_args = kwargs.get('tool_args', 'N/A')
         if tool:
             self.event_queue.put({
                "step": "TOOL_START",
                "content": f"Executing Tool: {tool.name}",
                "data": {"tool": tool.name, "args": str(tool_args)}
            })

async def run_adk_pipeline(query: str, event_queue: queue.Queue, api_key: str = None):
    """
    Runs the ADK pipeline and pushes events to the queue.
    """
    plugin = StreamlitEventPlugin(event_queue)
    
    # Create pipeline with user's API key
    pipeline_agent = get_pipeline(api_key)
    
    # Attach plugin to the pipeline agent so it can propagate to sub-agents
    pipeline_agent.plugins = [plugin]
    
    runner = InMemoryRunner(agent=pipeline_agent, plugins=[plugin])
    
    # We use run_debug because it's a high-level convenience method that handles session creation
    # and takes a string input. We capture the return value.
    # Note: run_debug prints to stdout, but our plugin will capture events.
    
    try:
        response = await runner.run_debug(query)
        # print(f"DEBUG: run_debug response type: {type(response)}")
        # print(f"DEBUG: run_debug response: {response}")
        
        # Signal completion
        final_text = _parse_response_text(response)
        # print(f"DEBUG: parsed final_text: {final_text}")
        
        event_queue.put({
            "step": "FINAL",
            "content": "Pipeline Completed",
            "data": {"final_output": final_text}
        })
    except Exception as e:
        event_queue.put({
            "step": "ERROR",
            "content": f"Error: {str(e)}",
            "data": {"error": str(e)}
        })

def run_system_generator(query: str, api_key: str = None):
    """
    Generator that yields events for Streamlit.
    """
    event_queue = queue.Queue()
    
    # Run the async pipeline in a separate thread or event loop
    # Since Streamlit runs in a specific way, we need to be careful with async.
    # We will use a helper to run async in a thread and yield from queue.
    
    import threading
    
    def target():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_adk_pipeline(query, event_queue, api_key))
            loop.close()
        except Exception as e:
            event_queue.put({
                "step": "ERROR",
                "content": f"Critical Thread Error: {str(e)}",
                "data": {"error": str(e)}
            })
        finally:
            # Signal end of queue
            event_queue.put(None)

    thread = threading.Thread(target=target)
    thread.start()
    
    while True:
        event = event_queue.get()
        if event is None:
            break
        yield event
        
    thread.join()

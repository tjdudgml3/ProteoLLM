import asyncio
import sys
from google.adk.runners import InMemoryRunner
from agents import get_pipeline, save_log
import time


# Create pipeline instance
bioinformatics_pipeline = get_pipeline()

# ==============================================================================
# Test Script for BioinformaticsPipeline
# ==============================================================================

async def main():
    print("🧪 Starting BioinformaticsPipeline Test...")
    
    
    query = "phospho id가 생각보다안되네"

    # InMemoryRunner handles the agent execution loop
    runner = InMemoryRunner(agent=bioinformatics_pipeline)
    t_start = time.perf_counter()
    try:
        
        # run_debug returns a list of events/responses
        print("🏃 Running pipeline... (this may take a minute)")
        responses = await runner.run_debug(query)
        
        # Extract and Print Final Output
        print("\n✅ Pipeline Completed!")
        print("-" * 60)
        
        final_text = "No output"
        
        if responses and isinstance(responses, list):
            last_response = responses[-1]
            if hasattr(last_response, 'candidates') and last_response.candidates:
                final_text = last_response.candidates[0].content.parts[0].text
            elif hasattr(last_response, 'content') and last_response.content.parts:
                final_text = last_response.content.parts[0].text
            elif isinstance(last_response, str):
                final_text = last_response
                
        print(final_text)
        print("-" * 60)
        save_log(f"total_time@@@@@@@@@@@@@@@@")
        save_log(f"Total time: {time.perf_counter() - t_start}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

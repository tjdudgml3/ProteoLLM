from google.genai.types import GenerateContentResponse, Candidate, Content, Part
import uuid
from datetime import datetime, timezone

try:
    final_response = GenerateContentResponse(candidates=[Candidate(content=Content(parts=[Part(text="test")]), finish_reason="STOP", index=0)])
    print("Created response object")
    
    object.__setattr__(final_response, 'partial', None)
    print("Set partial attribute")
    
    print(f"Partial value: {final_response.partial}")

except Exception as e:
    print(f"Error: {e}")

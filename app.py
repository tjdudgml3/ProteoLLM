import streamlit as st
import time
from pipeline import run_system_generator

st.set_page_config(page_title="Phospho-Proteomics ADK Agent", layout="wide")

st.title("🧬 Bioinformatics Multi-Agent System")
st.markdown("Developed By YHS")

# Sidebar for Observability & Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google API Key", type="password", help="Enter your Google API Key to proceed.")
    
    st.divider()
    
    # Real-time logs container
    st.header("Agent Observability")
    log_placeholder = st.empty()
    
# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not api_key:
    st.warning("Please enter your Google API Key in the sidebar to continue.")
    st.stop()

if prompt := st.chat_input("Ask about bioinformatics (e.g., 'Find targets for breast cancer')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Store logs for the current session to prevent overwriting in the loop
        session_logs = []
        
        # Run the pipeline
        with st.status("Agents working...", expanded=True) as status:
            try:
                for event in run_system_generator(prompt, api_key=api_key):
                    if not event: continue
                    
                    step = event.get("step")
                    content = event.get("content")
                    data = event.get("data", {})
                    
                    # Accumulate logs to maintain history in the sidebar
                    session_logs.append(event)
                    
                    # Re-render the entire log history in the sidebar container
                    # Using a fixed height to enable scrolling for long logs
                    with log_placeholder.container(height=500):
                        for log_event in session_logs:
                            l_step = log_event.get("step")
                            l_content = log_event.get("content")
                            l_data = log_event.get("data", {})
                            
                            if l_step == "LLM_REQUEST":
                                with st.expander(f"📤 Input: {l_content}", expanded=False):
                                    st.code(l_data.get("request", "No details"))
                            elif l_step == "LLM_RESPONSE":
                                with st.expander(f"📥 Output: {l_content}", expanded=False):
                                    st.markdown(l_data.get("response", "No details"))
                            elif l_step == "AGENT_START":
                                st.info(f"🤖 **{l_content}**")
                            elif l_step == "AGENT_END":
                                st.success(f"✅ **{l_content}**")
                            elif l_step == "TOOL_START":
                                with st.expander(f"🛠️ Tool: {l_content}", expanded=False):
                                    st.code(l_data.get("args", "No args"))
                            else:
                                st.markdown(f"**[{l_step}]** {l_content}")
                    
                    # Update status indicator in the main chat area
                    status.update(label=f"{step}: {content}")
                    
                    # Handle final output
                    if step == "FINAL":
                        full_response = data.get("final_output", "No output.")
                    
                    # Handle Error
                    if step == "ERROR":
                        full_response = f"An error occurred: {data.get('error', 'Unknown error')}"
                        status.update(label="Error!", state="error")
                    
                    # Handle intermediate LLM responses (optional streaming effect)
                    if step == "LLM_RESPONSE":
                        pass

                if not full_response:
                    full_response = "No response generated. Please check logs."
                    
                status.update(label="Complete!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="System Error", state="error")
                st.error(f"System Error: {e}")
        
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Run with: streamlit run src/app.py

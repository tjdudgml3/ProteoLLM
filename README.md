# 🧬 BioLLM: Bioinformatics Multi-Agent System

BioLLM is a sophisticated bioinformatics research assistant powered by a multi-agent architecture. It integrates local literature databases (via FAISS vector search) and real-time internet search to provide comprehensive answers for researchers and bioinformaticians.

## 🚀 Key Features

*   **Multi-Agent Orchestration**: Specialized agents for routing, contract building, literature retrieval, advice generation, and quality assurance.
*   **Dual Intelligence Pipelines**: 
    *   **Literature Pipeline**: Searches a curated database of scientific papers (2020-2025) to provide evidence-based protocols and analysis plans.
    *   **Internet Pipeline**: Real-time web search for recent news, general scientific inquiries, and follow-up details not found in static papers.
*   **Session Memory**: Maintains conversation context across multiple turns, allowing for fluid follow-up questions.
*   **Observability**: Real-time tracking of agent activities (inputs, outputs, tool calls) via a Streamlit sidebar.
*   **Automated Validation**: Integrated QA agent ensures every response contains proper citations and meets quality standards.

## 🏗️ System Architecture

1.  **Router Agent**: Classifies user intent into `experimenter`, `analyst`, or `internet`.
2.  **Contract Builder**: Extracts technical metadata from the query (assay type, organism, sample type).
3.  **Literature/Search Agent**: Generates optimized queries for vector databases or web search.
4.  **Advisor Agents**: High-reasoning agents that synthesize retrieved information into actionable advice.
5.  **QA Agent**: Validates the final output for accuracy and citations.

## 🛠️ Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/tjdudgml3/ProteoLLM.git
    cd ProteoLLM
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Environment**:
    Add your Google API Key to `src/config.py` or export it:
    ```bash
    export GOOGLE_API_KEY="your-google-api-key"
    ```

## 📅 Usage

Run the Streamlit application:
```bash
streamlit run src/app.py
```

## 📂 Project Structure

*   `src/app.py`: Streamlit frontend and chat interface.
*   `src/pipeline.py`: Logic for running the multi-agent system.
*   `src/agents.py`: Agent definitions and prompt templates.
*   `src/tools.py`: Search and retrieval utilities.
*   `src/vector_db.py`: FAISS vector database management.
*   `src/config.py`: Configuration and SSL patches.

## 📜 Requirements

*   Python 3.9+
*   Google Gemini API Key
*   FAISS (for local search)
*   Streamlit

---
Developed by YHS.

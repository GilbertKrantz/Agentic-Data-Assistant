# Multi-Agent Fraud Detection System

A sophisticated multi-agent AI system for fraud detection and analysis, powered by LangChain, Google Gemini, DuckDB, and ChromaDB. This system decomposes complex fraud analysis queries and delegates specialized tasks to intelligent agents that work collaboratively to provide accurate, cited insights.

## 🌟 Features

- **Multi-Agent Architecture**: Specialized agents for data analysis, retrieval, validation, and formatting
- **Advanced Data Access**: Query structured data via DuckDB with integrated Python code execution
- **Vector Search**: Semantic document retrieval using ChromaDB with Google embeddings
- **Intelligent Orchestration**: Central orchestrator agent that decomposes queries and coordinates specialized agents
- **Evidence-Based Answers**: All findings wrapped with citations and confidence scores
- **Parallel Processing**: Concurrent execution of compatible agents for improved performance
- **Fraud Detection**: Real-world fraud detection datasets with comprehensive analysis capabilities

### Agent Responsibilities

- **OrchestratorAgent**: Decomposes user queries, manages task delegation, and coordinates agent collaboration
- **DataScientistAgent**: Analyzes structured fraud data using SQL queries and Python code execution
- **RetrieverAgent**: Fetches relevant documents and facts from the knowledge base
- **ValidatorAgent**: Verifies findings and assesses confidence levels
- **SourceFormatterAgent**: Formats responses with proper citations and sources

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Google Gemini API key
- uv for dependency management

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Associate-AI-Engineer
   ```

2. **Create environment file**
   ```bash
   cp .env.example .env
   ```

3. **Configure your API keys in `.env`**
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   CHROMADB_API_KEY=your_chromadb_api_key_here
   LANGSMITH_API_KEY=your_langsmith_api_key_here
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT_NAME=AI-Data-Assistant
   ```

4. **Install dependencies**
   ```bash
   uv sync
   ```

5. **Initialize the database** (if needed)
   ```bash
   python app/ingest.py
   ```

### Running the Application

**Basic Usage:**
```bash
uv run app/main.py
```

This runs a predefined set of test queries and displays the results.

**Interactive UI** (Streamlit):
```bash
uv run streamlit run app/streamlit_app.py
```

## 📁 Project Structure

```
Associate-AI-Engineer/
├── app/
│   ├── agents/               # Agent implementations
│   │   ├── orchestrator/     # Central orchestrator agent
│   │   ├── data_scientist_agent/  # SQL & analysis agent
│   │   ├── retriever_agent/  # Document retrieval agent
│   │   ├── validator_agent/  # Answer validation agent
│   │   └── source_formatter_agent/ # Response formatting agent
│   ├── database/             # Data access layer
│   │   ├── duckdb_utils.py   # DuckDB operations
│   │   ├── chroma_utils.py   # ChromaDB operations
│   │   └── data_utils.py     # Query engine
│   ├── config.py             # Configuration management
│   ├── models.py             # Data models & UniversalEvidenceObject
│   ├── main.py               # Entry point
│   ├── ingest.py             # Data ingestion
│   └── streamlit_app.py      # Web UI
├── data/
│   ├── fraudTrain.csv        # Training dataset
│   └── fraudTest.csv         # Test dataset
├── pyproject.toml            # Project metadata & dependencies
└── README.md                 # This file
```

## ⚙️ Configuration

Configuration is managed through `app/config.py` using Pydantic-settings. Key environment variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | Yes |
| `CHROMADB_API_KEY` | ChromaDB API key | Yes |
| `LANGSMITH_API_KEY` | LangSmith API key for tracing | Optional |
| `LANGSMITH_TRACING` | Enable/disable LangSmith tracing | Optional (defaults to `false`) |
| `LANGSMITH_PROJECT_NAME` | LangSmith project name | Optional (defaults to `AI-Data-Assistant`) |
| `LANGSMITH_ENDPOINT` | LangSmith API endpoint | Optional (defaults to `https://api.smith.langchain.com`) |

Load your configuration:

```python
from app.config import Settings
settings = Settings()
```

## 🧩 Key Concepts

### UniversalEvidenceObject
All agents communicate findings using `UniversalEvidenceObject`, which includes:
- The analysis/finding
- Source references
- Confidence scores
- Metadata

### Agent Communication
All agents return `StandardAgentResponse` containing:
- Main response
- Supporting evidence
- Confidence metrics

### Evidence Pool
The orchestrator maintains a global evidence pool that accumulates findings across all agents during a session. This ensures consistency and prevents duplicate work.

## 🗄️ Data Storage

### DuckDB (Structured Data)
- **Location**: `data/fraud.db`
- **Main Tables**: 
  - `fraud_train`: Training dataset
  - `fraud_test`: Test dataset
- **Access**: Via `app/database/duckdb_utils.py`

### ChromaDB (Unstructured Data)
- **Type**: Vector database for semantic search
- **Embedding Model**: Google Generative AI (`models/gemini-embedding-001`)
- **Collection**: `documents_new`
- **Access**: Via `app/database/chroma_utils.py`

## 📦 Dependencies

Key dependencies (see `pyproject.toml` for full list):

- `langchain` - LLM orchestration framework
- `langchain-google-genai` - Google Gemini integration
- `duckdb-engine` - DuckDB SQL integration
- `chromadb` - Vector database
- `pydantic` (>2.0) - Data validation
- `sqlalchemy` - Database ORM
- `streamlit` - Web UI framework


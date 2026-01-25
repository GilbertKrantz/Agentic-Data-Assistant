# Codebase Instructions

This project is a **Multi-Agent Fraud Detection System** built with Python, LangChain, Google Gemini, DuckDB, and ChromaDB.

## 🏗 Big Picture Architecture

- **Orchestration**: `OrchestratorAgent` (`app/agents/orchestrator`) is the central brain. It decomposes user queries and delegates tasks to specialized agents.
- **Agents**:
  - `DataScientistAgent`: Analyzes structured data using SQL (DuckDB) and integrated Python code execution.
  - `RetrieverAgent`: Fetches documents and facts from knowledge bases.
  - `SourceFormatterAgent`: Formats output with citations.
  - `ValidatorAgent`: Verifies answers and checks confidence.
- **Data Flow**:
  - Agents exchange information using `UniversalEvidenceObject` (`app/models.py`).
  - Evidence is accumulated in a global pool managed by `app/agents/orchestrator/tools.py`.
- **Storage**:
  - **Structured**: DuckDB (`data/fraud.db`). Main tables are in the `main` schema (e.g., `fraud_train`).
  - **Unstructured**: ChromaDB (Cloud Client). Collection: `documents_new`.

## 🚀 Critical Workflows

- **Run the Application**: The entry point is `app/main.py`.
  ```bash
  python app/main.py
  ```
  This runs a predefined set of test queries.

- **Agent Development**:
  - Each agent lives in `app/agents/<agent_name>/`.
  - **Components**: `agent.py` (logic), `tools.py` (capabilities), `prompts.py` (instructions).
  - **Middleware**: Use `AgentMiddleware` to inject skill-specific instructions into system prompts (e.g., `DataScienceSkillMiddleware` in `app/agents/data_scientist_agent/agent.py`).

## 🧩 Project Patterns & Conventions

### 1. Agent Communication
- **Standard Response**: All agents must return `StandardAgentResponse`.
- **Evidence**: Always wrap findings in `UniversalEvidenceObject`.
- **Tools**: Define agent tools in `tools.py` using `@tool` decorator.

### 2. Data Access
- **DuckDB**: Access via `app/database/duckdb_utils.py`.
  - Use `QueryEngine` in `app/database/data_utils.py` for executing SQL.
  - **Warning**: `DataScientistAgent` uses Gemini's *integrated* code execution environment for Python logic, but custom tools for SQL access.
- **ChromaDB**: Access via `app/database/chroma_utils.py`.
  - Uses `GoogleGenerativeAiEmbeddingFunction` (`models/gemini-embedding-001`).

### 3. Configuration
- Manage settings in `app/config.py` using `pydantic-settings`.
- Environment variables are loaded from `.env`.

### 4. Middleware Pattern
- Use Middleware classes (inheriting from `AgentMiddleware`) to dynamically modify agent system prompts based on available tools. See `OrchestratorSkillMiddleware` for reference.

## 📦 Dependencies
- `langchain`, `langchain-google-genai` for AI logic.
- `duckdb-engine`, `sqlalchemy` for database interactions.
- `chromadb` for vector search.
- `pydantic` (>2.0) for data validation.

## ⚠️ Gotchas
- **Parallel Execution**: The Orchestrator can run Retriever and Data Scientist agents in parallel (`call_parallel_retrieval_and_analysis`). Ensure tools are thread-safe if modifying shared state.
- **Evidence Pool**: The evidence pool is global per session (`_accumulated_evidence` in `orchestrator/tools.py`). Ensure it is reset between independent runs if implementing a server.

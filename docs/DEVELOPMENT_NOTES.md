# 🛠️ Development Guidelines

**Welcome to the Nexus LLM Analytics codebase!**

This document provides essential information for developers contributing to the project.

---

## 🏗️ Environment Setup

1.  **Python Environment**:
    ```bash
    python -m venv env
    source env/bin/activate  # Windows: env\Scripts\activate
    pip install -r requirements.txt
    ```

2.  **Frontend Environment**:
    ```bash
    cd src/frontend
    npm install
    ```

3.  **Local LLM Service**:
    Ensure [Ollama](https://ollama.ai) is running:
    ```bash
    ollama serve
    # Required Models
    ollama pull llama3.1:8b
    ollama pull phi3:mini
    ollama pull nomic-embed-text
    ```

---

## 🧪 Testing Strategy

We use `pytest` for the backend. All tests are located in the `tests/` directory.

### Running Tests
```bash
# Run all tests
pytest tests/

# Run unit tests only
pytest tests/unit/

# Run benchmarks (Slow!)
pytest tests/benchmarks/
```

### Writing Tests
*   **Unit Tests**: Mock external dependencies (Ollama, File System).
*   **Integration Tests**: Use the `tests/data` directory for sample files.
*   **Naming**: Test files should start with `test_`.

---

## 🧩 Plugin Development

To add a new specialized agent:

1.  Create a python file in `plugins/` (e.g., `my_new_agent.py`).
2.  Inherit from `AgentInterface`.
3.  Implement `can_handle(query)` and `execute(query, context)`.

**Example:**
```python
from src.backend.agents.agent_interface import AgentInterface

class MyAgent(AgentInterface):
    def can_handle(self, query: str) -> float:
        return 0.9 if "my keyword" in query else 0.0

    def execute(self, query: str, context: dict) -> dict:
        return {"result": "Success!"}
```

---

## 📝 Code Style

*   **Python**: Follow PEP 8. Use Type Hints!.
*   **TypeScript**: Use Prettier and ESLint.
*   **Commits**: Use conventional commits (e.g., `feat: add new chart`, `fix: resolve upload error`).

---

## 📂 Key Architecture Concepts

*   **Query Orchestrator**: The central brain that routes queries (`src/backend/core/engine/query_orchestrator.py`).
*   **Dynamic Planner**: The CoT engine for complex logic (`src/backend/core/engine/self_correction_engine.py`).
*   **Security Sandbox**: All generated code runs here (`src/backend/core/security/sandbox.py`). Never bypass it!

Happy Coding! 🚀
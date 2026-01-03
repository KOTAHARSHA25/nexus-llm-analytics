# Nexus LLM Analytics - Strategic Improvement Roadmap

## 1. Architecture & Infrastructure (The Foundation)

### 🔌 A. True Asynchronous Task Queue
*   **Current State**: Analysis appears to run partially inline or via simple background tasks. In-memory state (`active_analyses` dict) is fragile.
*   **The Fix**: Implement a persistent **Task Queue** (using Redis + Celery or a lightweight SQLite-based queue).
*   **Why**: If the server restarts during a 3-minute analysis, the job is lost. A queue ensures durability and allows scaling workers independently of the API server.

### 🧠 B. The "Semantic Layer" (Data Agnosticism)
*   **Current State**: Agents use hardcoded lists (`['revenue', 'profit']`) to guess column meanings.
*   **The Fix**: Introduce a **Semantic Mapper** step *before* agent routing.
    *   **Action**: On file upload, run a lightweight LLM pass to map `User_Columns` → `Standard_Concepts` (e.g., `gross_inflow` → `revenue`).
    *   **Result**: Agents write logic against `revenue`, and the mapper handles the translation. No more magic strings in the agent code.

### 💾 C. Persistent State Management
*   **Current State**: Session state and conversation history seem file-based or memory-based.
*   **The Fix**: Move to a proper **SQLite/Postgres Database** for:
    *   `Jobs` (Status, Progress)
    *   `Conversations` (History)
    *   `Feedback` (User ratings)

---

## 2. Intelligence & Agents (The Brain)

### 🧩 D. From "Router" to "Planner"
*   **Current State**: `Router` picks *one* best agent (Financial OR Statistical).
*   **The Fix**: Upgrade to a **Planner-Executor Architecture** (e.g., ReAct or LangGraph).
    *   **Scenario**: User asks "Analyze sales trends and predict next month".
    *   **Plan**:
        1.  Call `FinancialAgent` to extract "Sales Trends".
        2.  Call `TimeSeriesAgent` to "Predict Next Month".
        3.  Synthesize results.
*   **Why**: Real-world queries often span multiple domains.

### 🛠️ E. Standardized Tool Protocol
*   **Current State**: Agents seem to have custom implementations for plotting, calculation, etc.
*   **The Fix**: Create a shared **`Tool` Interface**.
    *   `PythonSandboxTool`: Shared by all agents for code execution.
    *   `PlottingTool`: One unified way to generate Plotly JSON.
    *   `VectorSearchTool`: Shared RAG access.
*   **Why**: Reduces code duplication and makes it easier to add new agents.

### 🔍 F. Collaborative Multi-Agent Debate
*   **Current State**: One agent gives one answer.
*   **The Fix**: Implement a **"Critic" Loop**.
    *   Agent generates answer.
    *   `CriticAgent` (different system prompt) reviews it for hallucinations/logic errors.
    *   Agent regenerates if Critic rejects.

---

## 3. Reliability & DX (The Guardrails)

### 🧪 G. Online Benchmark Integration
*   **Current State**: `benchmarks/` folder exists but is offline.
*   **The Fix**: creating a **"Golden Set"** of queries/answers.
    *   Run these automatically in CI/CD (GitHub Actions) to prevent regression when tweaking prompts.
    *   Add **LLM-as-a-Judge** to score the outputs automatically.

### 🛡️ H. Strict Output Structuring (Pydanic)
*   **Current State**: Some agents return dictionaries, others might return text strings.
*   **The Fix**: Enforce **Pydantic Models** for *all* agent outputs.
    *   Ensure every agent returns a structured object: `{ "answer": str, "visualizations": [], "confidence": float, "reasoning": str }`.
    *   Use `instructor` or `outlines` libraries to force LLM JSON compliance.

---

## 4. User Experience (The Touch)

### 👎 I. Feedback Loop (The Flywheel)
*   **Current State**: System fires and forgets.
*   **The Fix**: Add **Thumbs Up/Down** in the UI that saves the (Query, Data, Answer) triplet to a "Fine-tuning Dataset".
*   **Why**: This allows the system to get smarter over time by identifying where it fails.

### ⚡ J. Streaming Responses
*   **Current State**: User waits for the full analysis.
*   **The Fix**: Implement **Token Streaming** (Server-Sent Events) for the "Thought Process".
*   **Why**: Seeing the agent "think" (Planner steps, Python code generation) makes the wait feel shorter and builds trust.

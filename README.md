# 🚀 Nexus LLM Analytics v2.1

**The Intelligent, Privacy-First, Multi-Agent Data Analysis Platform.**

> Transform natural language questions into rigorous data insights using a coordinated swarm of specialized AI agents. All processing runs 100% locally for complete data privacy.

---

## 🎯 **Why Nexus?**

Most "chat with data" tools are wrappers around a single LLM prompt. **Nexus is different.**

It uses an **Intelligent Query Orchestrator** to analyze your request and dynamically route it to the best specialist for the job—whether that's a Python code generator for complex statistics, a SQL expert for querying, or a specialized financial analyst.

### **Key Capabilities**
*   🧠 **Intelligent Routing**: Automatically selects the best model and agent (e.g., uses `llama3.1` for reasoning, `phi3` for speed).
*   🔄 **Dynamic Planner (CoT)**: Uses a "Two Friends" (Generator-Critic) loop to self-correct code before execution, ensuring high reliability.
*   🔒 **Privacy-First**: Built for **Ollama**. Your data never leaves your machine.
*   🔌 **Plugin System**: Easily extensible with new specialized agents.
*   ⚡ **Performance**: Smart caching and optimized data loading for "chat speed" analytics.

---

## ✨ **Agent Swarm**

Nexus employs a team of specialized agents to handle your data:

| Agent | Specialization |
| :--- | :--- |
| **🔍 Query Orchestrator** | The "Manager". Analyzes query complexity and routes to the best agent/model. |
| **📊 Statistical Agent** | Hypothesis testing, correlation, ANOVA, t-tests (Python-backed). |
| **📈 Time Series Agent** | ARIMA forecasting, trend analysis, seasonality detection. |
| **💰 Financial Agent** | Revenue forecasting, ratio analysis (ROI/ROE), liquidity checks. |
| **🤖 ML Insights Agent** | Clustering (K-Means), Classification (Random Forest), Anomaly Detection. |
| **👨‍💻 Data Analyst** | General-purpose Python data manipulation and visualization. |
| **📝 Reporter Agent** | Compiles findings into professional PDF reports. |

---

## 🚀 **Quick Start**

### **Prerequisites**
*   **Python 3.11+**
*   **Node.js 18+**
*   **[Ollama](https://ollama.ai)** (Installed and running)

### **1. AI Model Setup**
Nexus uses local models. Pull the recommended models:
```bash
ollama pull llama3.1:8b   # Primary reasoning model
ollama pull phi3:latest   # Fast/Efficient model
ollama pull tinyllama     # Low-memory fallback
```

### **2. Backend Setup**
```bash
# Clone the repo
git clone https://github.com/KOTAHARSHA25/nexus-llm-analytics.git
cd nexus-llm-analytics

# Setup Python Environment
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

### **3. Frontend Setup**
```bash
cd src/frontend
npm install
```

### **4. Run the Platform**
**Backend (Terminal 1):**
```bash
cd src/backend
python -m uvicorn main:app --reload
```

**Frontend (Terminal 2):**
```bash
cd src/frontend
npm run dev
```

Visit **[http://localhost:3000](http://localhost:3000)** to start analyzing!

---

## 🏗️ **Improved Architecture**

After the **Phase 2 Refactoring & Cleanup**, the system is organized for scalability:

```
nexus-llm-analytics/
├── src/
│   ├── backend/
│   │   ├── api/             # REST API & WebSocket endpoints
│   │   ├── core/            # Engine (Orchestrator, CoT, Cache)
│   │   ├── agents/          # Base Agent definitions
│   │   └── plugins/         # Specialized Agent implementations
│   └── frontend/            # Next.js React Application
├── plugins/                 # (Optional) External plugins directory
├── tests/                   # Centralized Test Suite
│   ├── benchmarks/          # Agent accuracy & speed tests
│   └── verification/        # Feature verification scripts
├── scripts/                 # Utility & Debug scripts
├── docs/                    # Detailed Documentation
└── data/                    # Local storage for uploads/DB
```

---

## 🔧 **Advanced Features**

### **Dynamic Planner & Self-Correction**
For complex tasks (e.g., "Write a script to reverse a string and print it"), Nexus engages the **Self-Correction Engine**.
1.  **Generator**: Drafts the code solution.
2.  **Critic**: Reviews the code for logic errors, bugs, or security risks.
3.  **Correction**: The Generator fixes the issues based on feedback.
*Result: Higher success rate for complex coding tasks without user intervention.*

### **Intelligent Caching**
Queries are hashed and cached intelligently using a multi-level strategy (Memory + Disk).
*   **Benefit**: Asking the same question twice is instant.
*   **Freshness**: use the `force_refresh` option to bypass cache.

---

---

## 📚 **Documentation & Resources**

For detailed guides, please refer to the `docs/` directory.

| Document | Description |
| :--- | :--- |
| **[`docs/COMPLETE_PROJECT_EXPLANATION.md`](docs/COMPLETE_PROJECT_EXPLANATION.md)** | Deep dive into every feature and component. |
| **[`docs/VISUAL_ARCHITECTURE_GUIDE.md`](docs/VISUAL_ARCHITECTURE_GUIDE.md)** | Diagrams showing how data flows through the system. |
| **[`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)** | Explanation of the file organization. |
| **[`docs/TECHNICAL_ARCHITECTURE_OVERVIEW.md`](docs/TECHNICAL_ARCHITECTURE_OVERVIEW.md)** | Technical specs for developers. |

### **🎓 Learning Path**
1.  **Beginner**: Follow the **[Quick Start](#-quick-start)** above to get running.
2.  **User**: Read **`COMPLETE_PROJECT_EXPLANATION.md`** to master the features.
3.  **Developer**: Study **`VISUAL_ARCHITECTURE_GUIDE.md`** and **`PROJECT_STRUCTURE.md`** before contributing.

---

## 🤝 **Contributing**
We welcome contributions! Please see `CONTRIBUTING.md` (coming soon) or check the issues tab.

## 📄 **License**
MIT License. See [LICENSE](LICENSE) for details.

---
<div align="center">
  <b>Local AI • Privacy First • Specialist Agents</b>
</div>
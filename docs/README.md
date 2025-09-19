# Nexus-LLM-Analytics

---

## 📝 Project Summary

Nexus-LLM-Analytics is a local-first, privacy-focused AI assistant for advanced data analysis. It leverages a multi-agent architecture and local LLMs to autonomously analyze both structured and unstructured data, generate code, create visualizations, and compile professional reports—all from natural language queries. No data ever leaves your machine.

---


## ✨ Features & Advanced Capabilities

- Natural language to actionable data workflows
- Handles both structured (tables, spreadsheets) and unstructured (documents, text) data
- Local LLM inference with Ollama for privacy
- Multi-agent CrewAI architecture for modularity and reliability
- Autonomous code review and correction
- Interactive visualizations and professional report generation
- Extensible with new agents and features
- **Automated test suite** (pytest)
- **Data versioning & audit trail** (JSONL log of all analysis steps)
- **Configurable logging levels** (DEBUG/INFO/WARNING/ERROR via env var)
- **Frontend error/status feedback** (real-time user feedback)
- **Pluggable agent system** (add/swap agents via registry)
- **Export/download logs and reports** (API endpoints)
- **User-friendly error messages** (actionable, with suggestions)

---

## 🎯 Goal

Build a privacy-centric AI assistant that transforms natural language queries into end-to-end data analysis workflows, producing results, charts, and reports autonomously and securely on your local machine.

---

## ⚙️ Architectural Overview

Nexus operates as a full-stack, multi-agent system:
- **Frontend:** Next.js (React) for user interaction, file uploads, and displaying results
- **Backend:** FastAPI server orchestrates requests and workflows
- **Multi-Agent Layer:** CrewAI agents collaborate to analyze data, generate code, review outputs, and create reports
- **LLM Inference:** All language model operations run locally via Ollama for maximum privacy

---

## 💻 Component Breakdown

**Frontend (React/Next.js):**
- Upload files, enter queries, view results (charts, tables, reports)

**Backend (FastAPI):**
- Handles API requests, file preprocessing, and workflow orchestration
- Stores structured data in-memory (Pandas DataFrame) and unstructured data in ChromaDB

**Multi-Agent Layer (CrewAI):**
- **Controller Agent:** Orchestrates and decomposes tasks
- **Data Agent:** Analyzes structured data (Python/SQL)
- **RAG Agent:** Retrieves and summarizes unstructured data
- **Review & Correction Agent:** Checks and fixes generated code
- **Visualization Agent:** Creates interactive charts (Plotly)
- **Report Agent:** Compiles outputs into downloadable PDF/Excel reports

**Data & Execution:**
- **Data Storage:** Pandas/Polars DataFrame (structured), ChromaDB (unstructured)
- **Code Execution:** Secure sandbox (RestrictedPython) for all generated code

---

## 🌟 Key Differentiators

- **Autonomous Self-Correction:** Dedicated agent for code review and correction
- **Dual-Model Analysis:** Integrates structured and unstructured data in one workflow
- **Privacy by Design:** 100% local processing, no cloud dependencies
- **Modular & Extensible:** Easily add new agents or features

---

## 📊 Example Workflow: Multi-Modal Query

**Query:** "Compare sales trends with customer sentiment across 2023."

1. **Controller Agent** decomposes the query into sub-tasks
2. **Data Agent** extracts 2023 sales data from structured sources
3. **RAG Agent** analyzes customer reviews and summarizes sentiment
4. **Controller Agent** merges the results
5. **Visualization Agent** creates a comparative line chart
6. **Report Agent** compiles the chart and summary into a downloadable PDF

---


## 🚀 Quick Start

### Step 1: System Requirements Check
```bash
# Check if your system is ready (needs 2GB+ available RAM)
python quick_check.py

# For detailed memory analysis and optimization tips
python -m backend.core.memory_optimizer
```

### Step 2: Install Dependencies
```bash
# Backend dependencies
pip install -r requirements.txt

# Frontend dependencies (in /frontend folder)
cd frontend
npm install
```

### Step 3: Start Ollama & Load Models
```bash
# Start Ollama server
ollama serve

# Load required models (will auto-select based on your RAM)
ollama pull phi3:mini          # 2GB RAM - good performance
ollama pull nomic-embed-text   # For document search
ollama pull llama3.1:8b        # Optional: 6GB+ RAM - best performance
```

### Step 4: Launch the Application
```bash
# Start backend
python -m uvicorn backend.main:app --reload

# Start frontend (new terminal)
cd frontend && npm run dev
```

### Step 5: Use the System
1. Open http://localhost:3000
2. Upload your data files (CSV, PDF, Excel, etc.)
3. Ask questions in natural language
4. Download results, charts, and reports

### 🧠 Intelligent Model Selection

The system automatically chooses the best AI model based on your available RAM:

- **16GB+ available**: Llama 3.1 8B (highest performance)
- **4-16GB available**: Phi-3 Mini (balanced performance)
- **<4GB available**: Optimized lightweight configuration

Configure in `.env`:
```env
AUTO_MODEL_SELECTION=true        # Enable smart selection
PRIMARY_MODEL=ollama/llama3.1:8b # High-RAM systems
# PRIMARY_MODEL=ollama/phi3:mini  # Low-RAM systems
```

---


## 🖥️ Commands

```bash
# Clone the repository
git clone https://github.com/KOTAHARSHA25/nexus-llm-analytics.git
cd nexus-llm-analytics

# Install backend dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..

# Run all backend tests
pytest tests/

# Start the backend (FastAPI)
uvicorn backend.main:app --reload

# Start the frontend (Next.js)
cd frontend
npm run dev
```

## 🧪 Testing & Linting

```bash
# Run all backend tests
pytest tests/

# (Optional) Lint Python code (requires flake8 or ruff)
flake8 backend/ tests/
# or
ruff check backend/ tests/

# (Optional) Lint frontend code
cd frontend
npm run lint
```

## ✅ Feature Checklist

- [x] Automated test suite (pytest)
- [x] Data versioning and audit trail
- [x] Configurable logging levels
- [x] Frontend error/status feedback
- [x] Pluggable agent system
- [x] Export/download logs and reports
- [x] User-friendly error messages

---


## 📫 Contributing & Support

Pull requests and issues are welcome! For questions, open an issue or contact the maintainer.

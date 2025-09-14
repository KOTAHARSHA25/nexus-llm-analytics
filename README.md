# Nexus-LLM-Analytics

---

## 📝 Project Summary

Nexus-LLM-Analytics is a local-first, privacy-focused AI assistant for advanced data analysis. It leverages a multi-agent architecture and local LLMs to autonomously analyze both structured and unstructured data, generate code, create visualizations, and compile professional reports—all from natural language queries. No data ever leaves your machine.

---

## ✨ Features

- Natural language to actionable data workflows
- Handles both structured (tables, spreadsheets) and unstructured (documents, text) data
- Local LLM inference with Ollama for privacy
- Multi-agent CrewAI architecture for modularity and reliability
- Autonomous code review and correction
- Interactive visualizations and professional report generation
- Extensible with new agents and features

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

## 🚀 Get Started

1. Clone the repository
2. Install dependencies (see requirements.txt and package.json)
3. Start the backend (FastAPI) and frontend (Next.js)
4. Upload your data and ask questions in natural language

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

# Start the backend (FastAPI)
uvicorn backend.main:app --reload

# Start the frontend (Next.js)
cd frontend
npm run dev
```

---

## 📫 Contributing & Support

Pull requests and issues are welcome! For questions, open an issue or contact the maintainer.

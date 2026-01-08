# 📂 Project Structure

**Version 2.1** - Updated for "Phase 2 Refactoring & Cleanup"

This document outlines the current file organization of the Nexus LLM Analytics platform.

---

## 🏗️ high-Level Directory Map

```
nexus-llm-analytics/
├── 📁 src/                          # Source Code
│   ├── 📁 backend/                  # FastAPI Backend
│   └── 📁 frontend/                 # Next.js Frontend
├── 📁 plugins/                      # 🔌 Specialized Agent Plugins
├── 📁 tests/                        # 🧪 Centralized Test Suite
├── 📁 scripts/                      # 🛠️ Utility & Debug Scripts
├── 📁 docs/                         # 📚 Documentation
├── 📁 data/                         # 💾 Local Data Storage
├── 📁 logs/                         # 📝 Application Logs
├── 📁 chroma_db/                    # 🧠 Vector Database Storage
└── 📄 requirements.txt              # Project Dependencies
```

---

## 🔧 Detailed Breakdown

### 1. Backend (`src/backend/`)
The brain of the operation.

```
src/backend/
├── main.py                          # Application Entry Point
├── 📁 api/                          # REST API Endpoints
│   ├── analyze.py                   # Main Analysis Endpoint
│   ├── upload.py                    # File Upload Handling
│   └── ...
├── 📁 core/                         # Core Engine Components
│   ├── 📁 engine/                   # Query Execution Engine
│   │   ├── query_orchestrator.py    # Intelligent Routing
│   │   └── self_correction_engine.py# CoT / Dynamic Planner
│   ├── 📁 security/                 # Security & Sandbox
│   │   ├── sandbox.py               # RestrictedPython Environment
│   │   └── security_guards.py       # Input Validation
│   ├── 📁 system/                   # System-Level Utilities
│   │   ├── llm_client.py            # Ollama Client
│   │   └── model_selector.py        # Resource-Aware Model Selection
│   └── config.py                    # Global Configuration
├── 📁 agents/                       # Base Agent Definitions
│   └── agent_interface.py           # Abstract Base Class for Agents
└── 📁 services/                     # Business Logic Layers
    └── analysis_service.py          # Coordinate Analysis Flow
```

### 2. Plugins (`plugins/`)
Specialized agents that extend the core capabilities.

```
plugins/
├── statistical_agent.py             # Advanced Statistics (ANOVA, t-tests)
├── time_series_agent.py             # ARIMA Forecasting
├── financial_agent.py               # Financial Ratios & Metrics
├── ml_insights_agent.py             # Clustering & Classification
└── sql_agent.py                     # Database Interaction
```

### 3. Frontend (`src/frontend/`)
The user interface.

```
src/frontend/
├── 📁 app/                          # Next.js App Router
├── 📁 components/                   # React Components
│   ├── 📁 ui/                       # Reusable UI Elements (Shadcn/Radix)
│   ├── analysis-display.tsx         # Results Visualization
│   └── query-input.tsx              # Natural Language Input
└── 📁 lib/                          # Frontend Utilities
```

### 4. Tests (`tests/`)
Comprehensive testing suite.

```
tests/
├── 📁 benchmarks/                   # Accuracy & Speed Benchmarks
├── 📁 verification/                 # Feature Verification Scripts
├── 📁 unit/                         # Unit Tests
└── 📁 data/                         # Test Datasets
```

### 5. Scripts (`scripts/`)
Helper tools for developers.

```
scripts/
├── 📁 debug/                        # Debugging Tools
│   └── debug_cot_isolated.py        # CoT Logic Tester
└── 📁 utils/                        # Maintenance Utilities
    └── verify_requirements.py       # Dependency Checker
```

### 6. Data (`data/`)
Local storage for user data (Privacy First!).

```
data/
├── 📁 uploads/                      # Raw Uploaded Files
└── 📁 exports/                      # Generated Reports (PDF/Excel)
```

---

## 🔄 Key Changes in v2.1
*   **Moved Tests**: All backend tests moved from `src/backend/tests` to root `tests/`.
*   **Cleaned Root**: Removed clutter, moved scripts to `scripts/` folder.
*   **Core Engine**: `src/backend/core` now contains the `engine` sub-directory for the Dynamic Planner.
*   **Plugins**: Specialized agents are now clearly separated in `plugins/`.

---
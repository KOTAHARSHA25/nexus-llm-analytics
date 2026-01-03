# 📁 Nexus LLM Analytics - Complete File Manifest (VERIFIED)

> **Generated:** December 22, 2025  
> **Last Updated:** Phase 2 Audit Complete  
> **Analysis Method:** Deep code inspection + import tracing via grep analysis  
> **Purpose:** Accurately identify which files are ACTUALLY used in production vs. tests vs. dead code

---

## 🆕 VERSION 1.2 UPDATE (Phase 2 System Audit)

### Integration Audit Completed

A comprehensive frontend↔backend integration audit was performed. See related documents:
- **AUDIT_REPORT.md** - Detailed findings, broken integrations fixed, orphaned endpoints cataloged
- **INTEGRATION_MAP.md** - Complete mapping of user features to backend services

### Key Findings

| Metric | Count |
|--------|-------|
| Backend Endpoints (Total) | 53 |
| Frontend API Calls (Verified Working) | 21 |
| Orphaned Backend Endpoints | 29 (documented) |
| Broken API Calls Fixed | 4 |

### Frontend Files Modified

| File | Change |
|------|--------|
| `src/frontend/lib/config.ts` | Cleaned unused endpoint declarations, organized by category |
| `src/frontend/app/page.tsx` | Fixed `handleCancelAnalysis` to use `getEndpoint()` |
| `src/frontend/components/file-upload.tsx` | Fixed `handleDownloadFile` to use `getEndpoint()` |
| `src/frontend/components/analytics-sidebar.tsx` | Fixed report download to use `getEndpoint()` |

### Backend Endpoints Marked for Removal

| Endpoint | Reason |
|----------|--------|
| `GET /api/models/health` | Duplicates `/api/health/` |
| `GET /api/models/current` | Duplicates `/api/models/status` |
| `POST /api/models/configure` | Complex unused configuration |
| `POST /api/visualize/execute` | Merged into `/api/visualize/` |

---

## 🆕 VERSION 1.1 UPDATE (December 27, 2025)

### Files Archived in `archive/removed_v1.1/`

| File | Previous Location | Reason | Size |
|------|------------------|--------|------|
| `intelligent_query_engine.py` | `src/backend/core/` | Over-engineered, never integrated into main flow | 43KB |
| `optimized_llm_client.py` | `src/backend/core/` | Duplicate of `llm_client.py`, never imported | 24KB |
| `websocket_manager.py` | `src/backend/core/` | Disabled in config, incomplete implementation | 11KB |

### Scope Changes
- **Authentication files** - Not to be created (out of scope)
- **WebSocket code** - Commented out in `main.py`, archived

---

## ⚠️ VERIFICATION METHODOLOGY

This manifest is based on **actual import statements** found in the codebase, not file names:
1. `grep_search` for all `import` and `from ... import` statements
2. Traced dependency chains from entry points (`main.py`, API routers)
3. Flagged files not reachable from any entry point as TEST-ONLY or DEAD

**NOTE:** As of December 2025, the system uses a **Custom Plugin Architecture** (not CrewAI).
The main orchestration is done via `analysis_service.py` and `plugin_system.py`.
**CrewAI Status:** ✅ Fully removed from production code (only legacy files remain in archive/)

**Legend:**
- 🟢 **PRODUCTION** - Imported and used in main application code (main.py, API endpoints, plugins, services)
- 🟡 **TEST-ONLY** - Only imported in test files, not in production runtime
- 🔴 **DEAD CODE** - Not imported anywhere, or only in archived/deprecated code
- 🟠 **LEGACY** - In archive folder, previously deprecated (includes CrewAI-related files)
- 📝 **DOCUMENTATION** - Documentation files
- ⚪ **DATA/CONFIG** - Data or configuration files
- 🧪 **TEST** - Test files

---

## 📊 Summary Statistics (Verified)

| Status | Count | Action |
|--------|-------|--------|
| 🟢 PRODUCTION (core) | ~25 files | **KEEP** - Essential for app to run |
| 🟢 PRODUCTION (plugins) | 10 agent files | **KEEP** - Plugin system relies on these |
| 🟢 PRODUCTION (api) | 8 files | **KEEP** - All API endpoints |
| 🟢 PRODUCTION (frontend) | ~30 files | **KEEP** - Next.js UI |
| 🟡 TEST-ONLY | ~8 core files | **REVIEW** - Could be useful but not in production path |
| 🔴 DEAD CODE | ~5 files | **SAFE TO REMOVE** |
| 🟠 LEGACY (archive) | ~100+ files | **SAFE TO REMOVE** |
| 📝 DOCUMENTATION | ~25 files | **KEEP** relevant ones |
| ⚪ DATA | ~50 files | **KEEP** essentials |
| 🧪 TEST | ~50+ files | **KEEP** for testing |

---

## 🏗️ ROOT DIRECTORY FILES

### Configuration Files

| File | Status | Purpose | Dependencies | Recommendation |
|------|--------|---------|--------------|----------------|
| `.env` | 🟢 CRITICAL | Environment variables for models, CORS, database paths | All backend services | **KEEP** - Essential configuration |
| `pyproject.toml` | 🟢 CRITICAL | Python project metadata, dependencies, tool configs | pip, setuptools | **KEEP** - Project definition |
| `requirements.txt` | 🟢 CRITICAL | Python package dependencies | pip install | **KEEP** - Dependency management |
| `start_backend.bat` | 🟡 IMPORTANT | Windows batch script to start FastAPI backend | Python, uvicorn | **KEEP** - Quick start utility |

### Documentation Files

| File | Status | Purpose | Recommendation |
|------|--------|---------|----------------|
| `README.md` | 📝 DOC | Main project documentation with features, setup, usage | **KEEP** - Primary docs |
| `LICENSE` | 📝 DOC | MIT License for the project | **KEEP** - Required for open source |
| `PROJECT_ARCHITECTURE.md` | 📝 DOC | System architecture diagrams and explanations | **KEEP** - Architecture reference |
| `DATA_FLOW_GUIDE.md` | 📝 DOC | Detailed data flow through all components | **KEEP** - Technical reference |
| `PAPER_ALIGNMENT_ROADMAP.md` | 📝 DOC | Research paper alignment tracking | **KEEP** - Academic alignment |

### Test/Utility Scripts

| File | Status | Purpose | Recommendation |
|------|--------|---------|----------------|
| `verify_strict_analysis.py` | 🧪 TEST | Tests analysis endpoint with sample query | **OPTIONAL** - Development testing |

---

## 📂 src/backend/ - CORE BACKEND

### Root Backend Files

| File | Status | Lines | Purpose | Key Functions/Classes | Dependencies |
|------|--------|-------|---------|----------------------|--------------|
| `main.py` | 🟢 CRITICAL | 217 | FastAPI application entry point, middleware setup, route mounting | `app`, `lifespan()`, `test_model_on_startup()` | FastAPI, all API routers |
| `__init__.py` | 🟢 CRITICAL | 0 | Package marker | - | - |
| `test_analysis_service.py` | 🧪 TEST | ~45 | Tests AnalysisService routing | `test_service()` | services.analysis_service |
| `test_plugin_loading.py` | 🧪 TEST | ~35 | Tests plugin agent discovery | - | core.plugin_system |

---

### 📂 src/backend/api/ - API ENDPOINTS (ALL 🟢 PRODUCTION)

All API files are **mounted in `main.py`** and actively used:

| File | Status | Lines | Router Prefix | Key Endpoints | Imported By |
|------|--------|-------|---------------|---------------|-------------|
| `analyze.py` | 🟢 PRODUCTION | 260 | `/api/analyze` | `POST /` - analyze_query | `main.py` |
| `upload.py` | 🟢 PRODUCTION | 1091 | `/api/upload` | `POST /`, `POST /raw-text` | `main.py` |
| `health.py` | 🟢 PRODUCTION | 191 | `/api/health` | `GET /status` | `main.py` |
| `visualize.py` | 🟢 PRODUCTION | 861 | `/api/visualize` | `POST /` - generate charts | `main.py` |
| `report.py` | 🟢 PRODUCTION | 246 | `/api/report` | `POST /`, `GET /download-log` | `main.py` |
| `history.py` | 🟢 PRODUCTION | 267 | `/api/history` | `GET /`, `POST /`, `DELETE /` | `main.py` |
| `models.py` | 🟢 PRODUCTION | 373 | `/api/models` | `GET /available`, `POST /configure` | `main.py` |
| `viz_enhance.py` | 🟢 PRODUCTION | 666 | `/api/viz` | `POST /edit`, `POST /repair` | `main.py` |

---

### 📂 src/backend/core/ - VERIFIED USAGE ANALYSIS

#### 🟢 PRODUCTION CODE (Actually Used at Runtime)

These files are imported by `main.py`, API endpoints, plugins, or services:

| File | Lines | Imported By | Usage Evidence |
|------|-------|-------------|----------------|
| `config.py` | 329 | `main.py`, most modules | `from backend.core.config import get_settings` everywhere |
| `llm_client.py` | 202 | Multiple plugins, services | `from backend.core.llm_client import LLMClient` |
| `plugin_system.py` | 357 | `services/analysis_service.py` | `get_agent_registry()` - CRITICAL |
| `sandbox.py` | 483 | `api/visualize.py`, plugins | `EnhancedSandbox` - secure execution |
| `security_guards.py` | ~200 | `sandbox.py` | `SecurityGuards, ResourceManager, CodeValidator` |
| `self_correction_engine.py` | 448 | plugins, viz endpoints | Chain-of-Thought loop |
| `cot_parser.py` | ~200 | `self_correction_engine.py` | CoT parsing |
| `document_indexer.py` | 274 | `plugins/rag_agent.py` | RAG indexing |
| `chromadb_client.py` | ~80 | `document_indexer.py`, `rag_agent.py` | Vector DB ops |
| `model_selector.py` | 340 | `main.py`, `api/models.py` | `select_optimal_models()` at startup |
| `dynamic_planner.py` | 163 | `plugins/data_analyst_agent.py` | `get_dynamic_planner()` |
| `analysis_manager.py` | 107 | `api/analyze.py` | `analysis_manager.start_analysis()` |
| `rate_limiter.py` | ~150 | `main.py` | `RateLimitMiddleware` |
| `error_handling.py` | ~200 | `main.py` | `error_handler` |
| `user_preferences.py` | ~100 | `model_selector.py`, `api/models.py` | User prefs storage |
| `advanced_cache.py` | 354 | `api/health.py`, `optimizers.py` | `get_cache_status()`, `clear_all_caches()` |
| `optimizers.py` | 759 | `main.py` | `optimize_startup()` in lifespan handler |
| `websocket_manager.py` | 345 | `main.py` | `websocket_endpoint` |
| `query_parser.py` | 383 | `agents/model_initializer.py` | Query intent classification |
| `query_complexity_analyzer.py` | ~300 | `agents/model_initializer.py` | `QueryComplexityAnalyzer` |
| `circuit_breaker.py` | ~150 | `llm_client.py`, `api/health.py` | Fault tolerance |
| `enhanced_logging.py` | ~200 | `config.py` | Logging configuration |
| `enhanced_reports.py` | ~300 | `api/report.py` | Report generation |

#### 🟡 TEST-ONLY FILES (Not in Production Path)

These are only imported by test files or other test-only modules:

| File | Lines | Only Imported By | Verdict |
|------|-------|------------------|---------|
| `optimized_data_structures.py` | 644 | `tests/performance/test_benchmarks.py`, `enhanced_cache_integration.py` | TEST-ONLY |
| `optimized_llm_client.py` | 636 | `tests/performance/test_benchmarks.py` | TEST-ONLY benchmark |
| `optimized_file_io.py` | 735 | `tests/performance/test_benchmarks.py` | TEST-ONLY benchmark |
| `enhanced_cache_integration.py` | ~400 | `intelligent_query_engine.py`, tests | TEST-ONLY |
| `intelligent_query_engine.py` | ~500 | `tests/conftest.py`, tests | TEST-ONLY |
| `model_detector.py` | ~200 | `tests/phase7_production/unit/test_model_detector.py` | TEST-ONLY |

#### 🔴 DEAD CODE (Not Used Anywhere)

These files are NOT imported by production code OR tests (only by archived/deprecated code):

| File | Lines | Evidence | Verdict |
|------|-------|----------|---------|
| `utils.py` | 60 | Only imported by OLD `nexus-llm-analytics-distribution_20251018_183430 (1)/` and `src/backend/archive/` | **DEAD** |
| `optimized_tools.py` | ~100 | Only in OLD distribution folder | **DEAD** |
| `crewai_base.py` | ~50 | Only `scripts/test_rag.py` and OLD distribution | **DEAD** |
| `memory_optimizer.py` | ~100 | Only `scripts/startup_check.py` (unused script) | **DEAD** |
| `crewai_import_manager.py` | ~50 | Imported by `optimizers.py` but only for legacy CrewAI preloading | **LEGACY** (low impact) |

---

### 📂 src/backend/plugins/ - PLUGIN AGENTS (ALL 🟢 PRODUCTION)

All plugins are **discovered at runtime** by `plugin_system.py`:

| File | Status | Lines | Agent Class | Capabilities |
|------|--------|-------|-------------|--------------|
| `data_analyst_agent.py` | 🟢 PRODUCTION | 246 | `DataAnalystAgent` | CSV, JSON, Excel analysis with CoT |
| `rag_agent.py` | 🟢 PRODUCTION | 210 | `RAGAgent` | PDF, DOCX, TXT via vector search |
| `visualizer_agent.py` | 🟢 PRODUCTION | 107 | `VisualizerAgent` | Plotly chart generation |
| `reporter_agent.py` | 🟢 PRODUCTION | 103 | `ReporterAgent` | Professional reports |
| `reviewer_agent.py` | 🟢 PRODUCTION | ~100 | `ReviewerAgent` | Analysis review |
| `statistical_agent.py` | 🟢 PRODUCTION | 1347 | `StatisticalAgent` | Hypothesis testing, regression |
| `financial_agent.py` | 🟢 PRODUCTION | 725 | `FinancialAgent` | ROI, ratios, forecasting |
| `ml_insights_agent.py` | � PRODUCTION | 813 | `MLInsightsAgent` | Clustering, anomaly detection |
| `time_series_agent.py` | 🟢 PRODUCTION | 1252 | `TimeSeriesAgent` | ARIMA, seasonality |
| `sql_agent.py` | 🟢 PRODUCTION | 576 | `SQLAgent` | Query generation |
| `agents_config.json` | ⚪ CONFIG | ~15 | - | Agent configuration |

---

### 📂 src/backend/services/ (🟢 PRODUCTION)

| File | Status | Lines | Imported By | Purpose |
|------|--------|-------|-------------|---------|
| `analysis_service.py` | 🟢 PRODUCTION | ~100 | `api/analyze.py` | High-level analysis orchestrator |

---

### 📂 src/backend/agents/ (🟢 PRODUCTION)

| File | Status | Lines | Imported By | Purpose |
|------|--------|-------|-------------|---------|
| `model_initializer.py` | 🟢 PRODUCTION | ~200 | ALL plugin agents | Lazy LLM initialization |
| `__init__.py` | 🟢 PRODUCTION | 0 | - | Package marker |

---

### 📂 src/backend/utils/ (🟢 PRODUCTION)

| File | Status | Lines | Purpose | Key Functions |
|------|--------|-------|---------|---------------|
| `data_utils.py` | 🟢 CRITICAL | 473 | DataFrame operations, path resolution | `DataPathResolver`, `read_dataframe()`, `create_data_summary()` |
| `data_optimizer.py` | 🟢 CRITICAL | 797 | Data optimization for LLM consumption | `DataOptimizer.optimize_for_llm()` |
| `__init__.py` | 🟢 CRITICAL | 0 | Package marker | - |

---

### 📂 src/backend/visualization/

| File | Status | Lines | Purpose | Key Functions |
|------|--------|-------|---------|---------------|
| `dynamic_charts.py` | 🟢 CRITICAL | 320 | Template-based chart generation | `ChartTypeAnalyzer`, `DynamicChartGenerator` |
| `scaffold.py` | 🟢 CRITICAL | 268 | LIDA-style chart scaffolding | `ChartScaffold.get_template()` |
| `__init__.py` | 🟢 CRITICAL | 0 | Package marker | - |

---

### 📂 src/backend/prompts/

| File | Status | Lines | Purpose |
|------|--------|-------|---------|
| `cot_generator_prompt.txt` | 🟢 CRITICAL | ~30 | Chain-of-Thought generator system prompt |
| `cot_critic_prompt.txt` | 🟢 CRITICAL | ~35 | Chain-of-Thought critic system prompt |

---

### 📂 src/backend/archive/ - 🟠 LEGACY CODE (SAFE TO REMOVE)

| File | Status | Lines | Purpose | Replacement |
|------|--------|-------|---------|-------------|
| `legacy_controller_agent.py` | 🟠 DEPRECATED | ~80 | Old CrewAI controller | `services.analysis_service` |
| `crew_manager.py` | 🟠 DEPRECATED | 504 | Old CrewAI manager (refactored) | `services.analysis_service` |
| `crew_singleton.py` | 🟠 DEPRECATED | ~50 | Old singleton pattern | `get_analysis_service()` |
| `agent_factory.py` | 🟠 DEPRECATED | ~200 | Old agent creation | `plugin_system` |
| `analysis_executor.py` | 🟠 DEPRECATED | ~300 | Old analysis execution | `data_analyst_agent` |
| `legacy_data_agent.py` | 🟠 DEPRECATED | ~200 | Old data agent | `data_analyst_agent.py` |
| `legacy_rag_agent.py` | 🟠 DEPRECATED | ~150 | Old RAG agent | `rag_agent.py` |
| `legacy_rag_handler.py` | 🟠 DEPRECATED | ~150 | Old RAG handler | `rag_agent.py` |
| `legacy_report_agent.py` | 🟠 DEPRECATED | ~100 | Old report agent | `reporter_agent.py` |
| `legacy_review_agent.py` | 🟠 DEPRECATED | ~100 | Old review agent | `reviewer_agent.py` |
| `legacy_visualization_agent.py` | 🟠 DEPRECATED | ~100 | Old viz agent | `visualizer_agent.py` |
| `legacy_specialized_agents.py` | 🟠 DEPRECATED | ~200 | Old specialized agents | Individual plugin agents |
| `legacy_intelligent_router.py` | 🟠 DEPRECATED | ~150 | Old query router | `plugin_system.route_query()` |
| `query_complexity_analyzer_v1.py` | 🟠 DEPRECATED | ~100 | Old complexity analyzer | `query_complexity_analyzer.py` |

**Recommendation:** Delete entire `src/backend/archive/` directory to clean up codebase.

---

### 📂 src/backend/tests/ - Backend Tests

| File | Status | Purpose |
|------|--------|---------|
| `test_analysis_flow.py` | 🧪 TEST | Tests analysis workflow |
| `test_api_sanity.py` | 🧪 TEST | API endpoint sanity tests |
| `test_domain_agnostic.py` | 🧪 TEST | Domain-agnostic analysis tests |
| `test_history.py` | 🧪 TEST | History API tests |
| `test_plugin_integration.py` | 🧪 TEST | Plugin loading tests |
| `test_upload_flow.py` | 🧪 TEST | File upload tests |
| `test_visualization.py` | 🧪 TEST | Visualization tests |
| `conftest.py` | 🧪 TEST | Pytest fixtures |
| `run_tests.py` | 🧪 TEST | Test runner script |

---

## 📂 src/frontend/ - NEXT.JS FRONTEND

### Configuration Files

| File | Status | Purpose |
|------|--------|---------|
| `package.json` | 🟢 CRITICAL | Node.js dependencies and scripts |
| `tsconfig.json` | 🟢 CRITICAL | TypeScript configuration |
| `next.config.js` | 🟢 CRITICAL | Next.js configuration |
| `tailwind.config.js` | 🟢 CRITICAL | Tailwind CSS configuration |
| `postcss.config.js` | 🟢 CRITICAL | PostCSS configuration |
| `components.json` | 🔵 UTILITY | shadcn/ui configuration |
| `.env.example` | 📝 DOC | Environment template |
| `next-env.d.ts` | 🔵 UTILITY | Next.js type declarations |

### 📂 src/frontend/app/

| File | Status | Lines | Purpose |
|------|--------|-------|---------|
| `page.tsx` | 🟢 CRITICAL | 608 | Main dashboard page with all UI logic |
| `layout.tsx` | 🟢 CRITICAL | ~30 | Root layout with fonts and metadata |
| `globals.css` | 🟢 CRITICAL | ~100 | Global CSS styles with Tailwind |

### 📂 src/frontend/components/

| File | Status | Purpose |
|------|--------|---------|
| `header.tsx` | 🟢 CRITICAL | Application header component |
| `file-upload.tsx` | 🟢 CRITICAL | File upload with drag-drop |
| `query-input.tsx` | 🟢 CRITICAL | Query input field |
| `results-display.tsx` | 🟢 CRITICAL | Analysis results display |
| `analytics-sidebar.tsx` | 🟢 CRITICAL | Sidebar with history/plugins |
| `model-settings.tsx` | 🟡 IMPORTANT | Model configuration UI |
| `setup-wizard.tsx` | 🟡 IMPORTANT | First-time setup wizard |
| `file-preview.tsx` | 🟡 IMPORTANT | File preview modal |
| `chart-viewer.tsx` | 🟡 IMPORTANT | Plotly chart viewer |
| `error-boundary.tsx` | 🔵 UTILITY | Error boundary wrapper |
| `backend-url-settings.tsx` | 🔵 UTILITY | Backend URL configuration |
| `routing-stats.tsx` | 🔵 UTILITY | Query routing statistics |
| `sidebar.tsx` | 🔵 UTILITY | Generic sidebar |
| `OptimizedComponents.tsx` | 🔵 UTILITY | Performance-optimized components |

### 📂 src/frontend/components/ui/ - shadcn/ui Components

| File | Status | Purpose |
|------|--------|---------|
| `button.tsx` | 🟢 CRITICAL | Button component |
| `card.tsx` | 🟢 CRITICAL | Card component |
| `input.tsx` | 🟢 CRITICAL | Input component |
| `dialog.tsx` | 🟢 CRITICAL | Modal dialog |
| `tabs.tsx` | 🟢 CRITICAL | Tabs component |
| `select.tsx` | 🟢 CRITICAL | Select dropdown |
| `textarea.tsx` | 🟢 CRITICAL | Textarea component |
| `progress.tsx` | 🟡 IMPORTANT | Progress bar |
| `table.tsx` | 🟡 IMPORTANT | Table component |
| `badge.tsx` | 🟡 IMPORTANT | Badge component |
| `alert.tsx` | 🔵 UTILITY | Alert component |
| `toast.tsx` | 🔵 UTILITY | Toast notifications |
| `dropdown-menu.tsx` | 🔵 UTILITY | Dropdown menu |
| `label.tsx` | 🔵 UTILITY | Form label |
| `scroll-area.tsx` | 🔵 UTILITY | Scrollable area |
| `separator.tsx` | 🔵 UTILITY | Visual separator |
| `switch.tsx` | 🔵 UTILITY | Toggle switch |

### 📂 src/frontend/hooks/

| File | Status | Purpose |
|------|--------|---------|
| `useDashboardState.ts` | 🟢 CRITICAL | Main dashboard state management |
| `useWebSocket.ts` | 🔵 UTILITY | WebSocket connection hook |
| `use-toast.ts` | 🔵 UTILITY | Toast notification hook |

### 📂 src/frontend/lib/

| File | Status | Purpose |
|------|--------|---------|
| `config.ts` | 🟢 CRITICAL | API endpoint configuration |
| `backend-config.ts` | 🟢 CRITICAL | Backend URL configuration |
| `utils.ts` | 🟢 CRITICAL | Utility functions (cn for classNames) |

---

## 📂 config/ - CONFIGURATION

| File | Status | Purpose |
|------|--------|---------|
| `cot_review_config.json` | 🟢 CRITICAL | Chain-of-Thought configuration |
| `user_preferences.json` | 🟡 IMPORTANT | User model preferences (auto-generated) |
| `.env.example` | 📝 DOC | Environment variable template |

---

## 📂 scripts/ - UTILITY SCRIPTS

| File | Status | Lines | Purpose | Recommendation |
|------|--------|-------|---------|----------------|
| `launch.py` | 🟡 IMPORTANT | 468 | Full application launcher | **KEEP** - Useful for startup |
| `health_check.py` | 🟡 IMPORTANT | 315 | System requirements checker | **KEEP** - Diagnostics |
| `startup_check.py` | 🔵 UTILITY | ~100 | Startup validation | **KEEP** - Diagnostics |
| `nexus_startup.py` | 🔵 UTILITY | ~100 | Alternative startup script | **REVIEW** - May be duplicate |
| `quick_check.py` | 🔵 UTILITY | ~50 | Quick health check | **REVIEW** - May be duplicate |
| `test_rag.py` | 🧪 TEST | ~100 | RAG functionality test | **OPTIONAL** |
| `verify_improvements.py` | 🧪 TEST | ~100 | Verify improvements | **OPTIONAL** |
| `create_distribution_zip.py` | 🔵 UTILITY | ~200 | Create distribution package | **KEEP** - Distribution |

---

## 📂 tests/ - TEST SUITE

### Root Test Files

| File | Status | Purpose |
|------|--------|---------|
| `conftest.py` | 🧪 TEST | Pytest configuration and fixtures |
| `test_runner.py` | 🧪 TEST | Test execution script |
| `test_sandbox_security.py` | 🧪 TEST | Sandbox security tests |
| `test_phase7_routing.py` | 🧪 TEST | Routing accuracy tests |
| `api_integration_tests.ps1` | 🧪 TEST | PowerShell integration tests |

### Test Documentation

| File | Status | Purpose |
|------|--------|---------|
| `COMPREHENSIVE_TEST_MASTER_PLAN.md` | 📝 DOC | Testing strategy |
| `COMPREHENSIVE_TEST_RESULTS.md` | 📝 DOC | Test results |
| `COMPREHENSIVE_TEST_RESULTS_PHASE2.md` | 📝 DOC | Phase 2 test results |
| `COMPREHENSIVE_TESTING_PROGRESS.md` | 📝 DOC | Testing progress |
| `FIXES_NEEDED.md` | 📝 DOC | Bug tracking |
| `TEST_DATA_LOG.md` | 📝 DOC | Test data documentation |

### Test Subdirectories

| Directory | Status | Purpose | Recommendation |
|-----------|--------|---------|----------------|
| `tests/backend/` | 🧪 TEST | Backend unit/integration tests | **KEEP** |
| `tests/comprehensive/` | 🧪 TEST | Full system tests | **KEEP** |
| `tests/csv/` | 🧪 TEST | CSV-specific tests | **KEEP** |
| `tests/document/` | 🧪 TEST | Document analysis tests | **KEEP** |
| `tests/performance/` | 🧪 TEST | Performance benchmarks | **KEEP** |
| `tests/plugins/` | 🧪 TEST | Plugin agent tests | **KEEP** |
| `tests/security/` | 🧪 TEST | Security tests | **KEEP** |
| `tests/unit/` | 🧪 TEST | Unit tests | **KEEP** |
| `tests/visualization/` | 🧪 TEST | Visualization tests | **KEEP** |
| `tests/upload_validation/` | 🧪 TEST | Upload validation tests | **KEEP** |
| `tests/phase7_production/` | 🧪 TEST | Production readiness tests | **KEEP** |

---

## 📂 docs/ - DOCUMENTATION

| File | Status | Purpose | Recommendation |
|------|--------|---------|----------------|
| `README.md` | 📝 DOC | Documentation index | **KEEP** |
| `QUICK_START.md` | 📝 DOC | Quick start guide | **KEEP** |
| `PRODUCTION_README.md` | 📝 DOC | Production deployment guide | **KEEP** |
| `TECH_STACK.md` | 📝 DOC | Technology stack overview | **KEEP** |
| `TECHNICAL_ARCHITECTURE_OVERVIEW.md` | 📝 DOC | Architecture details | **KEEP** |
| `PROJECT_STRUCTURE.md` | 📝 DOC | File structure explanation | **KEEP** |
| `BACKEND_CONFIG_QUICKREF.md` | 📝 DOC | Backend configuration reference | **KEEP** |
| `FRONTEND_BACKEND_SYNC.md` | 📝 DOC | API synchronization guide | **KEEP** |
| `MODEL_COMMUNICATION.md` | 📝 DOC | LLM communication patterns | **KEEP** |
| `INTELLIGENT_ROUTING_USER_GUIDE.md` | 📝 DOC | Query routing guide | **KEEP** |
| `SMART_MODEL_SELECTION.md` | 📝 DOC | Model selection algorithm | **KEEP** |
| `TWO_FRIENDS_MODEL_GUIDE.md` | 📝 DOC | Generator-Critic pattern guide | **KEEP** |
| `VISUAL_ARCHITECTURE_GUIDE.md` | 📝 DOC | Visual diagrams | **KEEP** |
| `DEVELOPMENT_NOTES.md` | 📝 DOC | Development notes | **KEEP** |
| `SECURITY_CHECKLIST.md` | 📝 DOC | Security checklist | **KEEP** |
| `COMPLETE_PROJECT_EXPLANATION.md` | 📝 DOC | Full project explanation | **KEEP** |
| `PHASE4_VISUALIZATION_COMPLETE.md` | 📝 DOC | Phase 4 completion notes | **REVIEW** |
| `PHASE7_TEST_PROGRESS_REPORT.md` | 📝 DOC | Phase 7 progress | **REVIEW** |
| `ROUTING_IMPROVEMENT_ACTION_PLAN.md` | 📝 DOC | Improvement plan | **REVIEW** |
| `STRESS_TEST_ANALYSIS_REPORT.md` | 📝 DOC | Stress test results | **REVIEW** |

---

## 📂 data/ - DATA FILES

### 📂 data/samples/ - Sample Data

| File | Status | Purpose | Recommendation |
|------|--------|---------|----------------|
| `sales_data.csv` | ⚪ DATA | Sample sales data | **KEEP** - Demo |
| `StressLevelDataset.csv` | ⚪ DATA | Health/stress data | **KEEP** - Demo |
| `test_employee_data.csv` | ⚪ DATA | Employee test data | **OPTIONAL** |
| `test_inventory.csv` | ⚪ DATA | Inventory test data | **OPTIONAL** |
| `test_iot_sensor.csv` | ⚪ DATA | IoT sensor test data | **OPTIONAL** |
| `test_sales_monthly.csv` | ⚪ DATA | Monthly sales test data | **OPTIONAL** |
| `test_student_grades.csv` | ⚪ DATA | Student grades test data | **OPTIONAL** |
| `test_university_grades.csv` | ⚪ DATA | University grades test data | **OPTIONAL** |
| `1.json`, `analyze.json`, `simple.json` | ⚪ DATA | JSON test files | **OPTIONAL** |
| `complex_nested.json` | ⚪ DATA | Nested JSON test | **OPTIONAL** |
| `financial_quarterly.json` | ⚪ DATA | Financial test data | **OPTIONAL** |
| `large_transactions.json` | ⚪ DATA | Large dataset test | **OPTIONAL** |
| `malformed.json` | ⚪ DATA | Error handling test | **KEEP** - Testing |
| `sales_timeseries.json` | ⚪ DATA | Time series test | **OPTIONAL** |

### 📂 data/samples/csv/ - CSV Test Files

| File | Status | Purpose |
|------|--------|---------|
| `customer_data.csv` | ⚪ DATA | Customer data |
| `customers.csv` | ⚪ DATA | Customer list |
| `orders.csv` | ⚪ DATA | Order data |
| `sales_simple.csv` | ⚪ DATA | Simple sales |
| `special_types.csv` | ⚪ DATA | Special data types |
| `transactions_large.csv` | ⚪ DATA | Large transaction set |

### 📂 data/samples/edge_cases/ - Edge Case Tests

| File | Status | Purpose |
|------|--------|---------|
| `boolean_fields.json` | ⚪ DATA | Boolean handling |
| `date_formats.json` | ⚪ DATA | Date parsing |
| `deep_nested.json` | ⚪ DATA | Deep nesting |
| `empty_array.json` | ⚪ DATA | Empty arrays |
| `empty_object.json` | ⚪ DATA | Empty objects |
| `mixed_types.json` | ⚪ DATA | Mixed types |
| `null_values.json` | ⚪ DATA | Null handling |
| `unicode_data.json` | ⚪ DATA | Unicode support |

### 📂 data/uploads/ - User Uploads

Contains user-uploaded files. **Keep empty for distribution.**

### Other Data Directories

| Directory | Status | Purpose |
|-----------|--------|---------|
| `data/audit/` | ⚪ DATA | Audit logs |
| `data/history/` | ⚪ DATA | Query history |
| `data/reports/` | ⚪ DATA | Generated reports |

---

## 📂 archive/ - 🟠 DEPRECATED/ARCHIVED (ROOT LEVEL)

### 📂 archive/dev_utilities/

| File | Status | Purpose | Recommendation |
|------|--------|---------|----------------|
| `analyze_failures.py` | 🟠 DEPRECATED | Debug script | **REMOVE** |
| `check_large_csv.py` | 🟠 DEPRECATED | Debug script | **REMOVE** |
| `check_missing_keywords.py` | 🟠 DEPRECATED | Debug script | **REMOVE** |
| `clear_cache.py` | 🟠 DEPRECATED | Cache clearing | **REMOVE** |
| `debug_visualization.py` | 🟠 DEPRECATED | Debug script | **REMOVE** |
| `find_critical.py` | 🟠 DEPRECATED | Debug script | **REMOVE** |

### 📂 archive/old_docs/

| File | Status | Recommendation |
|------|--------|----------------|
| All files | 🟠 DEPRECATED | **REMOVE** - Outdated documentation |

### 📂 archive/phase1_artifacts/

| File | Status | Recommendation |
|------|--------|----------------|
| All files | 🟠 DEPRECATED | **REMOVE** - Old phase artifacts |

### 📂 archive/removed_dead_code/

| File | Status | Recommendation |
|------|--------|----------------|
| All files | 🟠 DEPRECATED | **REMOVE** - Already removed code |

### 📂 archive/root_cleanup_20251221/

| File | Status | Recommendation |
|------|--------|----------------|
| All files | 🟠 DEPRECATED | **REMOVE** - Old cleanup artifacts |

### 📂 archive/test_outputs/

| File | Status | Recommendation |
|------|--------|----------------|
| All files | 🟠 DEPRECATED | **REVIEW** - May contain useful test data |

### 📂 archive/test_scripts/

| File | Status | Recommendation |
|------|--------|----------------|
| All files | 🟠 DEPRECATED | **REMOVE** - Old test scripts |

**Recommendation:** Delete entire `archive/` directory (~25+ files) to clean up codebase.

---

## 📂 nexus-llm-analytics-distribution_20251018_183430 (1)/ - 🟠 OLD DISTRIBUTION

This entire directory is an **old distribution snapshot**. 

| Status | Recommendation |
|--------|----------------|
| 🟠 DEPRECATED | **REMOVE** - Old distribution, creates confusion |

---

## 📂 Other Directories

| Directory | Status | Purpose | Recommendation |
|-----------|--------|---------|----------------|
| `.git/` | 🔵 UTILITY | Git version control | **KEEP** |
| `.pytest_cache/` | 🔵 UTILITY | Pytest cache | **KEEP** (auto-generated) |
| `.vscode/` | 🔵 UTILITY | VS Code settings | **KEEP** |
| `__pycache__/` | 🔵 UTILITY | Python bytecode cache | **KEEP** (auto-generated) |
| `chroma_db/` | ⚪ DATA | ChromaDB vector storage | **KEEP** |
| `logs/` | ⚪ DATA | Application logs | **KEEP** |
| `reports/` | ⚪ DATA | Generated reports | **KEEP** |
| `history/` | ⚪ DATA | Query history | **KEEP** |
| `broken/` | 🟠 DEPRECATED | Unknown | **REVIEW/REMOVE** |

---

## 🧹 CLEANUP RECOMMENDATIONS (VERIFIED)

### 1. Immediately Safe to Delete (~500+ files)

These folders contain OLD/ARCHIVED code not imported by any production code:

```
DELETE THESE:
- nexus-llm-analytics-distribution_20251018_183430 (1)/ (entire folder - old snapshot)
- archive/ (entire folder - old test scripts, docs, artifacts)
- broken/ (entire folder - broken archived code)
- src/backend/archive/ (entire folder - 14 legacy CrewAI files)
```

### 2. 🔴 DEAD CODE in src/backend/core/ (Safe to Remove)

Verified NOT imported by production code or tests:

| File | Evidence |
|------|----------|
| `utils.py` | Only imported by archived code |
| `optimized_tools.py` | Only in OLD distribution folder |
| `crewai_base.py` | CrewAI was replaced by plugins |
| `memory_optimizer.py` | Only unused script imports it |

### 3. 🟡 TEST-ONLY Files (Review Before Removing)

These are well-written but only used in performance tests:

| File | Keep If... |
|------|------------|
| `optimized_data_structures.py` | Running performance benchmarks |
| `optimized_llm_client.py` | Running performance benchmarks |
| `optimized_file_io.py` | Running performance benchmarks |
| `enhanced_cache_integration.py` | Using advanced caching features |
| `intelligent_query_engine.py` | Using advanced query routing |
| `model_detector.py` | Need model detection |

### 4. Keep for Legacy Compatibility (Low Impact)

```
- src/backend/core/crewai_import_manager.py (imported by optimizers.py, harmless)
```

---

## 🏁 PRODUCTION CODE TREE (Essential Files Only)

This is what your **minimal production codebase** looks like:

```
src/backend/
├── main.py                           # 🟢 FastAPI entry point
├── __init__.py
│
├── api/                              # 🟢 All API endpoints (8 files)
│   ├── analyze.py                    # POST /api/analyze
│   ├── upload.py                     # POST /api/upload
│   ├── health.py                     # GET /api/health/status
│   ├── visualize.py                  # POST /api/visualize
│   ├── report.py                     # POST /api/report
│   ├── history.py                    # GET/POST /api/history
│   ├── models.py                     # GET/POST /api/models
│   └── viz_enhance.py                # POST /api/viz/edit
│
├── core/                             # 🟢 Core modules (~23 files)
│   ├── config.py                     # Settings management
│   ├── llm_client.py                 # Ollama communication
│   ├── plugin_system.py              # Agent registry
│   ├── sandbox.py                    # Secure code execution
│   ├── security_guards.py            # Sandbox security
│   ├── self_correction_engine.py     # CoT correction
│   ├── cot_parser.py                 # CoT parsing
│   ├── document_indexer.py           # RAG indexing
│   ├── chromadb_client.py            # Vector DB
│   ├── model_selector.py             # Dynamic model selection
│   ├── dynamic_planner.py            # Analysis planning
│   ├── analysis_manager.py           # State tracking
│   ├── rate_limiter.py               # Rate limiting
│   ├── error_handling.py             # Error handling
│   ├── user_preferences.py           # User prefs
│   ├── advanced_cache.py             # Caching
│   ├── optimizers.py                 # Startup optimization
│   ├── websocket_manager.py          # Real-time updates
│   ├── query_parser.py               # Query classification
│   ├── query_complexity_analyzer.py  # Complexity scoring
│   ├── circuit_breaker.py            # Fault tolerance
│   ├── enhanced_logging.py           # Logging
│   └── enhanced_reports.py           # Report generation
│
├── plugins/                          # 🟢 All plugin agents (10 files)
│   ├── data_analyst_agent.py
│   ├── financial_agent.py
│   ├── ml_insights_agent.py
│   ├── rag_agent.py
│   ├── reporter_agent.py
│   ├── reviewer_agent.py
│   ├── sql_agent.py
│   ├── statistical_agent.py
│   ├── time_series_agent.py
│   ├── visualizer_agent.py
│   └── agents_config.json
│
├── services/
│   └── analysis_service.py           # 🟢 Service layer
│
├── agents/
│   ├── model_initializer.py          # 🟢 Model initialization
│   └── __init__.py
│
├── utils/
│   ├── data_utils.py                 # 🟢 Data utilities
│   ├── data_optimizer.py             # 🟢 Data optimization
│   └── __init__.py
│
├── visualization/
│   ├── dynamic_charts.py             # 🟢 Chart generation
│   ├── scaffold.py                   # 🟢 Viz scaffold
│   └── __init__.py
│
└── prompts/
    ├── cot_generator_prompt.txt      # 🟢 CoT prompt
    └── cot_critic_prompt.txt         # 🟢 CoT critic prompt
```

**Total essential backend files: ~50 files**

---

## 📊 FINAL FILE COUNT SUMMARY

| Category | Count | Action |
|----------|-------|--------|
| 🟢 **PRODUCTION** | ~50 backend + ~30 frontend | **KEEP** |
| 🟡 **TEST-ONLY** | ~8 core files | **REVIEW** |
| 🔴 **DEAD CODE** | ~5 files | **REMOVE** |
| 🟠 **LEGACY/ARCHIVE** | ~100+ files | **DELETE FOLDERS** |
| 📝 **DOCUMENTATION** | ~25 files | **KEEP** relevant |
| ⚪ **DATA** | ~50 files | **KEEP** essentials |
| 🧪 **TESTS** | ~50+ files | **KEEP** |

**Potential cleanup: ~100+ files can be safely removed**

---

## 🔄 DEPENDENCY GRAPH (Simplified)

```
main.py
├── api/
│   ├── analyze.py → services/analysis_service.py
│   ├── upload.py → utils/data_utils.py
│   ├── visualize.py → visualization/, core/sandbox.py
│   ├── report.py → core/enhanced_reports.py
│   ├── health.py → core/circuit_breaker.py, advanced_cache.py
│   └── models.py → core/model_selector.py
│
├── services/
│   └── analysis_service.py → core/plugin_system.py
│
├── core/
│   ├── plugin_system.py → plugins/*.py (runtime discovery)
│   ├── llm_client.py → circuit_breaker.py
│   ├── sandbox.py → security_guards.py
│   └── self_correction_engine.py → cot_parser.py, llm_client.py
│
├── plugins/ (discovered at runtime)
│   ├── data_analyst_agent.py → agents/model_initializer.py, dynamic_planner.py
│   ├── rag_agent.py → chromadb_client.py, document_indexer.py
│   └── ... (other agents)
│
└── utils/
    ├── data_utils.py → pandas
    └── data_optimizer.py → pandas, numpy
```

---

## ✅ FINAL NOTES

1. **Architecture:** CrewAI → Custom Plugin System (all CrewAI code is deprecated)

2. **Active Systems:**
   - `src/backend/plugins/` - Runtime-discovered agents
   - `src/backend/core/plugin_system.py` - Agent registry
   - `src/backend/services/analysis_service.py` - Request orchestration

3. **Entry Points:**
   - Backend: `src/backend/main.py` (FastAPI on port 8000)
   - Frontend: `src/frontend/app/page.tsx` (Next.js on port 3000)

4. **Key Features (Production):**
   - Multi-agent plugin system with capability-based routing
   - RAG with ChromaDB vector database
   - Self-correction engine (Generator→Critic→Feedback)
   - Secure sandbox execution (RestrictedPython)
   - Dynamic model selection based on RAM

---

*Generated: December 21, 2025*  
*Methodology: Deep code inspection + import tracing via grep analysis*

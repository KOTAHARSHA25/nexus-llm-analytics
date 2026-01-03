# Integration Map: Frontend → Backend → User Value

This document maps every frontend feature to its backend implementation and the user value it delivers.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Next.js)                          │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │  page.tsx   │  │file-upload   │  │model-settings│  │analytics- │ │
│  │  (Main UI)  │  │    .tsx      │  │    .tsx      │  │sidebar.tsx│ │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘ │
│         │                │                 │                │       │
│         └────────────────┴─────────────────┴────────────────┘       │
│                                    │                                 │
│                         lib/config.ts                               │
│                      getEndpoint() helper                           │
└────────────────────────────────────┬────────────────────────────────┘
                                     │ HTTP/REST
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        BACKEND (FastAPI)                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ analyze  │  │  upload  │  │  models  │  │  health  │  ...       │
│  │  .py     │  │   .py    │  │   .py    │  │   .py    │            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
│       │             │             │             │                   │
│       └─────────────┴─────────────┴─────────────┘                   │
│                           │                                         │
│                    ┌──────┴──────┐                                 │
│                    │ Orchestrator │                                 │
│                    │  + Agents    │                                 │
│                    └──────────────┘                                 │
│       ┌──────────────────────────────────────────┐                 │
│       │            10 Plugin Agents              │                 │
│       │  data_analysis | sql_query | statistical │                 │
│       │  text_analysis | time_series | web_search│                 │
│       │  pattern_discovery | data_engineering    │                 │
│       │  advanced_analytics | visualization      │                 │
│       └──────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Main Analysis Feature

### User Action
User types a natural language query and clicks "Analyze"

### Frontend
```
Component: app/page.tsx
Function:  handleSubmit()
Endpoint:  POST /api/analyze/
```

### Backend Flow
```
api/analyze.py:create_analysis()
    ↓
services/orchestrator.py:analyze()
    ↓
services/agent_router.py:route_query()
    ↓
[Selected Agent].analyze()
    ↓
Return: { code, insights, visualization, execution_time }
```

### User Value
**Instant data analysis with natural language** - User asks questions in plain English, gets Python code, statistical insights, and visualizations without writing any code.

---

## 2. File Upload Feature

### User Action
User uploads a CSV/Excel file for analysis

### Frontend
```
Component: components/file-upload.tsx
Functions: handleFileUpload(), handleDeleteFile(), handleDownloadFile()
Endpoints: 
  - POST /api/upload/
  - GET  /api/upload/files
  - DELETE /api/upload/delete-file/{name}
  - GET  /api/upload/download-file/{name}
```

### Backend Flow
```
api/upload.py:upload_file()
    ↓
services/file_handler.py:store_file()
    ↓
Save to: data/uploads/{filename}
    ↓
Return: { filename, size, columns, sample_data }
```

### User Value
**Seamless data ingestion** - User uploads their data once, it persists across sessions, and is automatically available for all analysis queries.

---

## 3. Model Selection Feature

### User Action
User selects which LLM to use (Gemini, GPT-4, Groq, etc.)

### Frontend
```
Component: components/model-settings.tsx
Functions: handleModelChange(), testConnection()
Endpoints:
  - GET  /api/models/available
  - POST /api/models/select
  - GET  /api/models/status
  - POST /api/models/test
```

### Backend Flow
```
api/models.py:get_available_models()
    ↓
services/model_manager.py:list_models()
    ↓
Return: [{ name, provider, status, api_key_required }]

api/models.py:select_model()
    ↓
services/model_manager.py:set_active_model()
    ↓
Update runtime configuration
```

### User Value
**Flexible AI provider choice** - User picks the model that fits their needs (speed vs accuracy, cost vs quality) and can switch on-the-fly.

---

## 4. Visualization Feature

### User Action
System generates visualizations as part of analysis, or user requests specific charts

### Frontend
```
Component: components/results-display.tsx
Implicit in: Analysis response contains visualization data
Endpoint:   POST /api/visualize/
```

### Backend Flow
```
api/visualize.py:create_visualization()
    ↓
services/visualization_engine.py:generate()
    ↓
Generate matplotlib/seaborn code
    ↓
Execute in sandbox
    ↓
Return: { image_base64, code, chart_type }
```

### User Value
**Automatic visual insights** - Complex data patterns rendered as clear charts without user needing to know matplotlib/seaborn.

---

## 5. Query History Feature

### User Action
User views past queries and re-runs them

### Frontend
```
Component: components/analytics-sidebar.tsx
Functions: loadHistory(), handleHistoryClick()
Endpoints:
  - GET /api/history/queries
  - GET /api/history/recent
```

### Backend Flow
```
api/history.py:get_queries()
    ↓
services/history_manager.py:load_history()
    ↓
Read from: data/history/query_history.json
    ↓
Return: [{ id, query, timestamp, result_summary }]
```

### User Value
**Continuity and reproducibility** - User never loses their work, can revisit and re-run past analyses.

---

## 6. Report Generation Feature

### User Action
User generates a downloadable report of their analysis session

### Frontend
```
Component: components/analytics-sidebar.tsx
Functions: handleGenerateReport(), downloadReport()
Endpoints:
  - POST /api/report/generate
  - GET  /api/report/download
```

### Backend Flow
```
api/report.py:generate_report()
    ↓
services/report_generator.py:create()
    ↓
Compile: queries + results + visualizations
    ↓
Return: { report_id, format, download_url }
```

### User Value
**Shareable deliverables** - User exports professional reports to share with stakeholders who don't have access to the tool.

---

## 7. Health Monitoring Feature

### User Action
User checks backend connection status, configures backend URL

### Frontend
```
Component: components/backend-url-settings.tsx
Functions: testConnection(), saveBackendUrl()
Endpoints:
  - GET /api/health/
  - GET /api/health/detailed
  - GET /api/health/network-info
```

### Backend Flow
```
api/health.py:health_check()
    ↓
Check: model availability, file system, dependencies
    ↓
Return: { status, models_available, uptime, version }
```

### User Value
**Reliability confidence** - User knows immediately if system is working and can troubleshoot connection issues.

---

## 8. Analysis Cancellation Feature

### User Action
User cancels a long-running analysis

### Frontend
```
Component: app/page.tsx
Function:  handleCancelAnalysis()
Endpoint:  POST /api/analyze/cancel/{analysis_id}
```

### Backend Flow
```
api/analyze.py:cancel_analysis()
    ↓
services/orchestrator.py:cancel()
    ↓
Interrupt running agent
    ↓
Return: { cancelled: true }
```

### User Value
**Control over long operations** - User can stop analyses that are taking too long without restarting the application.

---

## Integration Health Status

| Integration | Status | Verified |
|------------|--------|----------|
| Analysis → Orchestrator | ✅ Working | Yes |
| File Upload → Storage | ✅ Working | Yes |
| Model Select → ModelManager | ✅ Working | Yes |
| Visualize → VizEngine | ✅ Working | Yes |
| History → HistoryManager | ✅ Working | Yes |
| Report → ReportGenerator | ✅ Working | Yes |
| Health → HealthChecker | ✅ Working | Yes |
| Cancel → Orchestrator | ✅ Working | Yes |

---

## Unexposed Backend Capabilities

These backend features work but have no frontend UI:

| Capability | Endpoint | Potential User Value |
|-----------|----------|---------------------|
| Paste raw text | `/api/upload/raw-text` | Quick data entry without file |
| Preview file | `/api/upload/preview-file/{name}` | See data before analyzing |
| Edit visualization | `/api/viz/edit` | Modify charts with natural language |
| Explain visualization | `/api/viz/explain` | Understand what a chart shows |
| Recommend visualizations | `/api/viz/recommend` | Get chart suggestions |
| Search history | `/api/history/search` | Find specific past queries |
| Export history | `/api/history/export` | Backup query history |
| Query statistics | `/api/history/stats` | See usage patterns |

---

## API Endpoint Reference

### Used Endpoints (21)

```
POST   /api/analyze/                    # Main analysis
POST   /api/analyze/cancel/{id}         # Cancel analysis

POST   /api/upload/                     # Upload file
GET    /api/upload/files                # List files
GET    /api/upload/download-file/{name} # Download file
DELETE /api/upload/delete-file/{name}   # Delete file

GET    /api/models/available            # List models
POST   /api/models/select               # Select model
GET    /api/models/status               # Model status
POST   /api/models/test                 # Test connection

POST   /api/visualize/                  # Generate visualization

GET    /api/history/queries             # Query history
GET    /api/history/recent              # Recent queries

POST   /api/report/generate             # Generate report
GET    /api/report/download             # Download report

GET    /api/health/                     # Health check
GET    /api/health/detailed             # Detailed health
GET    /api/health/network-info         # Network info
```

### Orphaned Endpoints (29)
See AUDIT_REPORT.md for full categorization.

---

*This integration map reflects the verified state of the system after Phase 2 audit.*

# COMPREHENSIVE TESTING MASTER PLAN
**Date**: December 16, 2025  
**Objective**: Exhaustive unit and integration testing of entire codebase  
**Priority**: ACCURACY > COMPLETENESS > STABILITY > PERFORMANCE

---

## 📋 COMPLETE FILE INVENTORY

### 🔧 Backend Core Modules (src/backend/core/) - 36 files
1. ✅ `cot_parser.py` - TESTED (Accuracy: 100%)
2. ✅ `intelligent_router.py` - TESTED (Accuracy: 100%)
3. ✅ `query_complexity_analyzer.py` - TESTED (Routing tests)
4. ⏳ `query_complexity_analyzer_v2.py` - NOT TESTED
5. ⏳ `analysis_manager.py` - NOT TESTED
6. ⏳ `chromadb_client.py` - NOT TESTED (RAG backend)
7. ⏳ `advanced_cache.py` - NOT TESTED
8. ⏳ `circuit_breaker.py` - NOT TESTED
9. ⏳ `config.py` - NOT TESTED
10. ⏳ `crewai_base.py` - NOT TESTED
11. ⏳ `crewai_import_manager.py` - NOT TESTED
12. ⏳ `crew_singleton.py` - NOT TESTED
13. ⏳ `document_indexer.py` - NOT TESTED
14. ⏳ `enhanced_cache_integration.py` - NOT TESTED
15. ⏳ `enhanced_logging.py` - NOT TESTED
16. ⏳ `enhanced_reports.py` - NOT TESTED
17. ⏳ `error_handling.py` - NOT TESTED
18. ⏳ `intelligent_query_engine.py` - NOT TESTED
19. ⏳ `llm_client.py` - NOT TESTED (Critical!)
20. ⏳ `model_detector.py` - NOT TESTED
21. ⏳ `model_selector.py` - NOT TESTED
22. ⏳ `optimized_data_structures.py` - NOT TESTED
23. ⏳ `optimized_file_io.py` - NOT TESTED
24. ⏳ `optimized_llm_client.py` - NOT TESTED
25. ⏳ `optimized_tools.py` - NOT TESTED
26. ⏳ `optimizers.py` - NOT TESTED
27. ⏳ `plugin_system.py` - NOT TESTED
28. ⏳ `query_parser.py` - NOT TESTED
29. ⏳ `rate_limiter.py` - NOT TESTED
30. ⏳ `sandbox.py` - NOT TESTED
31. ⏳ `security_guards.py` - NOT TESTED
32. ⏳ `self_correction_engine.py` - NOT TESTED
33. ⏳ `user_preferences.py` - NOT TESTED
34. ⏳ `utils.py` - NOT TESTED
35. ⏳ `websocket_manager.py` - NOT TESTED
36. ⏳ `__init__.py` - NOT TESTED

### 🔌 Plugin Agents (src/backend/plugins/) - 5 files
1. ⚠️ `statistical_agent.py` - TESTED (Accuracy: UNCERTAIN - format issues)
2. ⚠️ `financial_agent.py` - TESTED (Accuracy: UNCERTAIN - format issues)
3. ✅ `time_series_agent.py` - TESTED (Accuracy: 100%)
4. ⚠️ `ml_insights_agent.py` - TESTED (Accuracy: UNCERTAIN - format issues)
5. ⏳ `sql_agent.py` - NOT TESTED

### 🤖 Agent System (src/backend/agents/) - 8 files
1. ⏳ `agent_factory.py` - NOT TESTED
2. ⏳ `analysis_executor.py` - NOT TESTED
3. ⏳ `crew_manager.py` - NOT TESTED
4. ⏳ `data_agent.py` - NOT TESTED
5. ⏳ `model_initializer.py` - NOT TESTED
6. ⏳ `rag_handler.py` - NOT TESTED
7. ⏳ `specialized_agents.py` - NOT TESTED
8. ⏳ `__init__.py` - NOT TESTED

### 🌐 API Endpoints (src/backend/api/) - 8 files
1. ⏳ `analyze.py` - NOT TESTED (Critical!)
2. ⏳ `health.py` - NOT TESTED
3. ⏳ `history.py` - NOT TESTED
4. ⏳ `models.py` - NOT TESTED
5. ⏳ `report.py` - NOT TESTED
6. ⏳ `upload.py` - NOT TESTED (Critical!)
7. ⏳ `visualize.py` - NOT TESTED
8. ⏳ `viz_enhance.py` - NOT TESTED

### 📊 Visualization (src/backend/visualization/)
- ⏳ Files need inventory

### 🛠️ Utilities (src/backend/utils/)
- ⏳ Files need inventory

### 🎨 Frontend (src/frontend/)
- ⏳ Components need inventory
- ⏳ Hooks need inventory
- ⏳ API integration needs testing

### 🔗 Main Entry Point
- ⏳ `main.py` - NOT TESTED (FastAPI app)

---

## 🎯 TESTING PHASES

### Phase 1: Backend Core Critical Path ⚡ [IN PROGRESS]
**Priority**: Test the most critical modules first

1. **LLM Communication** (CRITICAL)
   - [ ] `llm_client.py` - Test model communication
   - [ ] `optimized_llm_client.py` - Test optimized client
   - [ ] Test real queries to phi3:mini, tinyllama, llama3.1:8b

2. **Routing & Query Processing** (CRITICAL)
   - [✅] `intelligent_router.py` - DONE (100% accuracy)
   - [✅] `query_complexity_analyzer.py` - DONE
   - [ ] `query_complexity_analyzer_v2.py` - Test v2 improvements
   - [ ] `query_parser.py` - Test query parsing logic

3. **CoT & Self-Correction** (CRITICAL)
   - [✅] `cot_parser.py` - DONE (100% accuracy)
   - [ ] `self_correction_engine.py` - Test correction logic

4. **Plugin System** (CRITICAL)
   - [ ] `plugin_system.py` - Test plugin loading/execution
   - [⚠️] Fix plugin agent return formats
   - [ ] Re-test all 5 plugins with real data

### Phase 2: Plugin Agents Deep Dive 🔌 [PENDING]
**Objective**: Fix and verify all plugin calculation accuracy

1. **Statistical Agent**
   - [ ] Test descriptive statistics (mean, median, std, min, max)
   - [ ] Test hypothesis testing
   - [ ] Test correlation analysis
   - [ ] Verify return format includes structured data

2. **Financial Agent**
   - [ ] Test profitability analysis
   - [ ] Test ROI calculations
   - [ ] Test cost-benefit analysis
   - [ ] Verify numerical values in output

3. **Time Series Agent**
   - [✅] Trend detection - DONE
   - [ ] Seasonality detection
   - [ ] Forecasting accuracy
   - [ ] Anomaly detection

4. **ML Insights Agent**
   - [ ] Clustering (k-means, DBSCAN)
   - [ ] Classification metrics
   - [ ] Feature importance
   - [ ] Verify structured output

5. **SQL Agent**
   - [ ] Query generation
   - [ ] Query execution
   - [ ] Result formatting
   - [ ] Error handling

### Phase 3: API Endpoints Testing 🌐 [PENDING]
**Objective**: Test all REST API endpoints with real requests

1. **Upload Endpoint** (`/upload`)
   - [ ] CSV upload handling
   - [ ] Data validation
   - [ ] Storage mechanism
   - [ ] Error responses

2. **Analyze Endpoint** (`/analyze`)
   - [ ] Query processing
   - [ ] Model selection
   - [ ] Response formatting
   - [ ] Error handling

3. **Visualize Endpoint** (`/visualize`)
   - [ ] Chart generation
   - [ ] Data preparation
   - [ ] Multiple chart types
   - [ ] Error handling

4. **Report Endpoint** (`/report`)
   - [ ] Markdown generation
   - [ ] PDF generation
   - [ ] Complete report structure
   - [ ] Download handling

5. **Health Endpoint** (`/health`)
   - [ ] System status check
   - [ ] Ollama connection
   - [ ] Database status
   - [ ] Resource usage

### Phase 4: Agent System Testing 🤖 [PENDING]
**Objective**: Test agent orchestration and coordination

1. **Agent Factory**
   - [ ] Agent creation
   - [ ] Agent initialization
   - [ ] Agent configuration

2. **Crew Manager**
   - [ ] Multi-agent coordination
   - [ ] Task delegation
   - [ ] Result aggregation

3. **Data Agent**
   - [ ] Data loading
   - [ ] Data preprocessing
   - [ ] Data validation

4. **RAG Handler**
   - [ ] Document indexing
   - [ ] Vector search
   - [ ] Context retrieval

### Phase 5: Data Processing Testing 📊 [PENDING]
**Objective**: Test data handling with real CSVs

1. **File Upload & Storage**
   - [ ] CSV parsing
   - [ ] Data validation
   - [ ] Storage mechanism
   - [ ] Error handling

2. **Data Transformation**
   - [ ] Column detection
   - [ ] Type inference
   - [ ] Missing value handling
   - [ ] Outlier detection

3. **Query Execution on Data**
   - [ ] Filtering
   - [ ] Aggregation
   - [ ] Joins (if multiple datasets)
   - [ ] Complex queries

### Phase 6: Visualization Testing 📈 [PENDING]
**Objective**: Test chart generation accuracy

1. **Chart Types**
   - [ ] Line charts
   - [ ] Bar charts
   - [ ] Scatter plots
   - [ ] Pie charts
   - [ ] Heatmaps

2. **Data Accuracy**
   - [ ] Correct values plotted
   - [ ] Proper axis labels
   - [ ] Legend accuracy
   - [ ] Color mapping

### Phase 7: Frontend Testing 🎨 [PENDING]
**Objective**: Test UI components and interactions

1. **File Upload Component**
   - [ ] File selection
   - [ ] Upload progress
   - [ ] Error display
   - [ ] Success confirmation

2. **Query Input Component**
   - [ ] Text input
   - [ ] Submit handling
   - [ ] Loading states
   - [ ] Error display

3. **Results Display Component**
   - [ ] Text rendering
   - [ ] Chart embedding
   - [ ] Table display
   - [ ] Download buttons

4. **Settings Component**
   - [ ] Model selection
   - [ ] Configuration updates
   - [ ] Persistence

### Phase 8: Integration Testing 🔗 [PENDING]
**Objective**: Test component interactions

1. **Frontend ↔ Backend**
   - [ ] Upload → Analyze flow
   - [ ] Query → Response flow
   - [ ] Visualize → Display flow
   - [ ] Report → Download flow

2. **Backend ↔ Backend**
   - [ ] Router → Plugin flow
   - [ ] CoT → Self-correction flow
   - [ ] Analysis → Visualization flow
   - [ ] Primary → Review analysis flow

3. **API ↔ Services**
   - [ ] Backend → Ollama
   - [ ] Backend → ChromaDB
   - [ ] Backend → File system

### Phase 9: End-to-End Testing 🎯 [PENDING]
**Objective**: Test complete user workflows

1. **Complete Analysis Workflow**
   - [ ] Upload CSV with known data
   - [ ] Ask question with known answer
   - [ ] Verify response accuracy
   - [ ] Generate visualization
   - [ ] Create report
   - [ ] Download results

2. **Error Recovery Workflow**
   - [ ] Invalid CSV upload
   - [ ] Malformed query
   - [ ] Model failure
   - [ ] Retry mechanism

3. **Complex Query Workflow**
   - [ ] Multi-step analysis
   - [ ] Plugin orchestration
   - [ ] Primary + Review
   - [ ] Self-correction

### Phase 10: Performance & Stress Testing 🔥 [PENDING]
**Objective**: Test under load

1. **Response Time**
   - [ ] Simple queries (<1s)
   - [ ] Medium queries (<5s)
   - [ ] Complex queries (<15s)

2. **Concurrent Requests**
   - [ ] Multiple users
   - [ ] Rate limiting
   - [ ] Queue management

3. **Large Data**
   - [ ] Large CSV (>10MB)
   - [ ] Many rows (>100k)
   - [ ] Many columns (>100)

---

## 📊 CURRENT STATUS

### Completion Summary
- **Total Files Identified**: ~80+ files
- **Files Tested**: 6 files
- **Tests Passed**: 3 tests (Routing, Time Series, CoT)
- **Tests Failed/Uncertain**: 3 tests (Statistical, Financial, ML)
- **Overall Completion**: ~7% (6/80)

### Critical Issues Found
1. ❌ Plugin agents return `success=True` but missing calculation data
2. ❌ Statistical agent: No 'statistics' dict in response
3. ❌ Financial agent: No numerical values in analysis text
4. ❌ ML agent: No 'clusters' dict in response
5. ⚠️ Backend failed to start (exit code 1 on uvicorn)

### Next Immediate Actions
1. Fix plugin agent return formats
2. Test LLM client communication
3. Test API endpoints with real backend
4. Fix backend startup issue

---

## 🧪 TESTING METHODOLOGY

### Unit Testing Approach
```python
# Template for each file test
def test_<module_name>():
    # 1. Import module
    # 2. Prepare REAL test data
    # 3. Execute function/method
    # 4. Verify actual vs expected output
    # 5. Check edge cases
    # 6. Document findings
```

### Integration Testing Approach
```python
# Template for integration test
def test_integration_<component_a>_<component_b>():
    # 1. Set up both components
    # 2. Trigger interaction
    # 3. Verify data flow
    # 4. Check error handling
    # 5. Document findings
```

### Real Data Test Samples
- CSV with known totals (e.g., revenue=$5,850)
- Queries with known answers
- Statistical data with known mean/median
- Time series with known trends
- Financial data with known profits

---

## 📈 SUCCESS CRITERIA

### Accuracy Targets
- Unit tests: ≥95% pass rate
- Integration tests: ≥90% pass rate
- End-to-end tests: ≥85% pass rate
- Plugin calculation accuracy: 100% (verified with ground truth)

### Coverage Targets
- Code coverage: ≥80%
- Function coverage: 100%
- Branch coverage: ≥70%
- Integration paths: 100%

---

**Last Updated**: December 16, 2025  
**Status**: Phase 1 in progress (7% complete)  
**Next Milestone**: Fix plugin formats → Test LLM client → Test API endpoints

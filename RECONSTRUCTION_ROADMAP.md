# RECONSTRUCTION ROADMAP
## Nexus LLM Analytics System - Forensic Analysis & Reconstruction Plan

**Created:** December 16, 2025  
**Status:** BASELINE TESTING COMPLETE - CRITICAL FINDINGS DOCUMENTED  
**Author:** AI Analysis Agent (Opus 4.5)
**Last Updated:** December 16, 2025

---

## ⚠️ CRITICAL FINDING: NEW VERSION RETURNS EMPTY RESULTS

### Baseline Test Results (December 16, 2025)

| Metric | OLD Version | NEW Version |
|--------|-------------|-------------|
| Tests Passed | 15/15 | 15/15 |
| **Results Returned** | **ACTUAL CONTENT** | **EMPTY STRINGS** |
| Avg Simple Query Time | 46.1s | 12.97s |
| Avg Medium Query Time | 33.0s | 8.25s |
| Avg Complex Query Time | 53.0s | 8.53s |
| Total Time | 660.6s | 129.9s |

### Root Cause Identified
The NEW version has a **critical bug** in `model_initializer.py`:
```python
# BUG: select_optimal_models() returns TUPLE, not dict
models = ModelSelector.select_optimal_models()
primary_model = models.get("primary")  # FAILS: tuple has no .get()
```

This causes `LLM initialization failed: 'tuple' object has no attribute 'get'`, resulting in:
- All queries return `"result": ""` (empty string)
- Status reports "success" despite failure
- No actual LLM inference occurs

### FIX APPLIED (pending validation)
Changed to tuple unpacking:
```python
primary_model, review_model, _ = ModelSelector.select_optimal_models()
```

### Accuracy Assessment: OLD Version

The OLD version returns content but has **quality issues**:

| Query Type | Observation | Quality |
|------------|-------------|---------|
| Simple aggregations | Returns values ("$500,000") | ⚠️ May hallucinate |
| Row counts | Correct ("50 rows") | ✅ Accurate |
| Correlations | Qualitative ("strong positive") | ⚠️ No numeric value |
| Complex analysis | Plausible narratives | ⚠️ Not verifiable |

**Key Finding**: OLD version's "Reviewer" agent often **contradicts** the primary analysis, e.g.:
> "The total revenue is $500,00... Reviewer's correction: The total revenue from 'sales_data.csv' is $500,000."

This suggests the two-agent review system catches errors but creates **confusing outputs**

---

## 1. SYSTEM OVERVIEW

### 1.1 Purpose
This is an **LLM-powered data analytics system** that enables natural language queries against structured (CSV, JSON) and unstructured (PDF, TXT, DOCX) data files. The system uses local Ollama models for inference.

### 1.2 Core Workflows
1. **Structured Data Analysis**: User uploads CSV/JSON → asks natural language questions → system generates analysis
2. **Unstructured Data Analysis (RAG)**: User uploads documents → system indexes content → answers questions via retrieval-augmented generation
3. **Multi-File Analysis**: Join and analyze multiple data files
4. **Visualization Generation**: Create Plotly charts from data
5. **Report Generation**: Compile analysis into professional reports

### 1.3 Two Versions Identified

| Aspect | OLD Version (Distribution) | NEW Version (Root) |
|--------|---------------------------|-------------------|
| Location | `nexus-llm-analytics-distribution_20251018_183430 (1)/` | Root `src/` directory |
| Date | October 18, 2025 | Current (Nov-Dec 2025) |
| Architecture | Monolithic `crew_manager.py` (960 lines) | Refactored, modular (610 lines + helpers) |
| CrewAI Usage | Direct agent creation in crew_manager | Delegated to AgentFactory, ModelInitializer |
| Features | Basic analysis, review protocol | + Intelligent routing, CoT, query complexity analyzer |
| Code Quality | Single-file approach | Separated concerns, lazy loading |

---

## 2. REPOSITORY MAP

### 2.1 Directory Structure

```
nexus-llm-analytics-dist/
├── [OLD VERSION] nexus-llm-analytics-distribution_20251018_183430 (1)/
│   ├── src/backend/
│   │   ├── main.py                    (269 lines - FastAPI entry)
│   │   ├── agents/
│   │   │   ├── crew_manager.py        (960 lines - MONOLITHIC)
│   │   │   ├── controller_agent.py    (Legacy)
│   │   │   ├── data_agent.py          
│   │   │   ├── rag_agent.py
│   │   │   ├── review_agent.py
│   │   │   ├── visualization_agent.py
│   │   │   └── report_agent.py
│   │   ├── core/
│   │   │   ├── llm_client.py          (133 lines)
│   │   │   ├── query_parser.py        (379 lines)
│   │   │   ├── model_selector.py
│   │   │   ├── crewai_base.py
│   │   │   └── [19 other modules]
│   │   └── api/
│   │       └── analyze.py             (247 lines - simpler)
│   ├── data/samples/                   (11 sample files)
│   └── tests/
│
├── [NEW VERSION] src/
│   ├── backend/
│   │   ├── main.py                    (217 lines - FastAPI entry)
│   │   ├── agents/
│   │   │   ├── crew_manager.py        (610 lines - REFACTORED facade)
│   │   │   ├── agent_factory.py       (NEW - 203 lines)
│   │   │   ├── model_initializer.py   (NEW - 219 lines)
│   │   │   ├── analysis_executor.py   (NEW - 510 lines)
│   │   │   ├── rag_handler.py         (NEW - 336 lines)
│   │   │   ├── data_agent.py
│   │   │   └── specialized_agents.py
│   │   ├── core/
│   │   │   ├── llm_client.py          (208 lines - expanded)
│   │   │   ├── query_parser.py        (383 lines)
│   │   │   ├── intelligent_router.py  (NEW - 465 lines)
│   │   │   ├── query_complexity_analyzer.py (NEW - 609 lines)
│   │   │   ├── query_complexity_analyzer_v2.py (NEW)
│   │   │   ├── self_correction_engine.py (NEW - 380 lines)
│   │   │   ├── cot_parser.py          (NEW - 158 lines)
│   │   │   ├── enhanced_logging.py    (NEW)
│   │   │   ├── enhanced_reports.py    (NEW)
│   │   │   ├── model_detector.py      (NEW)
│   │   │   └── [25 other modules]
│   │   └── api/
│   │       └── analyze.py             (386 lines - expanded)
│   └── frontend/
│
├── data/samples/                       (22+ sample files)
├── config/
│   └── cot_review_config.json         (CoT configuration)
├── tests/
│   └── test_phase7_routing.py         (Routing system tests)
└── requirements.txt                    (75 dependencies)
```

### 2.2 Entry Points
- **Backend**: `src/backend/main.py` → FastAPI application
- **Analysis**: `POST /analyze/` → `crew_manager.handle_query()`
- **Launch Script**: `scripts/launch.py`

### 2.3 Key Dependencies (Both Versions)
- CrewAI >= 0.28.0 (agent orchestration)
- LiteLLM >= 1.28.0 (LLM abstraction)
- LangChain-Community (utilities)
- FastAPI (web framework)
- Pandas/Polars (data processing)
- ChromaDB (vector storage for RAG)

---

## 3. GROUND TRUTH ANCHORS

### 3.1 Sample Data Files Identified

| File | Location | Type | Rows | Purpose |
|------|----------|------|------|---------|
| sales_data.csv | data/samples/ | CSV | 102 | Product sales by region |
| StressLevelDataset.csv | data/samples/ | CSV | 1,102 | Mental health dataset |
| simple.json | data/samples/ | JSON | 5 | Basic sales data |
| 1.json | data/samples/ | JSON | 1 | Minimal test data |
| complex_nested.json | data/samples/ | JSON | - | Nested structure test |
| financial_quarterly.json | data/samples/ | JSON | - | Financial data |

### 3.2 Test Queries for Ground Truth Validation
Based on test files and data structure:

```python
# Simple queries (should route to FAST tier)
"What is the average sales?"
"Count total records"
"Show top 5 products"

# Medium queries (should route to BALANCED tier)  
"Compare year-over-year sales growth by region"
"Show correlation between sales and marketing_spend"

# Complex queries (should route to FULL_POWER tier)
"Predict customer churn using machine learning"
"Run Monte Carlo simulation for risk assessment"
```

### 3.3 Expected Baseline Behavior (from code analysis)

**OLD Version Inference Flow:**
```
User Query → crew_manager.handle_query()
  → analyze_structured_data() or analyze_unstructured_data()
  → _perform_structured_analysis()
  → Direct LLM call with simple prompt: "DATA: {data}\nQUESTION: {query}\nAnswer directly..."
  → Review LLM call: "Is this correct? Reply 'Approved' or give correct answer"
  → Return combined result
```

**NEW Version Inference Flow:**
```
User Query → crew_manager.handle_query()
  → analysis_executor.analyze_structured() or rag_handler.analyze_unstructured()
  → intelligent_router.route() → query_complexity_analyzer.analyze()
  → Model selection based on complexity score
  → Optional CoT self-correction loop (if complexity > 0.5)
  → Optional review step
  → Return result with routing metadata
```

### 3.4 CRITICAL OBSERVATION: Accuracy Regression Source

**SUSPECTED ROOT CAUSE:** The NEW version's intelligent routing and CoT system may be:
1. **Over-engineering simple queries** - Adding complexity where none needed
2. **Introducing latency** - Multiple LLM calls instead of one
3. **Parsing failures** - CoT tag extraction may fail silently
4. **Model mismatch** - Routing to inappropriate model tiers

**OLD version's simplicity** (direct LLM call + review) may produce more reliable results for typical queries.

---

## 4. FEATURE INVENTORY

### 4.1 Features in NEW Version

| Feature | Location | Accuracy-Sensitive? | Value |
|---------|----------|---------------------|-------|
| **Intelligent Routing** | `intelligent_router.py` | YES - HIGH RISK | Routes queries to appropriate model tiers |
| **Query Complexity Analyzer** | `query_complexity_analyzer.py` | YES - HIGH RISK | Scores queries 0-1 for routing decisions |
| **Chain-of-Thought (CoT)** | `self_correction_engine.py`, `cot_parser.py` | YES - HIGH RISK | Generator→Critic feedback loop |
| **Enhanced Query Parser** | `query_parser.py` | MEDIUM | Intent classification, column extraction |
| **Multi-File Analysis** | `crew_manager.py` | LOW | Join and analyze multiple files |
| **Text Input Analysis** | `analyze.py` | LOW | Direct text analysis without file |
| **Lazy Loading** | `model_initializer.py` | NO | Performance optimization |
| **Modular Architecture** | `agent_factory.py`, etc. | NO | Code organization |
| **Enhanced Error Messages** | `analyze.py` | NO | Better user experience |

### 4.2 Features MISSING from OLD Version
- Intelligent routing (all queries use same model)
- Query complexity analysis
- CoT self-correction
- Multi-file support
- Text input analysis (no file needed)
- Modular code structure

### 4.3 Features in BOTH Versions
- CrewAI agent orchestration
- Primary + Review model pattern
- Structured data analysis (CSV, JSON)
- RAG analysis (PDF, TXT, DOCX)
- Visualization generation
- Report generation
- Plugin system
- Caching

---

## 5. ACCURACY SURFACE MAP

### 5.1 Where Accuracy is Determined

```
ACCURACY-CRITICAL PATH (traced through code):

1. QUERY PARSING (medium risk)
   - IntentClassifier.classify_intent() → regex patterns
   - ColumnExtractor.extract_columns() → fuzzy matching
   
2. ROUTING DECISION (HIGH RISK - NEW only)
   - QueryComplexityAnalyzer.analyze() → complexity score
   - IntelligentRouter.route() → model selection
   - Thresholds: FAST < 0.25 < BALANCED < 0.45 < FULL_POWER
   
3. PROMPT CONSTRUCTION (HIGH RISK)
   OLD: Simple direct prompt "DATA:...\nQUESTION:...\nAnswer directly"
   NEW: Complex prompts through CoT template system
   
4. LLM INFERENCE (inherent variability)
   - LLMClient.generate() → Ollama API call
   - Circuit breaker, adaptive timeouts
   
5. POST-PROCESSING (medium risk)
   OLD: Simple review check "Is this correct?"
   NEW: CoT parsing, critic evaluation, feedback synthesis

6. RESULT EXTRACTION (medium risk)
   - CoTParser.parse() → tag extraction [REASONING]...[OUTPUT]
   - Fallback to raw response on parse failure
```

### 5.2 Fragile/High-Risk Areas

| Area | Risk | Reason |
|------|------|--------|
| Query Complexity Scoring | HIGH | Keyword-based heuristics may misclassify |
| Routing Thresholds | HIGH | 0.25/0.45 thresholds are empirically tuned |
| CoT Tag Parsing | HIGH | Relies on LLM producing exact tag format |
| Model Selection | MEDIUM | Depends on which models are installed |
| Prompt Templates | MEDIUM | Prompt files may be missing |

### 5.3 Components That MUST NOT Change (Invariants)

1. **LLM Input Immutability**: Query text must not be mutated after parsing
2. **Model Call Isolation**: Each LLM call should be independent
3. **Response Preservation**: LLM output must not be modified in accuracy-critical paths
4. **Deterministic Fallbacks**: When routing fails, must fall back predictably

---

## 6. ORCHESTRATION ANALYSIS

### 6.1 CrewAI Usage Assessment

**Is CrewAI present in NEW version?** YES
- `from crewai import Agent, Task, Crew, Process` in agent_factory.py
- Agents: DataAnalyst, RAGSpecialist, Reviewer, Visualizer, Reporter

**Is CrewAI present in OLD version?** YES
- Same agent structure but defined inline in crew_manager.py
- `_Agent`, `_Task`, `_Crew`, `_Process` stored as class attributes

### 6.2 CrewAI Correlation with Accuracy Issues

**CRITICAL FINDING:** The OLD version BYPASSES CrewAI for actual inference!

```python
# OLD VERSION crew_manager.py lines 540-560:
# "DIRECT LLM CALL - Bypass CrewAI completely to avoid hallucinations"
direct_prompt = f"""DATA: {filename}
{data_info}
QUESTION: {query}
Answer directly in 1 sentence. NO code, NO JSON, just the answer:"""

analysis_response = self.primary_llm.call([{"role": "user", "content": direct_prompt}])
```

The OLD version explicitly bypasses CrewAI agents because they caused "hallucinations". It uses direct LLM calls with simple prompts.

**NEW version** reintroduces CrewAI-style complexity through:
- Multiple routing decisions
- CoT self-correction loop (multiple LLM calls)
- Complex prompt templates

### 6.3 Decision: Direct Orchestration vs Framework

**RECOMMENDATION: HYBRID APPROACH**

| Use Case | Recommendation | Justification |
|----------|---------------|---------------|
| Simple queries | Direct LLM call (like OLD) | Proven reliability, low latency |
| Complex analysis | Optional CoT + Review | May improve quality for hard queries |
| RAG queries | Direct RAG retrieval + LLM | Avoid CrewAI agent overhead |
| Visualization | CrewAI acceptable | Non-accuracy-critical |
| Reports | CrewAI acceptable | Non-accuracy-critical |

**Key Principle:** CrewAI should NOT be in the accuracy-critical inference path for data analysis. Use it only for orchestration of multi-step workflows where intermediate errors are recoverable.

---

## 7. DIFFERENCE & RISK MATRIX

### 7.1 Key Architectural Differences

| Component | OLD Version | NEW Version | Risk |
|-----------|-------------|-------------|------|
| crew_manager.py | 960 lines monolithic | 610 lines facade | LOW (refactoring) |
| LLM calls | Direct, bypasses CrewAI | Through routing system | HIGH |
| Prompt format | Simple inline string | Template files + CoT | HIGH |
| Model selection | Fixed/user-pref | Intelligent routing | MEDIUM |
| Query analysis | Basic parsing | Complexity scoring | HIGH |
| Review step | Simple approval check | CoT critic loop | MEDIUM |

### 7.2 Suspected Bugs in BOTH Versions

**OLD Version:**
1. ⚠️ Model switching on retry (`original_primary` variable may not exist)
2. ⚠️ Placeholder review response when force_model is used
3. ⚠️ No cancellation check in visualization/report generation

**NEW Version:**
1. 🔴 Duplicate `_calculate_adaptive_timeout` method in llm_client.py (lines 108-166 and 168-212)
2. 🔴 model_initializer uses deprecated chromadb settings (`chroma_db_impl="duckdb+parquet"`)
3. ⚠️ cot_review_config.json references non-existent "cot_review" key (should be root level)
4. ⚠️ Potential infinite loop if CoT parsing always fails
5. ⚠️ intelligent_router may fail if qwen models not installed

### 7.3 Risk Classification

| Component | Classification | Action |
|-----------|---------------|--------|
| LLM Client core | LIKELY CORRECT | Preserve |
| Direct prompt path (OLD) | LIKELY CORRECT | Preserve as fallback |
| Query parser basics | LIKELY CORRECT | Preserve |
| Intelligent routing | UNCERTAIN | Isolate, make configurable |
| CoT self-correction | UNCERTAIN | Disable by default |
| Query complexity analyzer | UNCERTAIN | Validate against benchmarks |
| Multi-file analysis | LIKELY CORRECT | Preserve |
| RAG handler | LIKELY CORRECT | Preserve |

---

## 8. INVARIANTS & ASSUMPTIONS

### 8.1 Known Invariants (Proven from Code)

1. **Query length limit**: 1000 characters (enforced in analyze.py)
2. **File resolution order**: uploads/ → samples/ → samples/csv/ → samples/json/
3. **Supported structured formats**: .csv, .json, .xlsx, .xls
4. **Supported unstructured formats**: .pdf, .txt, .docx, .pptx, .rtf
5. **Routing thresholds**: FAST < 0.25 < BALANCED < 0.45 < FULL_POWER
6. **CoT tags**: `[REASONING]...[/REASONING]` and `[OUTPUT]...[/OUTPUT]`

### 8.2 Assumed Invariants (Mark as Assumptions)

1. **ASSUMPTION**: Ollama is running on localhost:11434
2. **ASSUMPTION**: At least one model (tinyllama/phi3) is installed
3. **ASSUMPTION**: Simple queries benefit from faster models
4. **ASSUMPTION**: Complex queries require larger models
5. **ASSUMPTION**: Review step improves accuracy (not proven)
6. **ASSUMPTION**: CoT improves accuracy for complex queries (not proven)

### 8.3 Unknowns Requiring Investigation

1. ❓ What is the actual accuracy difference between versions?
2. ❓ Does intelligent routing improve or hurt accuracy?
3. ❓ Does CoT self-correction help or add noise?
4. ❓ Which model tier thresholds are optimal?
5. ❓ What prompts produce the most accurate results?

---

## 9. RECONSTRUCTION PLAN

### Phase 1: FOUNDATION (No Accuracy Changes)
**Goal:** Establish clean, tested base

1. Fix obvious bugs in BOTH versions:
   - Remove duplicate `_calculate_adaptive_timeout` method
   - Fix chromadb settings deprecation
   - Fix cot_review_config.json structure
   
2. Create unified test harness:
   - Run same queries through both versions
   - Capture outputs for comparison
   
3. Preserve OLD version inference path:
   - Extract direct LLM call pattern as `DirectInferencePath`
   - This becomes the "safe fallback"

### Phase 2: FEATURE FLAG SYSTEM
**Goal:** Enable A/B testing of accuracy-affecting features

1. Create feature configuration:
   ```json
   {
     "enable_intelligent_routing": false,  // Default OFF
     "enable_cot_self_correction": false,  // Default OFF
     "enable_review_step": true,           // Keep simple review
     "routing_thresholds": {"fast": 0.25, "balanced": 0.45},
     "fallback_to_direct": true
   }
   ```

2. Wrap NEW features in conditional execution:
   - If routing enabled → use IntelligentRouter
   - Else → use direct LLM call

### Phase 3: ACCURACY VALIDATION
**Goal:** Measure actual accuracy impact

1. Define accuracy metrics:
   - Factual correctness (does answer match data?)
   - Relevance (does answer address the question?)
   - Completeness (are key insights captured?)

2. Run benchmark suite:
   - 20 simple queries
   - 20 medium queries  
   - 20 complex queries
   - Compare OLD vs NEW vs HYBRID

3. Document findings before proceeding

### Phase 4: SELECTIVE FEATURE INTEGRATION
**Goal:** Reintroduce NEW features that IMPROVE accuracy

Only if Phase 3 proves benefit:
1. Enable intelligent routing (if accuracy >= OLD)
2. Enable CoT for complex queries only (if accuracy improves)
3. Preserve OLD path as fallback for failures

### Phase 5: CLEANUP & VERIFICATION
**Goal:** Production-ready system

1. Remove dead code
2. Consolidate duplicate functionality
3. Full regression test
4. Documentation update

---

## 10. VALIDATION STRATEGY

### 10.1 Accuracy Preservation Test

```python
# Test queries with known answers
test_cases = [
    {
        "file": "sales_data.csv",
        "query": "What is the total revenue?",
        "expected_contains": ["revenue", number_pattern]
    },
    {
        "file": "sales_data.csv", 
        "query": "Which region has highest sales?",
        "expected_contains": ["North|South|East|West"]
    }
]
```

### 10.2 Feature Parity Verification

| Feature | OLD Has | NEW Has | Reconstructed | Behavior Changed? |
|---------|---------|---------|---------------|-------------------|
| Structured analysis | ✓ | ✓ | TBD | TBD |
| RAG analysis | ✓ | ✓ | TBD | TBD |
| Multi-file | ✗ | ✓ | TBD | N/A (new) |
| Intelligent routing | ✗ | ✓ | TBD | N/A (new, optional) |
| CoT self-correction | ✗ | ✓ | TBD | N/A (new, optional) |

### 10.3 Success Criteria

1. **Accuracy**: Reconstructed system accuracy >= OLD version accuracy
2. **Features**: All NEW version features available (even if disabled by default)
3. **Stability**: No regressions in error handling
4. **Performance**: Response time within 2x of OLD version
5. **Testability**: All accuracy-affecting code paths have tests

---

## 11. STOP CONDITIONS

### 11.1 STOP and Seek Human Review If:

1. **Accuracy regression detected** - Any measurable accuracy drop vs baseline
2. **Conflicting invariants** - Two correct-looking implementations disagree
3. **Silent failures** - Code path that swallows errors without logging
4. **Global state mutation** - Discovery of hidden state affecting inference
5. **Prompt injection risk** - User input flows unescaped into prompts
6. **Model output mutation** - Any code that modifies LLM response before returning

### 11.2 Escalation Triggers

| Trigger | Action |
|---------|--------|
| >10% accuracy drop on any query category | STOP, document, propose alternatives |
| CoT parsing fails >20% of time | STOP, disable CoT, investigate |
| Routing sends simple queries to full_power >30% | STOP, recalibrate thresholds |
| Any test that passed in OLD fails in reconstructed | STOP, investigate before proceeding |

### 11.3 Safe Defaults

When uncertain, use these defaults:
- Use OLD version's direct inference path
- Disable intelligent routing
- Disable CoT self-correction
- Enable simple review step
- Log all decisions for debugging

---

## NEXT STEPS (Awaiting Human Approval)

**Before proceeding to Phase 1 implementation, please confirm:**

1. ✅ Do you approve this analysis and reconstruction plan?
2. ⚠️ Should I run both versions on sample data now to establish concrete baseline metrics?
3. ⚠️ Are there specific queries or data files you want prioritized in testing?
4. ⚠️ Do you have preferences on which features MUST be preserved vs which can be disabled?
5. ⚠️ Is there additional context about known bugs or accuracy issues not captured here?

---

## APPENDIX A: File-by-File Comparison

### A.1 analyze.py Differences

| Line Range | OLD | NEW | Impact |
|------------|-----|-----|--------|
| 1-20 | Basic imports, model | Extended imports, models, multi-file | LOW |
| 14-25 | Simple AnalyzeRequest | Extended with filenames, text_data | LOW (additive) |
| 50-120 | Basic analyze_query | Extended with text input handling | LOW |
| 120-200 | Error handling | Same structure, more detail | LOW |

### A.2 crew_manager.py Differences

| Line Range | OLD | NEW | Impact |
|------------|-----|-----|--------|
| 1-50 | Direct CrewAI imports | Delegated to model_initializer | LOW |
| 50-150 | Inline initialization | Lazy loading via properties | LOW |
| 150-350 | _create_agents() inline | Delegated to AgentFactory | LOW |
| 350-600 | Direct LLM analysis | Delegated to analysis_executor | **HIGH** |
| 600-750 | RAG analysis inline | Delegated to rag_handler | MEDIUM |

### A.3 NEW Files Not in OLD

- `agent_factory.py` - Agent creation (LOW risk, organization only)
- `model_initializer.py` - Model loading (LOW risk, organization only)
- `analysis_executor.py` - Analysis execution (**HIGH risk - contains routing logic**)
- `rag_handler.py` - RAG execution (MEDIUM risk)
- `intelligent_router.py` - Query routing (**HIGH risk**)
- `query_complexity_analyzer.py` - Complexity scoring (**HIGH risk**)
- `self_correction_engine.py` - CoT loop (**HIGH risk**)
- `cot_parser.py` - CoT extraction (MEDIUM risk)

---

*Document generated by AI forensic analysis. Human review required before implementation.*

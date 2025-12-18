# Nexus LLM Analytics: Recovery & Improvement Roadmap

**Date:** December 18, 2025  
**Based on Analysis by:** Antigravity Agent

---

## 🏗️ 1. Executive Summary

The project is currently in a state of **architectural drift**. A major refactor moved core logic from a monolithic `crew_manager.py` to modular components (`AgentFactory`, `AnalysisExecutor`), but the public API and tests were not fully updated to match. This has resulted in:
- **~46% Test Failure Rate:** Due to `AttributeError` and API mismatches.
- **Critical Security Risk:** Unsanitized SQL execution.
- **Silent Failures:** Model initialization errors are swallowed, returning empty results.

This roadmap prescribes a 3-phase recovery plan to stabilize the system, secure it, and then optimize it for your research goals.

---

## 🚨 Phase 1: Stabilization (Critical Fixes)
**Goal:** Restore system stability, ensure all tests pass, and fix "silent failures".

### 1.1 Fix "Code Drift" (API Mismatches)
**The Issue:** Legacy code calls `factory.create_agent("data_analyst")` but the new `AgentFactory` only has properties like `@property def data_analyst`.
**The Fix:** Add a simplified wrapper method to `AgentFactory` to restore backward compatibility.

**File:** `src/backend/agents/agent_factory.py`
```python
# Add this method to AgentFactory class
def create_agent(self, agent_type: str, **kwargs):
    """Legacy compatibility wrapper for agent creation."""
    agent_map = {
        'data_analyst': self.data_analyst,
        'rag_specialist': self.rag_specialist,
        'reviewer': self.reviewer,
        'visualizer': self.visualizer,
        'reporter': self.reporter
    }
    agent = agent_map.get(agent_type.lower())
    if not agent:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return agent
```

### 1.2 Fix "Silent Failure" in Model Initialization
**The Issue:** `ModelSelector.select_optimal_models()` returns a `tuple` `(primary, review, embedding)`, but usage code expects a `dict` or tries to call `.get()`.
**The Fix:** Update `model_initializer.py` to correctly unpack the tuple.

**File:** `src/backend/agents/model_initializer.py`
```python
# In _initialize_models method:
# OLD (Buggy):
# models = ModelSelector.select_optimal_models()
# self.primary_llm = models.get('primary')

# NEW (Fixed):
primary_model_name, review_model_name, embed_model_name = ModelSelector.select_optimal_models()
# Then use these names to initialize LLM objects
```

### 1.3 Fix API Route Consistency
**The Issue:** Tests expect standard REST paths (e.g., `/api/upload`), but `main.py` uses descriptive paths (`/upload-documents`).
**The Fix:** Standardize routes in `main.py`.

**File:** `src/backend/main.py`
```python
# Update router includes
app.include_router(analyze.router, prefix="/api/analyze", tags=["analyze"])
app.include_router(upload.router, prefix="/api/upload", tags=["upload"]) # Was /upload-documents
app.include_router(report.router, prefix="/api/report", tags=["report"]) # Was /generate-report
app.include_router(visualize.router, prefix="/api/visualize", tags=["visualize"])
```

---

## 🔒 Phase 2: Security (Must Do)
**Goal:** Close the critical SQL injection vulnerability before any deployment.

### 2.1 Patch SQL Injection
**The Issue:** `SQLAgent.execute()` runs raw strings passed from the user/LLM.
**The Fix:** Implement a validator that blocks DDL/DML (Drop, Alter, Insert) commands and only allows read-only operations.

**File:** `src/backend/plugins/sql_agent.py`
```python
import re

def validate_query(self, query: str) -> bool:
    # 1. Deny-list for destructive commands
    dangerous_patterns = [
        r"\bDROP\b", r"\bALTER\b", r"\bTRUNCATE\b", r"\bDELETE\b", 
        r"\bINSERT\b", r"\bUPDATE\b", r"\bGRANT\b", r"\bEXEC\b"
    ]
    query_upper = query.upper()
    for pattern in dangerous_patterns:
        if re.search(pattern, query_upper):
            raise SecurityError(f"Rejected unsafe SQL command: {pattern}")
            
    # 2. Allow-list (optional strict mode)
    if not query_upper.strip().startswith("SELECT") and not query_upper.strip().startswith("WITH"):
        raise SecurityError("Only SELECT queries are allowed.")
        
    return True
```

---

## ⚡ Phase 3: Optimization & Performance
**Goal:** Reduce latency for simple queries (currently too slow due to over-engineering).

### 3.1 Implement Hybrid Inference Routing
**The Strategy:** Do not send every query to the slow "Agent Crew". Use the `QueryComplexityAnalyzer` to decide.
- **Fast Lane (Complexity < 0.3):** Direct LLM call (`LLMClient.generate`). ~1-3s latency.
- **Slow Lane (Complexity >= 0.3):** Full Agent Crew loop. ~15-30s latency.

**Implementation Plan:**
Modify `AnalysisExecutor.execute()`:
1. Run `complexity = analyzer.analyze(query)`
2. `if complexity < 0.3: return fast_track_execution(query, data)`
3. `else: return crew_execution(query, data)`

---

## 🔬 Phase 4: Research & Innovation Features
**Goal:** Add novel features suitable for your Patent/Publication.

### 4.1 "Glass Box" Explainability (XAI)
**Feature:** Show the user *why* an answer was given.
- **How:** The CoT parser already extracts `[REASONING]` tags.
- **UI Update:** Add a "Reasoning" tab in the frontend results view that displays this extracted text.

### 4.2 Self-Correction Loop Metrics
**Feature:** Quantify the system's "Self-Healing" capability.
- **Metric:** `Correction Rate = (Number of Reviewer Rejections / Total Queries)`.
- **Use Case:** Show in your research paper that your "Self-Correction Engine" fixes X% of initial errors.

### 4.3 Adaptive "Prompt Optimization"
**Feature:** Use the feedback from the Reviewer Agent to permanently improve the System Prompt.
- **Concept:** If the Reviewer constantly says "You forgot to format numbers as currency", add "Always format money as currency" to the Data Analyst's system prompt automatically.

---

## 📋 Recommended Execution Order

1. **Apply Phase 1 Fixes** (Code Drift & Routes) - *Est: 2 hours*
2. **Apply Phase 2 Fixes** (SQL Security) - *Est: 1 hour*
3. **Verify Fixes** by running `pytest`.
4. **Apply Phase 3 Optimization** (Hybrid Routing) - *Est: 4 hours*
5. **Start Phase 4 Research Features** - *Est: Ongoing*

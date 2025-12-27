# 🎯 MASTER ROADMAP - Nexus LLM Analytics
> **Version:** FINAL 1.0  
> **Date:** December 27, 2025  
> **Authority:** This is the ONLY roadmap to follow. All other roadmaps are superseded.  
> **Total Duration:** 14 weeks  
> **Goal:** Full-fledged working system ready for research publication and patent filing

---

## ⚡ QUICK REFERENCE

### Start Commands
```powershell
# Backend
python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend
cd src/frontend; npm run dev

# Tests
pytest tests/ -v --ignore=tests/archive
```

### Runtime Models (Ollama - Used IN the Project)
| Model | Size | Purpose |
|-------|------|---------|
| `llama3.1:8b` | 4.9 GB | Primary analysis |
| `phi3:mini` | 2.2 GB | Fallback for low RAM |
| `tinyllama` | 637 MB | Lightweight tasks |
| `nomic-embed-text` | 274 MB | Vector embeddings |

### Development Models (VS Code Copilot)
| Task | Best Model |
|------|------------|
| Complex refactoring | Claude Opus 4.5 |
| New features | GPT-5.1-Codex-Max |
| Bug fixes | Claude Sonnet 4.5 |
| Documentation | Claude Sonnet 4 |
| Simple edits | Claude Haiku 4.5 |

---

## 📅 MASTER TIMELINE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        14-WEEK EXECUTION PLAN                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 0: CLEANUP & STABILIZATION                    [Week 1-2]            │
│  ├── 0.1 Archive unused files                        ✅ DONE               │
│  ├── 0.2 Fix CoT parser fragility                    ✅ DONE               │
│  ├── 0.3 Fix dynamic planner JSON handling           ⬜ TODO               │
│  ├── 0.4 Remove self-learning false claim            ⬜ TODO               │
│  ├── 0.5 Fix data_optimizer business bias            ⬜ TODO               │
│  ├── 0.6 Run all tests, establish baseline           ⬜ TODO               │
│  ├── 0.7 Fix Two Friends Model (automated validator) ✅ DONE               │
│  ├── 0.8 Create QueryOrchestrator (unify 3 tracks)   ✅ DONE               │
│  ├── 0.9 Integrate orchestrator with agents          ⬜ TODO               │
│  ├── 0.10 Test unified decision system               ⬜ TODO               │
│  └── 0.11 ✅ VERIFIED: Two Friends Model Working     ✅ VERIFIED           │
│       └── Generator-Critic communication PROVEN                            │
│       └── Enterprise-level tests: 3/3 passed                               │
│       └── Average quality score: 78%                                       │
│                                                                             │
│  PHASE 1: UNIFIED INTELLIGENCE (3-TRACK INTEGRATION) [Week 3-4]            │
│  ├── 1.1 Track 1: Complexity → Model Selection       ⬜ TODO               │
│  │   └── Simple→tinyllama, Medium→phi3, Complex→llama3.1                   │
│  ├── 1.2 Track 2: Query Type → Execution Method      ⬜ TODO               │
│  │   └── Computation→Code Gen, Conversational→Direct LLM                  │
│  ├── 1.3 Track 3: Two Friends Activation Rules       ⬜ TODO               │
│  │   └── Skip for simple, Optional for medium, Mandatory for complex      │
│  ├── 1.4 Wire QueryOrchestrator to data_analyst      ⬜ TODO               │
│  └── 1.5 End-to-end integration testing              ⬜ TODO               │
│                                                                             │
│  PHASE 2: LLM CODE GENERATION                        [Week 5-7]            │
│  ├── 2.1 Code generation prompt templates            ⬜ TODO               │
│  ├── 2.2 Code validation layer (syntax, security)    ⬜ TODO               │
│  ├── 2.3 Sandbox hardening & testing                 ⬜ TODO               │
│  ├── 2.4 Result interpretation prompts               ⬜ TODO               │
│  ├── 2.5 Integration with existing agents            ⬜ TODO               │
│  └── 2.6 Error recovery & retry logic                ⬜ TODO               │
│                                                                             │
│  PHASE 3: CAPABILITY COMPLETION                      [Week 8-10]           │
│  ├── 3.1 RAG semantic chunking                       ⬜ TODO               │
│  ├── 3.2 Hybrid search (vector + keyword)            ⬜ TODO               │
│  ├── 3.3 Citation tracking in responses              ⬜ TODO               │
│  ├── 3.4 Visualization execution                     ⬜ TODO               │
│  ├── 3.5 Add scientific file formats                 ⬜ TODO               │
│  ├── 3.6 Fix bare exception handlers                 ⬜ TODO               │
│  ├── 3.7 Enhance cache mechanism (semantic)          ⬜ TODO               │
│  ├── 3.8 Add Prometheus metrics                      ⬜ TODO               │
│  └── 3.9 Add structured logging                      ⬜ TODO               │
│                                                                             │
│  PHASE 4: RESEARCH READINESS                         [Week 11-13]          │
│  ├── 4.1 Create benchmark dataset (150+ queries)     ⬜ TODO               │
│  ├── 4.2 Implement evaluation metrics                ⬜ TODO               │
│  ├── 4.3 Run baseline comparisons                    ⬜ TODO               │
│  ├── 4.4 Complete ablation studies                   ⬜ TODO               │
│  ├── 4.5 Add test coverage measurement               ⬜ TODO               │
│  ├── 4.6 Set up CI/CD pipeline                       ⬜ TODO               │
│  └── 4.7 Write research paper                        ⬜ TODO               │
│                                                                             │
│  PHASE 5: PATENT & POLISH                            [Week 14]             │
│  ├── 5.1 Prior art search                            ⬜ TODO               │
│  ├── 5.2 Document patent claims                      ⬜ TODO               │
│  └── 5.3 Final polish & documentation                ⬜ TODO               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 UNIFIED INTELLIGENCE SYSTEM (THE 3-TRACK INNOVATION)

This is the **core research contribution** - a unified decision system combining three tracks:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUERY ORCHESTRATOR                                  │
│                    (Master Decision Maker)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │  ANALYZE COMPLEXITY     │
                    │  (0.0 - 1.0 score)      │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
   [0.0 - 0.3]             [0.3 - 0.7]              [0.7 - 1.0]
    SIMPLE                   MEDIUM                  COMPLEX
        │                        │                        │
   ┌────┴────┐             ┌────┴────┐             ┌────┴────┐
   │ MODEL:  │             │ MODEL:  │             │ MODEL:  │
   │tinyllama│             │phi3:mini│             │llama3.1 │
   └────┬────┘             └────┬────┘             └────┬────┘
        │                        │                        │
   ┌────┴────┐             ┌────┴────┐             ┌────┴────┐
   │ METHOD: │             │ METHOD: │             │ METHOD: │
   │Direct   │             │Auto-pick│             │Code Gen │
   │LLM      │             │         │             │         │
   └────┬────┘             └────┬────┘             └────┬────┘
        │                        │                        │
   ┌────┴────┐             ┌────┴────┐             ┌────┴────┐
   │ REVIEW: │             │ REVIEW: │             │ REVIEW: │
   │Skip     │             │Optional │             │Mandatory│
   │(fast)   │             │         │             │Two Friends│
   └─────────┘             └─────────┘             └─────────┘
```

### Track 1: Complexity → Model Selection (RAM-Aware)
| Complexity | Model | RAM | Use Case |
|------------|-------|-----|----------|
| < 0.3 | tinyllama | 637 MB | Simple lookups, definitions |
| 0.3-0.7 | phi3:mini | 2.2 GB | Medium analysis, filtering |
| > 0.7 | llama3.1:8b | 4.9 GB | Complex reasoning, correlations |

### Track 2: Query Type → Execution Method
| Query Type | Method | Reason |
|------------|--------|--------|
| Calculations, aggregations | Code Generation | Python math > LLM approximation |
| Filtering, grouping | Code Generation | Exact results |
| Explanations, "why" questions | Direct LLM | Natural language task |
| Definitions, "what is" | Direct LLM | No computation needed |

### Track 3: Two Friends Model Activation
| Scenario | Apply Review? | Reason |
|----------|---------------|--------|
| Simple query (< 0.3) | ❌ Skip | Overhead not worth it |
| Medium query (0.3-0.7) | ⚠️ Optional | User preference |
| Complex query (> 0.7) | ✅ Mandatory | High stakes |
| Code generation used | ✅ Mandatory | Validate generated code |
| First attempt failed | ✅ Mandatory | Error recovery |

---

## ❌ OUT OF SCOPE (DO NOT IMPLEMENT)

These items are explicitly **NOT** part of this project:

| Item | Reason |
|------|--------|
| Authentication (JWT/OAuth) | Not required for project goals |
| API Key management | Not required |
| User management | Not required |
| Multi-tenancy | Not required |
| WebSocket real-time updates | Archived, polling sufficient |
| CrewAI integration | Abandoned, replaced with plugins |

---

## ✅ WHAT'S ALREADY WORKING (Don't Break These)

| Feature | Status | Main File |
|---------|--------|-----------|
| 10 Plugin Agents | ✅ Working | `plugin_system.py` |
| Agent Routing (100% accuracy) | ✅ Working | All agent files |
| Self-Correction Loop (CoT) | ⚠️ Fragile | `self_correction_engine.py` |
| RAG with ChromaDB | ✅ Working | `rag_agent.py` |
| Sandbox Code Execution | ✅ Working | `sandbox.py` |
| Dynamic Model Selection | ✅ Working | `model_selector.py` |
| Circuit Breaker | ✅ Working | `circuit_breaker.py` |
| Query Complexity Analysis | ✅ Working | `query_complexity_analyzer.py` |
| Frontend UI | ✅ Working | `src/frontend/` |

---

# PHASE 0: CLEANUP & STABILIZATION
**Duration:** 2 weeks | **Priority:** CRITICAL

## Week 1: Dead Code & Fixes

### Task 0.1: Archive Unused Files ✅ DONE
Files already moved to `archive/removed_v1.1/`:
- `intelligent_query_engine.py`
- `optimized_llm_client.py`
- `websocket_manager.py`

### Task 0.2: Fix CoT Parser Fragility
**File:** `src/backend/core/cot_parser.py`  
**Effort:** 1 day  
**Priority:** HIGH

**Problem:** Parser fails if LLM produces tag variations like `<REASONING>` instead of `[REASONING]`

**Solution:** Add fallback parsing strategies:

```python
def parse(self, response: str) -> ParsedCoT:
    """Enhanced parsing with fallback strategies"""
    
    # Strategy 1: Exact match (current)
    result = self._parse_exact(response)
    if result.is_valid:
        return result
    
    # Strategy 2: Fuzzy tag matching
    result = self._parse_fuzzy(response)
    if result.is_valid:
        return result
    
    # Strategy 3: Fallback - return entire response as output
    return ParsedCoT(
        reasoning="Unable to extract structured reasoning",
        output=response.strip(),
        is_valid=False,
        error_message="All parsing strategies failed",
        raw_response=response
    )

def _parse_fuzzy(self, response: str) -> ParsedCoT:
    """Match common variations of tags"""
    import re
    
    fuzzy_patterns = [
        (r'\[REASONING\](.*?)\[/REASONING\]', r'\[OUTPUT\](.*?)\[/OUTPUT\]'),
        (r'<reasoning>(.*?)</reasoning>', r'<output>(.*?)</output>'),
        (r'<REASONING>(.*?)</REASONING>', r'<OUTPUT>(.*?)</OUTPUT>'),
        (r'REASONING:(.*?)OUTPUT:', r'OUTPUT:(.*?)$'),
        (r'Reasoning:(.*?)Answer:', r'Answer:(.*?)$'),
    ]
    
    for reasoning_pattern, output_pattern in fuzzy_patterns:
        reasoning_match = re.search(reasoning_pattern, response, re.DOTALL | re.IGNORECASE)
        output_match = re.search(output_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if reasoning_match and output_match:
            return ParsedCoT(
                reasoning=reasoning_match.group(1).strip(),
                output=output_match.group(1).strip(),
                is_valid=True,
                raw_response=response
            )
    
    return ParsedCoT(reasoning="", output="", is_valid=False, raw_response=response)
```

**Verification:**
```python
# Test cases
test_inputs = [
    "[REASONING]Think step by step[/REASONING][OUTPUT]Answer here[/OUTPUT]",
    "<reasoning>My thought process</reasoning><output>Final answer</output>",
    "REASONING: Let me analyze OUTPUT: The result is 42",
]
```

---

### Task 0.3: Fix Dynamic Planner JSON Handling
**File:** `src/backend/core/dynamic_planner.py`  
**Effort:** 2 hours  
**Priority:** MEDIUM

**Problem:** LLM sometimes produces invalid JSON, causing crashes.

**Solution:** Add JSON repair and fallback:

```python
import json
import re

def _parse_plan(self, llm_output: str) -> AnalysisPlan:
    """Parse with JSON repair fallback"""
    try:
        # Clean markdown code blocks
        cleaned = re.sub(r'```json\s*', '', llm_output)
        cleaned = re.sub(r'```\s*', '', cleaned).strip()
        
        # Try direct parse
        plan_dict = json.loads(cleaned)
        
    except json.JSONDecodeError:
        # Try to extract JSON object from text
        json_match = re.search(r'\{[\s\S]*\}', cleaned)
        if json_match:
            try:
                plan_dict = json.loads(json_match.group())
            except json.JSONDecodeError:
                return self._fallback_plan(llm_output)
        else:
            return self._fallback_plan(llm_output)
    
    return self._dict_to_plan(plan_dict)

def _fallback_plan(self, raw_output: str) -> AnalysisPlan:
    """Return simple fallback when JSON parsing fails"""
    return AnalysisPlan(
        domain="General",
        summary="Analysis based on LLM response",
        steps=[AnalysisStep(1, "Analyze data", "python_pandas", "Direct analysis")],
        confidence=0.3
    )
```

---

### Task 0.4: Remove Self-Learning False Claim
**File:** `src/backend/plugins/data_analyst_agent.py`  
**Effort:** 2 hours  
**Priority:** HIGH (research integrity)

**Problem:** `_learn_from_correction()` is an empty stub but documentation claims "self-learning".

**Solution:** Remove the stub and update documentation:

```python
# DELETE this function:
# def _learn_from_correction(self, first_cot, final_cot, query):
#     pass  # Empty stub

# DELETE the call on line 176 (or wherever it's called)
```

**Update docs:**
- Remove "self-learning" claims from `methologies.md`
- Update `PROJECT_UNDERSTANDING.md` to mark as "Future Work"

---

## Week 2: Data-Agnostic Fixes

### Task 0.5: Fix data_optimizer Business Bias
**File:** `src/backend/utils/data_optimizer.py`  
**Effort:** 4 hours  
**Priority:** HIGH (research claim)

**Problem:** Hardcoded business keywords (`revenue`, `customer`, `product`) create domain bias.

**Solution:** Replace domain-specific keywords with generic patterns:

```python
# BEFORE (lines 639-747) - REMOVE:
# if 'customer' in col_lower:
#     ranking_cols.insert(0, col)
# elif 'product' in col_lower:
#     ranking_cols.insert(1 if len(ranking_cols) > 0 else 0, col)

# AFTER - Generic importance detection:
def _detect_important_columns(self, df: pd.DataFrame) -> List[str]:
    """Domain-agnostic column importance detection"""
    important_cols = []
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Generic patterns (not domain-specific)
        is_id_column = '_id' in col_lower or col_lower.endswith('id')
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        has_high_variance = df[col].std() > df[col].mean() * 0.1 if is_numeric else False
        has_unique_values = df[col].nunique() / len(df) > 0.8
        
        # Importance based on data characteristics, not vocabulary
        if is_numeric and has_high_variance:
            important_cols.append(col)
        elif not has_unique_values and not is_id_column:
            important_cols.append(col)  # Good for grouping
    
    return important_cols
```

---

### Task 0.6: Run All Tests & Establish Baseline
**Effort:** 2 hours  
**Priority:** HIGH

```powershell
# Run all tests
pytest tests/ -v --ignore=tests/archive

# Check for errors
python -c "from src.backend.main import app; print('Backend imports OK')"

# Test health endpoint
python scripts/health_check.py
```

**Document:**
- Number of passing tests
- Any failures
- Current test coverage

---

### Task 0.7: Fix Two Friends Model Critic ✅ DONE (Dec 27, 2025)
**Files:** 
- `src/backend/prompts/cot_critic_prompt.txt` (rewritten)
- `src/backend/core/automated_validation.py` (NEW - 458 lines)
- `src/backend/core/cot_parser.py` (added feedback property)
- `src/backend/core/self_correction_engine.py` (fixed CriticFeedback creation)
- `tests/test_critic_catches_errors.py` (NEW - error detection test)
- `tests/test_two_friends_integration.py` (NEW - full integration test)

**Problem:** Validation test revealed critic approves everything (0% improvement rate)  
**Root Cause:** LLM critic (phi3:mini) too lenient + no deterministic checks + bug in CriticFeedback init

**Solution Implemented:** Hybrid validation approach
1. **AutomatedValidator** - Rule-based pre-checks before LLM critic
2. **Stricter Critic Prompt** - With explicit rejection examples
3. **Integrated into SelfCorrectionEngine** - Step 2.5
4. **Fixed CriticFeedback** - Added feedback property, fixed field names
5. **Domain-Agnostic Validation** - Works with any domain (healthcare, finance, education, etc.)

**Validation Checks Now Implemented:**
- ✅ Arithmetic errors (verifies A+B=C, A×B=C dynamically)
- ✅ Logic inversions (< vs > confusion)
- ✅ Formula errors (profit margin = profit/revenue, not revenue/cost)
- ✅ Percentage format (0.75 → 75%)
- ✅ Causation vs correlation confusion
- ✅ Time period errors (Q4 = Oct/Nov/Dec)
- ✅ Missing filter/group operations

**Test Results:**
```
✅ Error Detection: 8/8 (100.0%)
- 6/8 caught by AutomatedValidator (deterministic, fast)
- 2/8 caught by LLM Critic (semantic understanding)

✅ Integration: 5/5 (100.0%)
- AutomatedValidator integration
- CriticFeedback generation  
- SelfCorrectionEngine setup
- Domain-agnostic validation (healthcare/finance/education/time)
- Feedback loop simulation
```

**Novel Research Claim Validated:**
The system is now **domain-agnostic and data-agnostic** - it can catch errors across:
- Healthcare data (patient recovery rates)
- Finance data (profit margins)
- Education data (pass rates)
- Time series data (Q4 calculations)
- Any domain with similar patterns

---

### Task 0.8: Create Query Orchestrator ✅ DONE (Dec 27, 2025)
**New File:** `src/backend/core/query_orchestrator.py`  
**Effort:** 3 days  
**Priority:** HIGH (Integration Layer)

**Files Created:**
- `src/backend/core/query_orchestrator.py` (365 lines)
- `docs/UNIFIED_ARCHITECTURE.md` (design document)

**Three Tracks Unified:**
1. ✅ Complexity → Model selection (tinyllama/phi3/llama3.1)
2. ✅ Query type → Execution method (code gen vs direct LLM)
3. ✅ Complexity + method → Two Friends review decision

**Key Classes:**
- `ExecutionMethod` enum: DIRECT_LLM, CODE_GENERATION
- `ReviewLevel` enum: NONE, OPTIONAL, MANDATORY
- `ExecutionPlan` dataclass: model + method + review + complexity
- `QueryOrchestrator.create_execution_plan()` - Main entry point

**Decision Rules:**
| Complexity | Model | Method | Review |
|------------|-------|--------|--------|
| < 0.3 | tinyllama | Direct LLM | None |
| 0.3-0.7 | phi3:mini | Auto | Optional |
| > 0.7 | llama3.1:8b | Auto | Mandatory |
| Any + Code Gen | - | Code Gen | Mandatory |

**Status:** Created but NOT integrated with agents yet (see Task 0.9)

---

### Task 0.9: Integrate Orchestrator with Agents ⏳ TODO
**File:** `src/backend/plugins/data_analyst_agent.py`  
**Effort:** 2 days  
**Priority:** HIGH

**Change:** Wire QueryOrchestrator into agent execution flow

```python
class DataAnalystAgent:
    def __init__(self, ...):
        self.orchestrator = QueryOrchestrator(
            self.complexity_analyzer,
            config
        )
    
    async def execute(self, query: str, data: Any = None, **kwargs):
        # NEW: Unified decision making
        plan = self.orchestrator.create_execution_plan(query, data)
        
        logger.info(f"Plan: {plan.model} | {plan.execution_method.value} | {plan.review_level.value}")
        
        # Execute based on plan
        if plan.execution_method == ExecutionMethod.CODE_GENERATION:
            result = await self._execute_with_code_generation(query, data, plan)
        else:
            result = await self._execute_direct(query, data, plan)
        
        return result
```

---

# PHASE 1: CORE ENHANCEMENT

## Week 3: Cache & Monitoring

### Task 1.1: Enhance Cache Mechanism
**File:** `src/backend/core/advanced_cache.py`  
**Effort:** 2 days  
**Priority:** HIGH

**Enhancement:** Add semantic similarity caching (cache similar queries):

```python
import hashlib
from typing import Optional, Dict
import time

class EnhancedCache:
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
    
    def get_cache_key(self, query: str, data_hash: str, model: str) -> str:
        """Generate deterministic cache key"""
        content = f"{query.lower().strip()}:{data_hash}:{model}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Dict]:
        """Get cached value if not expired"""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() < entry['expires']:
                self.stats['hits'] += 1
                return entry['value']
            else:
                del self.cache[key]
        self.stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Dict, ttl: Optional[int] = None):
        """Set cache value with TTL"""
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = {
            'value': value,
            'expires': time.time() + (ttl or self.default_ttl),
            'created': time.time()
        }
    
    def _evict_oldest(self):
        """Remove oldest entry"""
        if self.cache:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['created'])
            del self.cache[oldest_key]
            self.stats['evictions'] += 1
    
    def get_stats(self) -> Dict:
        """Return cache statistics"""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total if total > 0 else 0
        return {**self.stats, 'hit_rate': f"{hit_rate:.2%}", 'size': len(self.cache)}
```

---

### Task 1.2: Add Prometheus Metrics
**New File:** `src/backend/core/metrics.py`  
**Effort:** 2 days  
**Priority:** MEDIUM

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from functools import wraps
import time

# Define metrics
REQUEST_COUNT = Counter('nexus_requests_total', 'Total requests', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('nexus_request_latency_seconds', 'Request latency', ['endpoint'])
ACTIVE_ANALYSES = Gauge('nexus_active_analyses', 'Currently running analyses')
AGENT_USAGE = Counter('nexus_agent_usage_total', 'Agent usage count', ['agent_name'])
LLM_CALLS = Counter('nexus_llm_calls_total', 'LLM API calls', ['model', 'status'])
CACHE_HITS = Counter('nexus_cache_hits_total', 'Cache hits')
CACHE_MISSES = Counter('nexus_cache_misses_total', 'Cache misses')

def track_request(endpoint: str):
    """Decorator to track request metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            ACTIVE_ANALYSES.inc()
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                REQUEST_COUNT.labels(endpoint=endpoint, status='success').inc()
                return result
            except Exception as e:
                REQUEST_COUNT.labels(endpoint=endpoint, status='error').inc()
                raise
            finally:
                ACTIVE_ANALYSES.dec()
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.time() - start)
        return wrapper
    return decorator
```

**Add to `main.py`:**
```python
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

---

## Week 4: Testing Infrastructure

### Task 1.3: Add Structured Logging
**File:** `src/backend/core/enhanced_logging.py`  
**Effort:** 1 day

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)
        return json.dumps(log_entry)

def setup_logging(level: str = "INFO"):
    """Configure JSON structured logging"""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))
    root_logger.addHandler(handler)
    
    return root_logger
```

---

### Task 1.4: Add Test Coverage Measurement
**Update `pyproject.toml`:**

```toml
[tool.pytest.ini_options]
addopts = "-v --tb=short --cov=src/backend --cov-report=html --cov-report=term"
testpaths = ["tests"]

[tool.coverage.run]
source = ["src/backend"]
omit = ["*/tests/*", "*/__pycache__/*", "*/archive/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
]
show_missing = true
```

**Run coverage:**
```powershell
pip install pytest-cov
pytest tests/ --cov=src/backend --cov-report=html
# Open htmlcov/index.html to view report
```

---

### Task 1.5: Set up CI/CD Pipeline
**New File:** `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-asyncio
          
      - name: Run tests
        run: pytest tests/ -v --cov=src/backend --cov-report=xml --ignore=tests/archive
        
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

---

# PHASE 2: LLM CODE GENERATION
**Duration:** 3 weeks | **Priority:** HIGH (Key Innovation)

This phase adds the ability for LLMs to **generate executable Python code** for data analysis instead of doing analysis directly. This provides:
- ✅ **Verifiable Results** - Code can be reviewed before execution
- ✅ **Reproducible Analysis** - Same code = same output
- ✅ **Accurate Computations** - Python math, not LLM approximation
- ✅ **Debugging Capability** - Fix code, not LLM prompts

## Week 5: Code Generation Templates

### Task 2.1: Create Code Generation Prompts
**New File:** `src/backend/prompts/code_generation_prompt.txt`

```text
You are a Python data analysis expert. Generate executable Python code to answer the user's question.

DATA CONTEXT:
{data_preview}

COLUMNS AVAILABLE:
{columns}

USER QUESTION:
{query}

REQUIREMENTS:
1. Use pandas for data manipulation
2. Store the final answer in a variable called `result`
3. If visualization needed, use plotly and store figure in `fig`
4. Handle missing values appropriately
5. Include comments explaining each step

OUTPUT FORMAT:
```python
# Your code here
result = ...  # Final answer
```

Generate ONLY executable Python code. No explanations outside code blocks.
```

**New File:** `src/backend/core/code_generator.py`

```python
from typing import Dict, Any, Optional
from dataclasses import dataclass
import re

@dataclass
class GeneratedCode:
    code: str
    language: str = "python"
    is_valid: bool = True
    error_message: Optional[str] = None

class CodeGenerator:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.prompt_template = self._load_prompt()
    
    def _load_prompt(self) -> str:
        with open("src/backend/prompts/code_generation_prompt.txt") as f:
            return f.read()
    
    def generate(self, query: str, data_preview: str, columns: list) -> GeneratedCode:
        """Generate Python code for the query"""
        prompt = self.prompt_template.format(
            data_preview=data_preview,
            columns=", ".join(columns),
            query=query
        )
        
        response = self.llm.generate(prompt)
        code = self._extract_code(response)
        
        if not code:
            return GeneratedCode(
                code="",
                is_valid=False,
                error_message="No code block found in LLM response"
            )
        
        return GeneratedCode(code=code, is_valid=True)
    
    def _extract_code(self, response: str) -> str:
        """Extract code from markdown code blocks"""
        # Match ```python ... ``` blocks
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Fallback: try ``` ... ``` without language
        pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        return matches[0].strip() if matches else ""
```

---

### Task 2.2: Code Validation Layer
**New File:** `src/backend/core/code_validator.py`  
**Effort:** 3 days

```python
import ast
import re
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_code: str

class CodeValidator:
    # Dangerous patterns that should NEVER be executed
    FORBIDDEN_PATTERNS = [
        r'\bimport\s+os\b',
        r'\bimport\s+subprocess\b',
        r'\bimport\s+sys\b',
        r'\bopen\s*\(',
        r'\bexec\s*\(',
        r'\beval\s*\(',
        r'\b__import__\s*\(',
        r'\bcompile\s*\(',
        r'\bgetattr\s*\(',
        r'\bsetattr\s*\(',
        r'\bdelattr\s*\(',
        r'\bglobals\s*\(',
        r'\blocals\s*\(',
        r'\bvars\s*\(',
        r'\bdir\s*\(',
        r'\.read\s*\(',
        r'\.write\s*\(',
        r'subprocess',
        r'shutil',
        r'socket',
        r'requests\.',
        r'urllib',
    ]
    
    # Allowed imports
    ALLOWED_IMPORTS = [
        'pandas', 'pd',
        'numpy', 'np',
        'plotly', 'plotly.express', 'px', 'plotly.graph_objects', 'go',
        'math',
        'statistics',
        'datetime', 'timedelta',
        're',
        'json',
        'collections',
    ]
    
    def validate(self, code: str) -> ValidationResult:
        """Validate generated code for safety and syntax"""
        errors = []
        warnings = []
        
        # Check syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e.msg} at line {e.lineno}")
            return ValidationResult(False, errors, warnings, code)
        
        # Check forbidden patterns
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                errors.append(f"Forbidden pattern detected: {pattern}")
        
        # Check imports
        import_errors = self._validate_imports(code)
        errors.extend(import_errors)
        
        # Check for infinite loops (basic heuristic)
        if 'while True' in code and 'break' not in code:
            warnings.append("Potential infinite loop detected")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, code)
    
    def _validate_imports(self, code: str) -> List[str]:
        """Check that only allowed modules are imported"""
        errors = []
        import_pattern = r'^(?:from\s+(\S+)|import\s+(\S+))'
        
        for line in code.split('\n'):
            match = re.match(import_pattern, line.strip())
            if match:
                module = match.group(1) or match.group(2)
                module_base = module.split('.')[0]
                
                if module_base not in self.ALLOWED_IMPORTS:
                    errors.append(f"Import not allowed: {module}")
        
        return errors
```

---

## Week 6: Sandbox Hardening

### Task 2.3: Sandbox Hardening & Testing
**File:** `src/backend/core/sandbox.py`  
**Effort:** 4 days

**Add these safety measures:**

```python
import resource
import signal
from contextlib import contextmanager
from typing import Dict, Any

class SecureSandbox:
    def __init__(self, timeout: int = 30, max_memory_mb: int = 512):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
    
    @contextmanager
    def resource_limits(self):
        """Apply resource limits during execution"""
        # Set memory limit (Linux only)
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(
                resource.RLIMIT_AS, 
                (self.max_memory_mb * 1024 * 1024, hard)
            )
        except (ValueError, resource.error):
            pass  # Windows doesn't support this
        
        # Set timeout
        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout)
        
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def execute(self, code: str, namespace: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute code in sandboxed environment"""
        if namespace is None:
            namespace = {}
        
        # Add safe builtins
        safe_builtins = {
            'abs': abs, 'all': all, 'any': any, 'bool': bool,
            'dict': dict, 'enumerate': enumerate, 'filter': filter,
            'float': float, 'int': int, 'len': len, 'list': list,
            'map': map, 'max': max, 'min': min, 'print': print,
            'range': range, 'round': round, 'set': set, 'sorted': sorted,
            'str': str, 'sum': sum, 'tuple': tuple, 'zip': zip,
            'True': True, 'False': False, 'None': None,
        }
        
        # Create execution namespace
        exec_namespace = {
            '__builtins__': safe_builtins,
            **namespace
        }
        
        try:
            with self.resource_limits():
                exec(code, exec_namespace)
            
            return {
                'success': True,
                'result': exec_namespace.get('result'),
                'fig': exec_namespace.get('fig'),
                'namespace': {k: v for k, v in exec_namespace.items() 
                            if not k.startswith('_')}
            }
        except TimeoutError as e:
            return {'success': False, 'error': str(e)}
        except MemoryError:
            return {'success': False, 'error': 'Memory limit exceeded'}
        except Exception as e:
            return {'success': False, 'error': f"{type(e).__name__}: {str(e)}"}
```

---

## Week 7: Integration

### Task 2.4: Result Interpretation Prompts
**New File:** `src/backend/prompts/interpret_result_prompt.txt`

```text
You are a data analysis expert. Interpret the following code execution result and provide a clear, natural language explanation.

ORIGINAL QUESTION:
{query}

CODE EXECUTED:
```python
{code}
```

RESULT:
{result}

Provide a clear, concise interpretation of this result that directly answers the user's question.
Focus on:
1. The key findings
2. Any notable patterns or insights
3. Actionable recommendations if applicable

Keep your response focused and avoid technical jargon.
```

---

### Task 2.5: Integration with Existing Agents
**File:** `src/backend/plugins/data_analyst_agent.py`  
**Effort:** 3 days

```python
# Add code generation path to execute method:

async def execute(self, query: str, data: Any = None, **kwargs) -> Dict[str, Any]:
    """Execute with code generation pipeline"""
    
    # Determine if code generation is appropriate
    if self._should_use_code_generation(query, data):
        return await self._execute_with_code_generation(query, data, **kwargs)
    else:
        return await self._execute_direct(query, data, **kwargs)

def _should_use_code_generation(self, query: str, data: Any) -> bool:
    """Determine if query benefits from code generation"""
    code_gen_keywords = [
        'calculate', 'compute', 'sum', 'average', 'mean', 'total',
        'count', 'group by', 'aggregate', 'filter', 'sort', 'rank',
        'correlation', 'percentage', 'ratio', 'compare'
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in code_gen_keywords) and data is not None

async def _execute_with_code_generation(self, query: str, data: Any, **kwargs):
    """Code generation pipeline"""
    from ..core.code_generator import CodeGenerator
    from ..core.code_validator import CodeValidator
    from ..core.sandbox import SecureSandbox
    
    # 1. Generate code
    generator = CodeGenerator(self.llm)
    generated = generator.generate(query, self._get_data_preview(data), list(data.columns))
    
    if not generated.is_valid:
        return {'success': False, 'error': generated.error_message}
    
    # 2. Validate code
    validator = CodeValidator()
    validation = validator.validate(generated.code)
    
    if not validation.is_valid:
        return {'success': False, 'error': f"Code validation failed: {validation.errors}"}
    
    # 3. Execute in sandbox
    sandbox = SecureSandbox(timeout=30)
    result = sandbox.execute(generated.code, {'df': data, 'pd': pd, 'np': np})
    
    if not result['success']:
        return {'success': False, 'error': result['error']}
    
    # 4. Interpret result
    interpretation = await self._interpret_result(query, generated.code, result['result'])
    
    return {
        'success': True,
        'result': interpretation,
        'code': generated.code,
        'raw_result': result['result'],
        'method': 'code_generation'
    }
```

---

### Task 2.6: Error Recovery & Retry Logic
**Effort:** 2 days

```python
async def _execute_with_code_generation_retry(self, query: str, data: Any, max_retries: int = 2):
    """Execute with automatic retry on failure"""
    
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            result = await self._execute_with_code_generation(query, data)
            
            if result['success']:
                return result
            
            # If code generation failed, try to fix based on error
            if attempt < max_retries:
                query = self._augment_query_with_error(query, result['error'])
                last_error = result['error']
                continue
                
        except Exception as e:
            last_error = str(e)
    
    # All retries failed, fall back to direct LLM analysis
    return await self._execute_direct(query, data, fallback_reason=last_error)

def _augment_query_with_error(self, query: str, error: str) -> str:
    """Add error context to query for retry"""
    return f"{query}\n\nNote: Previous attempt failed with: {error}. Please fix the issue."
```

---

# PHASE 3: CAPABILITY COMPLETION
**Duration:** 3 weeks | **Priority:** MEDIUM

## Week 8-9: RAG Enhancement

### Task 3.1: Semantic Chunking
**File:** `src/backend/core/document_indexer.py`

```python
class SemanticChunker:
    """Split documents at semantic boundaries (paragraphs, sections)"""
    
    def __init__(self, max_chunk_size: int = 500, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """Split text into semantic chunks"""
        # Split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_words = len(para.split())
            
            if current_size + para_words > self.max_chunk_size and current_size >= self.min_chunk_size:
                chunks.append({
                    'text': '\n\n'.join(current_chunk),
                    'word_count': current_size,
                    'type': 'semantic'
                })
                current_chunk = [para]
                current_size = para_words
            else:
                current_chunk.append(para)
                current_size += para_words
        
        if current_chunk:
            chunks.append({
                'text': '\n\n'.join(current_chunk),
                'word_count': current_size,
                'type': 'semantic'
            })
        
        return chunks
```

---

### Task 3.2: Hybrid Search (Vector + Keyword)
**File:** `src/backend/core/chromadb_client.py`

```python
def hybrid_query(self, query_text: str, n_results: int = 5) -> Dict:
    """Combine vector similarity with keyword matching"""
    
    # Vector search (get more for re-ranking)
    vector_results = self.collection.query(
        query_texts=[query_text],
        n_results=n_results * 2
    )
    
    # Extract keywords from query
    keywords = self._extract_keywords(query_text)
    
    # Re-rank by keyword overlap
    scored_results = []
    for i, doc in enumerate(vector_results['documents'][0]):
        vector_score = 1.0 / (1 + vector_results['distances'][0][i])
        keyword_score = self._keyword_overlap(doc, keywords)
        combined_score = 0.7 * vector_score + 0.3 * keyword_score
        scored_results.append((doc, combined_score, vector_results['metadatas'][0][i]))
    
    # Sort by combined score
    scored_results.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'documents': [[r[0] for r in scored_results[:n_results]]],
        'scores': [[r[1] for r in scored_results[:n_results]]],
        'metadatas': [[r[2] for r in scored_results[:n_results]]]
    }

def _extract_keywords(self, text: str) -> set:
    """Extract significant keywords from text"""
    import re
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out'}
    return set(words) - stopwords

def _keyword_overlap(self, doc: str, keywords: set) -> float:
    """Calculate keyword overlap score"""
    doc_words = set(doc.lower().split())
    overlap = len(keywords & doc_words)
    return overlap / len(keywords) if keywords else 0
```

---

### Task 3.3: Citation Tracking
```python
@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float

def generate_with_citations(self, query: str, context_chunks: List[Dict]) -> RAGResponse:
    """Generate answer with source citations"""
    
    # Format context with citation markers
    formatted_context = ""
    for i, chunk in enumerate(context_chunks):
        formatted_context += f"[Source {i+1}]: {chunk['text']}\n\n"
    
    prompt = f"""Answer the question using the provided sources. Cite sources using [Source N] format.

Question: {query}

Sources:
{formatted_context}

Answer:"""
    
    answer = self.llm.generate(prompt)
    
    return RAGResponse(
        answer=answer,
        sources=[{
            'id': i+1,
            'document': chunk['metadata'].get('filename', 'Unknown'),
            'text_preview': chunk['text'][:200],
            'relevance': chunk.get('score', 0)
        } for i, chunk in enumerate(context_chunks)],
        confidence=self._calculate_confidence(context_chunks)
    )
```

---

## Week 10: Additional Fixes

### Task 3.4: Visualization Execution
**File:** `src/backend/plugins/visualizer_agent.py`

Currently generates code but doesn't execute. Add execution:

```python
def execute(self, query: str, data: Any = None, **kwargs) -> Dict[str, Any]:
    # Generate Plotly code
    code = self._generate_viz_code(query, data)
    
    # Execute in sandbox
    from ..core.sandbox import SecureSandbox
    sandbox = SecureSandbox()
    
    result = sandbox.execute(code, {
        'df': data, 
        'pd': pd, 
        'px': plotly.express,
        'go': plotly.graph_objects
    })
    
    if result.get('fig'):
        return {
            'success': True,
            'result': result['fig'].to_json(),
            'type': 'visualization'
        }
    
    return {'success': False, 'error': 'Failed to generate visualization'}
```

---

### Task 3.5: Add Scientific File Formats
**File:** `src/backend/api/upload.py`

```python
# Update ALLOWED_EXTENSIONS
ALLOWED_EXTENSIONS = {
    # Existing
    '.csv', '.json', '.pdf', '.txt', '.xlsx', '.xls', '.docx', '.pptx', '.rtf',
    # Scientific formats (NEW)
    '.parquet', '.feather', '.hdf5', '.h5', '.nc', '.mat'
}
```

**File:** `src/backend/utils/data_utils.py`

```python
# Add to read_funcs dictionary
read_funcs = {
    # ... existing ...
    'parquet': lambda: pd.read_parquet(file_location),
    'feather': lambda: pd.read_feather(file_location),
    'hdf5': lambda: pd.read_hdf(file_location),
    'h5': lambda: pd.read_hdf(file_location),
}
```

---

### Task 3.6: Fix Bare Exception Handlers
**All Files:** Search and fix `except: pass`

```python
# BEFORE (dangerous):
try:
    something()
except:
    pass

# AFTER (safe):
try:
    something()
except Exception as e:
    logger.warning(f"Operation failed: {e}")
    # Handle gracefully or re-raise
```

---

# PHASE 4: RESEARCH READINESS
**Duration:** 3 weeks | **Priority:** HIGH (Publication Goal)

## Week 11-12: Benchmarking

### Task 4.1: Create Benchmark Dataset
**New Directory:** `benchmarks/`

```
benchmarks/
├── datasets/
│   ├── data_analysis/
│   │   ├── sales_queries.json        # 50 queries
│   │   ├── statistical_queries.json  # 50 queries
│   │   └── visualization_queries.json # 50 queries
│   └── document_qa/
│       ├── technical_docs.json       # 30 documents
│       └── research_papers.json      # 30 documents
├── baselines/
│   ├── direct_llama_responses.json
│   └── single_agent_responses.json
└── evaluation/
    ├── metrics.py
    └── run_benchmark.py
```

**Query Format (JSON):**
```json
{
  "id": "da_001",
  "query": "What is the total revenue by product category?",
  "dataset": "sales_data.csv",
  "ground_truth": {
    "type": "table",
    "answer": {"Electronics": 50000, "Clothing": 30000}
  },
  "complexity": "simple",
  "expected_agent": "DataAnalystAgent"
}
```

---

### Task 4.2: Implement Evaluation Metrics
**File:** `benchmarks/evaluation/metrics.py`

```python
import numpy as np
from typing import List, Dict, Any

class BenchmarkMetrics:
    @staticmethod
    def accuracy(predictions: List, ground_truth: List) -> float:
        """Exact match accuracy"""
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        return correct / len(predictions)
    
    @staticmethod
    def routing_accuracy(predicted_agents: List, optimal_agents: List) -> float:
        """Measure if correct agent was selected"""
        correct = sum(1 for p, o in zip(predicted_agents, optimal_agents) if p == o)
        return correct / len(predicted_agents)
    
    @staticmethod
    def latency_p95(latencies: List[float]) -> float:
        """95th percentile latency"""
        return np.percentile(latencies, 95)
    
    @staticmethod
    def code_execution_success_rate(results: List[Dict]) -> float:
        """Rate of successful code executions"""
        successful = sum(1 for r in results if r.get('success', False))
        return successful / len(results)
```

---

### Task 4.3: Run Baseline Comparisons
**Compare against:**

1. **Direct Ollama** - No agent routing (ablation)
2. **Single Agent** - All queries to DataAnalyst (ablation)
3. **ChatGPT-4** - If API key available
4. **Claude** - If API key available

---

### Task 4.4: Ablation Studies

**Must prove:**
1. Multi-agent routing improves over single agent
2. Self-correction improves answer quality
3. Code generation improves accuracy over direct LLM
4. RAM-aware selection enables resource-constrained deployment

---

## Week 13: Paper Writing

### Task 4.5: Write Research Paper

**Target Venue:** ACL/EMNLP/NeurIPS Workshop

**Structure:**
```
1. Introduction (1 page)
   - Problem: Data analysis requires expertise
   - Solution: Multi-agent LLM with local inference
   - Contributions (3-4 bullets)

2. Related Work (1 page)
   - LLM data analysis (LIDA, Code Interpreter)
   - Multi-agent systems (AutoGPT, CrewAI)
   - Self-correction in LLMs

3. System Design (2 pages)
   - Plugin architecture
   - Code generation pipeline
   - Self-correction loop

4. Experiments (2 pages)
   - Benchmark datasets
   - Baselines
   - Results

5. Analysis (1 page)
   - Ablation studies
   - Failure cases
   - Resource analysis

6. Conclusion (0.5 page)
```

---

# PHASE 5: PATENT & POLISH
**Duration:** 1 week | **Priority:** MEDIUM

### Task 5.1: Prior Art Search
Search USPTO and Google Patents for:
- "multi-agent LLM routing"
- "RAM-aware model selection"
- "self-correction data analysis"
- "plugin-based AI agent"

### Task 5.2: Document Patent Claims

**Claim 1: System Claim**
> A computer-implemented system for data analysis comprising:
> - A plugin registry that dynamically discovers agent modules at runtime
> - A routing mechanism using capability-based confidence scoring
> - A local LLM interface adapting model selection based on available RAM
> - A self-correction loop refining outputs through generator-critic iteration
> - A code generation pipeline producing executable Python for verifiable analysis

**Claim 2: Method Claim**
> A method for LLM-based data analysis comprising:
> - Receiving a natural language query
> - Generating executable code to answer the query
> - Validating the code for security vulnerabilities
> - Executing the code in a sandboxed environment
> - Interpreting results in natural language

---

### Task 5.3: Final Polish

- [ ] All tests passing (>90% coverage)
- [ ] Documentation complete and accurate
- [ ] No false claims in methodologies
- [ ] All features working end-to-end
- [ ] Performance benchmarks documented

---

# ✅ SUCCESS CRITERIA

## Research Publication
- [ ] Benchmark shows 15%+ improvement over baselines
- [ ] Ablation proves each component adds value
- [ ] Paper accepted at target venue

## Technical Completeness
- [ ] Code generation pipeline working
- [ ] All 10 agents functional
- [ ] RAG with semantic chunking
- [ ] Test coverage > 70%
- [ ] No critical security vulnerabilities

## Patent Readiness
- [ ] Prior art search completed
- [ ] Claims differentiated from prior art
- [ ] Documentation complete

---

# 📋 WEEKLY CHECKLIST TEMPLATE

Use this for each week:

```markdown
## Week N Checklist

### Tasks
- [ ] Task A: Description
- [ ] Task B: Description
- [ ] Task C: Description

### Verification
- [ ] All new code has tests
- [ ] Existing tests still pass
- [ ] Documentation updated
- [ ] No new linting errors

### Blockers
- (List any blockers)

### Notes
- (Any relevant notes)
```

---

*This roadmap supersedes all previous roadmaps. Follow phases in order.*  
*Last Updated: December 27, 2025*

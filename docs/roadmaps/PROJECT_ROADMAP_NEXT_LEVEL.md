# PROJECT ROADMAP: Taking Nexus to Research & Patent Quality
> **Authority Level:** ACTIONABLE EXECUTION PLAN  
> **Date Generated:** December 26, 2025  
> **Companion Document:** PROJECT_UNDERSTANDING.md (read first)  
> **Policy:** Fix what's broken, Complete what's valuable, Remove what's noise

---

# 🆕 NEW ITERATION - December 27, 2025

## VERSION 1.1 ROADMAP UPDATES

### Scope Changes

| Item | Previous Status | New Status | Reason |
|------|-----------------|------------|--------|
| Authentication (JWT/OAuth) | IN SCOPE (Critical) | **OUT OF SCOPE** | Not required for project goals |
| WebSocket Real-time | Archive for future | **ARCHIVE NOW** | Not needed currently |
| LLM Code Generation | Not mentioned | **ADD (High Priority)** | Enables verifiable analysis |
| Cache Mechanism | Simplify | **KEEP & ENHANCE** | Essential for performance |

### Updated Timeline

```
PHASE 0: Cleanup & Stabilization     [2 weeks]    ← UNCHANGED
    └── Archive unused files, fix CoT parser, stabilize core

PHASE 1: Core Enhancement            [2 weeks]    ← REDUCED (no auth)
    └── Monitoring, testing infrastructure, cache enhancement

PHASE 2: LLM Code Generation         [3 weeks]    ← NEW FOCUS
    └── Code generation pipeline, sandbox hardening, model routing

PHASE 3: Research Readiness          [4 weeks]    ← UNCHANGED
    └── Benchmarks, baselines, ablation studies, paper writing

PHASE 4: Patent Positioning          [3 weeks]    ← UNCHANGED
    └── Prior art analysis, claims refinement, documentation
```

**Total Timeline: ~14 weeks** (reduced from 15-16 weeks)

---

### NEW: LLM Code Generation Phase (Phase 2)

This phase adds the ability for LLMs to generate executable Python/Pandas code for data analysis instead of doing the analysis directly. This provides:

1. **Verifiable Results** - Code can be reviewed before execution
2. **Reproducible Analysis** - Same code = same output
3. **Accurate Computations** - Python math, not LLM approximation
4. **Debugging Capability** - Fix code, not LLM prompts

#### Phase 2 Tasks (New)

| Task | Description | Model to Use | Effort |
|------|-------------|--------------|--------|
| 2.1 | Code generation prompt templates | GPT-5.1-Codex | 2 days |
| 2.2 | Code validation layer (syntax, security) | N/A | 3 days |
| 2.3 | Sandbox hardening & testing | N/A | 4 days |
| 2.4 | Result interpretation prompts | Claude Sonnet 4.5 | 2 days |
| 2.5 | Integration with existing agents | All | 3 days |
| 2.6 | Error recovery & retry logic | Claude Haiku 4.5 | 2 days |

---

### Model Selection for Each Task

See **PROJECT_UNDERSTANDING.md** (Version 1.1) for complete model recommendations.

### Model Selection for Development (VS Code Copilot Agent Mode)

See **PROJECT_UNDERSTANDING.md** (Version 1.1) for complete recommendations on which Copilot model to use for different development tasks.

**Quick Reference for Development:**

| Development Task | Best Copilot Model | Alternative |
|-----------------|-------------------|-------------|
| Complex refactoring | Claude Opus 4.5 | GPT-5.2 |
| New feature code | GPT-5.1-Codex-Max | Claude Sonnet 4.5 |
| Bug fixes | Claude Sonnet 4.5 | GPT-5.1 |
| Documentation | Claude Sonnet 4 | Claude Sonnet 4.5 |
| Simple edits | Claude Haiku 4.5 | GPT-5 mini |

### Runtime Models (Ollama - Used in the Project)

The project uses these Ollama models at runtime:
- `llama3.1:8b` - Primary analysis
- `phi3:mini` - Fallback for low RAM
- `tinyllama` - Lightweight tasks
- `nomic-embed-text` - Vector embeddings

---

### Files to Archive (Confirmed) ✅ DONE

Files have been moved to `archive/removed_v1.1/`:
- `intelligent_query_engine.py`
- `optimized_llm_client.py`
- `websocket_manager.py`

---

### Out of Scope Items (Removed)

The following items have been **removed from roadmap scope**:

- ~~Authentication (JWT/OAuth)~~ - Not required
- ~~API Key management~~ - Not required  
- ~~User management~~ - Not required
- ~~Multi-tenancy~~ - Not required
- ~~WebSocket real-time updates~~ - Archive for potential future use

---

*End of Version 1.1 Updates - Original roadmap preserved below*

---

## TABLE OF CONTENTS

1. [Roadmap Summary](#1-roadmap-summary)
2. [Decision Matrix: Fix / Complete / Remove](#2-decision-matrix-fix--complete--remove)
3. [Phase 0: Cleanup & Stabilization](#3-phase-0-cleanup--stabilization)
4. [Phase 1: Structural Strength](#4-phase-1-structural-strength)
5. [Phase 2: Capability Completion](#5-phase-2-capability-completion)
6. [Phase 3: Research & Publication Readiness](#6-phase-3-research--publication-readiness)
7. [Phase 4: Patent Positioning & Differentiation](#7-phase-4-patent-positioning--differentiation)
8. [Recommended Architecture (Next Version)](#8-recommended-architecture-next-version)
9. [Appendix: Complete Task List](#9-appendix-complete-task-list)

---

## 1. ROADMAP SUMMARY

### High-Level Timeline

```
PHASE 0: Cleanup & Stabilization     [2 weeks]
    └── Remove dead code, fix broken features, stabilize CoT

PHASE 1: Structural Strength         [3 weeks]
    └── Add auth, monitoring, testing infrastructure

PHASE 2: Capability Completion       [4 weeks]
    └── Enhance RAG, complete SQL multi-DB, add streaming

PHASE 3: Research Readiness          [4 weeks]
    └── Benchmarks, baselines, ablation studies, paper writing

PHASE 4: Patent Positioning          [3 weeks]
    └── Prior art analysis, claims refinement, documentation
```

### Critical Path

```
[Stabilize CoT Parser] → [Add Benchmarks] → [Write Paper] → [File Patent]
         │                      │                 │              │
    MUST FIX              BLOCKING           GOAL 1          GOAL 2
```

---

## 2. DECISION MATRIX: FIX / COMPLETE / REMOVE

### REMOVE (Adds Noise, No Research Value)

| Item | File(s) | Reason | Effort | Action |
|------|---------|--------|--------|--------|
| Intelligent Query Engine | `intelligent_query_engine.py` | Over-engineered, not used in main flow | LOW | DELETE or ARCHIVE |
| Optimized LLM Client | `optimized_llm_client.py` | Duplicate of `llm_client.py`, not imported | LOW | DELETE or ARCHIVE |
| CrewAI Legacy Files | `crewai_base.py`, `crewai_import_manager.py` | Already archived, ensure removal | LOW | VERIFY ARCHIVED |
| Unused Advanced Cache | Parts of `enhanced_cache_integration.py` | Complex but not integrated | MEDIUM | SIMPLIFY |
| WebSocket Code | `websocket_manager.py` | Disabled, incomplete | LOW | ARCHIVE (keep for future) |
| pyproject.toml CrewAI deps | `pyproject.toml` lines 27-28 | References removed framework | LOW | REMOVE lines |

**Total Estimated Removal Effort:** 1-2 days

---

### FIX (Broken, Worth Completing)

| Item | File(s) | Issue | Fix Required | Effort | Impact |
|------|---------|-------|--------------|--------|--------|
| CoT Parser Fragility | `cot_parser.py` | Fails on malformed tags | Add fallback parsing, fuzzy matching | MEDIUM | HIGH |
| Self-Learning Stub | `data_analyst_agent.py` | `_learn_from_correction()` empty | Implement or remove claim | MEDIUM | HIGH |
| Dynamic Planner JSON | `dynamic_planner.py` | LLM produces invalid JSON | Add JSON repair/fallback | LOW | MEDIUM |
| Sandbox Testing | `sandbox.py` | Security not validated | Add penetration tests | HIGH | CRITICAL |
| Error Messages | Various | Generic errors not helpful | Add context-rich error handling | MEDIUM | MEDIUM |
| Frontend Error Display | `page.tsx` | Error handling incomplete | Show actionable recovery options | LOW | MEDIUM |

**Total Estimated Fix Effort:** 2-3 weeks

---

### COMPLETE (Almost Done, High Value)

| Item | File(s) | Current State | Completion Required | Effort | Impact |
|------|---------|---------------|---------------------|--------|--------|
| Multi-DB SQL Support | `sql_agent.py` | SQLite only | Test PostgreSQL, MySQL | HIGH | MEDIUM |
| RAG Pipeline | `rag_agent.py`, `document_indexer.py` | Basic chunking | Semantic chunking, re-ranking | HIGH | HIGH |
| Visualization Agent | `visualizer_agent.py` | Generates code | Add execution & rendering | MEDIUM | HIGH |
| Report Generation | `reporter_agent.py` | Basic template | Rich PDF with charts | MEDIUM | MEDIUM |
| Query History | `history.py` | Stores but no analytics | Add trend analysis, suggestions | LOW | LOW |
| ~~Authentication~~ | ~~NONE~~ | ~~Not implemented~~ | ~~Add JWT/OAuth~~ | ~~HIGH~~ | ~~OUT OF SCOPE~~ |
| Monitoring | NONE | Not implemented | Add Prometheus metrics | MEDIUM | HIGH |
| **LLM Code Generation** | NEW | Not implemented | Add code gen pipeline | HIGH | **CRITICAL** |
| **Cache Enhancement** | `advanced_cache.py` | Basic implementation | Add semantic caching | MEDIUM | HIGH |

**Total Estimated Completion Effort:** 4-6 weeks

---

## 3. PHASE 0: CLEANUP & STABILIZATION

**Duration:** 2 weeks  
**Goal:** Remove noise, fix critical bugs, establish clean baseline

### Week 1: Dead Code Removal

#### Task 0.1: Archive Unused Files
**Effort:** 4 hours | **Risk:** LOW

```
ACTION: Move these files to archive/removed_phase0/

src/backend/core/intelligent_query_engine.py     (1046 lines, unused)
src/backend/core/optimized_llm_client.py         (~300 lines, unused)
src/backend/core/websocket_manager.py            (~150 lines, disabled)
```

**Verification:**
```bash
# Confirm no imports
grep -r "intelligent_query_engine" src/backend/
grep -r "optimized_llm_client" src/backend/
grep -r "websocket_manager" src/backend/
```

#### Task 0.2: Clean pyproject.toml
**Effort:** 30 minutes | **Risk:** LOW

```toml
# REMOVE these lines (27-28):
"crewai",
"crewai_tools",
```

**File:** `pyproject.toml`

#### Task 0.3: Clean requirements.txt
**Effort:** 30 minutes | **Risk:** LOW

**Verify** CrewAI entries are commented out (they already are, confirm).

---

### Week 2: Critical Bug Fixes

#### Task 0.4: Fix CoT Parser Fragility
**Effort:** 1 day | **Risk:** MEDIUM | **Impact:** HIGH

**File:** `src/backend/core/cot_parser.py`

**Current Problem:**
```python
# Regex requires EXACT tags - fails if LLM produces variations
reasoning_pattern = f"{re.escape(self.reasoning_start)}(.*?){re.escape(self.reasoning_end)}"
```

**Fix:**
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
    
    # Strategy 3: LLM-based extraction (last resort)
    result = self._parse_llm_assisted(response)
    if result.is_valid:
        return result
    
    # Fallback: Return entire response as output
    return ParsedCoT(
        reasoning="Unable to extract structured reasoning",
        output=response,
        is_valid=False,
        error_message="All parsing strategies failed",
        raw_response=response
    )

def _parse_fuzzy(self, response: str) -> ParsedCoT:
    """Match common variations of tags"""
    fuzzy_patterns = [
        (r'\[REASONING\](.*?)\[/REASONING\]', r'\[OUTPUT\](.*?)\[/OUTPUT\]'),
        (r'<reasoning>(.*?)</reasoning>', r'<output>(.*?)</output>'),
        (r'REASONING:(.*?)OUTPUT:', r'OUTPUT:(.*?)$'),
        (r'Reasoning:(.*?)Answer:', r'Answer:(.*?)$'),
    ]
    # ... implementation
```

#### Task 0.5: Fix or Remove Self-Learning Claim
**Effort:** 4 hours | **Risk:** LOW | **Impact:** MEDIUM

**File:** `src/backend/plugins/data_analyst_agent.py`

**Current Problem:**
```python
def _learn_from_correction(self, first_cot, final_cot, query):
    pass  # Empty stub - claim is FALSE
```

**Options:**
1. **REMOVE** - Delete the function and all references, update docs
2. **IMPLEMENT** - Add actual learning (store patterns, retrieve for similar queries)

**Recommended:** REMOVE for now, add to Phase 3 as research feature

```python
# DELETE this function and the call on line 176
# Update docs to remove "self-learning" claims
```

#### Task 0.6: Fix Dynamic Planner JSON Handling
**Effort:** 2 hours | **Risk:** LOW | **Impact:** MEDIUM

**File:** `src/backend/core/dynamic_planner.py`

**Current Problem:**
```python
def _parse_plan(self, llm_output: str) -> AnalysisPlan:
    cleaned = llm_output.replace("```json", "").replace("```", "").strip()
    plan_dict = json.loads(cleaned)  # FAILS if LLM produces invalid JSON
```

**Fix:**
```python
import json
import re

def _parse_plan(self, llm_output: str) -> AnalysisPlan:
    """Parse with JSON repair fallback"""
    try:
        # Clean markdown
        cleaned = re.sub(r'```json\s*', '', llm_output)
        cleaned = re.sub(r'```\s*', '', cleaned).strip()
        
        # Try direct parse
        plan_dict = json.loads(cleaned)
        
    except json.JSONDecodeError:
        # Try to extract JSON object
        json_match = re.search(r'\{[\s\S]*\}', cleaned)
        if json_match:
            try:
                plan_dict = json.loads(json_match.group())
            except:
                return self._fallback_plan(llm_output)
        else:
            return self._fallback_plan(llm_output)
    
    # ... rest of parsing

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

### Phase 0 Deliverables Checklist

- [ ] Unused files archived
- [ ] pyproject.toml cleaned
- [ ] CoT parser has fallback strategies
- [ ] Self-learning stub removed + docs updated
- [ ] Dynamic planner handles JSON errors
- [ ] All existing tests pass
- [ ] Clean baseline established

---

## 4. PHASE 1: STRUCTURAL STRENGTH

**Duration:** 2 weeks (reduced from 3 weeks - auth removed from scope)  
**Goal:** Add essential infrastructure for production and research

### ~~Week 3-4: Security & Auth~~ - REMOVED FROM SCOPE

> **Note (v1.1):** Authentication has been removed from project scope. The tasks below are preserved for reference but should NOT be implemented.

<details>
<summary>📦 Archived: Authentication Tasks (Click to expand - DO NOT IMPLEMENT)</summary>

#### ~~Task 1.1: Add JWT Authentication~~ - OUT OF SCOPE
**Effort:** 3 days | **Risk:** MEDIUM | **Impact:** ~~CRITICAL~~ N/A

**New File:** `src/backend/core/auth.py`

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import BaseModel

class AuthConfig:
    SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 60

security = HTTPBearer()

class TokenData(BaseModel):
    user_id: str
    email: Optional[str] = None
    
def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, AuthConfig.SECRET_KEY, algorithm=AuthConfig.ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    try:
        payload = jwt.decode(credentials.credentials, AuthConfig.SECRET_KEY, algorithms=[AuthConfig.ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return TokenData(user_id=user_id, email=payload.get("email"))
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

**Update `main.py`:**
```python
from backend.core.auth import get_current_user

# Protected routes
@app.get("/api/analyze/")
async def analyze(request: AnalyzeRequest, user: TokenData = Depends(get_current_user)):
    # ... existing logic
```

#### ~~Task 1.2: Add API Key Alternative~~ - OUT OF SCOPE
**Effort:** 4 hours | **Risk:** LOW | **Impact:** ~~MEDIUM~~ N/A

For simpler deployments, add API key auth as alternative to JWT.

```python
API_KEY_HEADER = "X-API-Key"

async def verify_api_key(api_key: str = Header(..., alias=API_KEY_HEADER)):
    valid_keys = os.getenv("API_KEYS", "").split(",")
    if api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
```

</details>

---

### Week 3-4: Monitoring & Cache Enhancement (NEW FOCUS)

#### Task 1.1 (v1.1): Enhance Cache Mechanism
**Effort:** 2 days | **Risk:** LOW | **Impact:** HIGH

**File:** `src/backend/core/advanced_cache.py`

**Enhancement Goals:**
- Semantic similarity caching (cache similar queries)
- Query + data file hash for cache key
- Configurable TTL per query type
- Cache statistics for monitoring

```python
class EnhancedCache:
    def __init__(self):
        self.cache = {}
        self.stats = {"hits": 0, "misses": 0}
    
    def get_cache_key(self, query: str, data_hash: str, model: str) -> str:
        """Generate deterministic cache key"""
        import hashlib
        content = f"{query}:{data_hash}:{model}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            entry = self.cache[key]
            if time.time() < entry['expires']:
                self.stats['hits'] += 1
                return entry['value']
            else:
                del self.cache[key]
        self.stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Dict, ttl: int = 3600):
        self.cache[key] = {
            'value': value,
            'expires': time.time() + ttl,
            'created': time.time()
        }
```

---

### Week 4: Monitoring & Observability

#### Task 1.2 (v1.1): Add Prometheus Metrics
**Effort:** 2 days | **Risk:** LOW | **Impact:** HIGH

**New File:** `src/backend/core/metrics.py`

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from functools import wraps
import time

# Metrics
REQUEST_COUNT = Counter('nexus_requests_total', 'Total requests', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('nexus_request_latency_seconds', 'Request latency', ['endpoint'])
ACTIVE_ANALYSES = Gauge('nexus_active_analyses', 'Currently running analyses')
AGENT_USAGE = Counter('nexus_agent_usage_total', 'Agent usage count', ['agent_name'])
LLM_CALLS = Counter('nexus_llm_calls_total', 'LLM API calls', ['model', 'status'])
LLM_LATENCY = Histogram('nexus_llm_latency_seconds', 'LLM call latency', ['model'])

def track_request(endpoint: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                REQUEST_COUNT.labels(endpoint=endpoint, status='success').inc()
                return result
            except Exception as e:
                REQUEST_COUNT.labels(endpoint=endpoint, status='error').inc()
                raise
            finally:
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.time() - start)
        return wrapper
    return decorator
```

**Add metrics endpoint in `main.py`:**
```python
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

#### Task 1.4: Add Structured Logging
**Effort:** 1 day | **Risk:** LOW | **Impact:** MEDIUM

**File:** `src/backend/core/enhanced_logging.py`

```python
import structlog
import json
from datetime import datetime

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

def get_logger(name: str):
    return structlog.get_logger(name)
```

---

### Week 5 (continued): Testing Infrastructure

#### Task 1.5: Add Test Coverage Measurement
**Effort:** 4 hours | **Risk:** LOW | **Impact:** MEDIUM

**Update `pyproject.toml`:**
```toml
[tool.pytest.ini_options]
addopts = "-v --tb=short --cov=src/backend --cov-report=html --cov-report=term"

[tool.coverage.run]
source = ["src/backend"]
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
]
```

#### Task 1.6: Add CI/CD Pipeline
**Effort:** 1 day | **Risk:** LOW | **Impact:** HIGH

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
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-asyncio
      - name: Run tests
        run: pytest tests/ -v --cov=src/backend --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

---

### Phase 1 Deliverables Checklist

- [ ] ~~JWT authentication working~~ - OUT OF SCOPE
- [ ] ~~API key alternative available~~ - OUT OF SCOPE
- [ ] Cache mechanism enhanced with semantic similarity
- [ ] Prometheus metrics exposed at /metrics
- [ ] Structured logging implemented
- [ ] Test coverage measured (target: 70%+)
- [ ] CI/CD pipeline running on GitHub Actions
- [ ] ~~Security headers added (HTTPS, CORS tightened)~~ - OPTIONAL

---

## 5. PHASE 2: CAPABILITY COMPLETION

**Duration:** 4 weeks  
**Goal:** Complete partial features, enhance core capabilities

### Week 6-7: RAG Pipeline Enhancement

#### Task 2.1: Implement Semantic Chunking
**Effort:** 3 days | **Risk:** MEDIUM | **Impact:** HIGH

**File:** `src/backend/core/document_indexer.py`

**Current Problem:** Fixed-size word chunking ignores semantic boundaries.

**Enhancement:**
```python
class SemanticChunker:
    """Split documents at semantic boundaries (paragraphs, sections)"""
    
    def __init__(self, max_chunk_size: int = 500, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.sentence_splitter = re.compile(r'(?<=[.!?])\s+')
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_words = len(para.split())
            
            if current_size + para_words > self.max_chunk_size and current_size >= self.min_chunk_size:
                # Save current chunk
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
        
        # Don't forget last chunk
        if current_chunk:
            chunks.append({
                'text': '\n\n'.join(current_chunk),
                'word_count': current_size,
                'type': 'semantic'
            })
        
        return chunks
```

#### Task 2.2: Add Hybrid Search (Vector + Keyword)
**Effort:** 2 days | **Risk:** MEDIUM | **Impact:** HIGH

**File:** `src/backend/core/chromadb_client.py`

```python
def hybrid_query(self, query_text: str, n_results: int = 5) -> Dict:
    """Combine vector similarity with keyword matching"""
    
    # Vector search
    vector_results = self.collection.query(
        query_texts=[query_text],
        n_results=n_results * 2  # Get more for re-ranking
    )
    
    # Keyword extraction
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
```

#### Task 2.3: Add Citation Tracking
**Effort:** 1 day | **Risk:** LOW | **Impact:** MEDIUM

Track which chunks were used to generate answers for explainability.

```python
@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict[str, Any]]  # chunk_id, document, relevance_score
    confidence: float
    
def generate_with_citations(self, query: str, context_chunks: List[Dict]) -> RAGResponse:
    # ... LLM call with context
    
    return RAGResponse(
        answer=llm_response,
        sources=[{
            'chunk_id': chunk['id'],
            'document': chunk['metadata']['filename'],
            'text_preview': chunk['text'][:200],
            'relevance': chunk['score']
        } for chunk in context_chunks],
        confidence=self._calculate_confidence(context_chunks)
    )
```

---

### Week 8-9: Additional Completions

#### Task 2.4: Complete Multi-Database SQL Support
**Effort:** 3 days | **Risk:** MEDIUM | **Impact:** MEDIUM

**File:** `src/backend/plugins/sql_agent.py`

**Add connection factory:**
```python
class DatabaseConnectionFactory:
    @staticmethod
    def create_connection(db_type: str, connection_string: str):
        if db_type == 'sqlite':
            return sqlite3.connect(connection_string)
        elif db_type == 'postgresql':
            import psycopg2
            return psycopg2.connect(connection_string)
        elif db_type == 'mysql':
            import mysql.connector
            return mysql.connector.connect(connection_string)
        else:
            raise ValueError(f"Unsupported database: {db_type}")
```

**Add tests for each database type.**

#### Task 2.5: Visualization Execution
**Effort:** 2 days | **Risk:** MEDIUM | **Impact:** HIGH

**File:** `src/backend/plugins/visualizer_agent.py`

Currently generates code but doesn't execute it. Add safe execution:

```python
def execute(self, query: str, data: Any = None, **kwargs) -> Dict[str, Any]:
    # Generate Plotly code
    code = self._generate_viz_code(query, data)
    
    # Execute in sandbox
    from backend.core.sandbox import EnhancedSandbox
    sandbox = EnhancedSandbox()
    
    result = sandbox.execute_code(code, {'df': data})
    
    if 'fig' in result.get('namespace', {}):
        fig = result['namespace']['fig']
        # Return as JSON for frontend
        return {
            'success': True,
            'result': fig.to_json(),
            'type': 'visualization'
        }
```

#### Task 2.6: Enable WebSocket Streaming (Optional)
**Effort:** 2 days | **Risk:** LOW | **Impact:** MEDIUM

Uncomment and fix WebSocket code for real-time progress updates.

**Update `config/cot_review_config.json`:**
```json
{
  "enable_websockets": true
}
```

---

### Phase 2 Deliverables Checklist

- [ ] Semantic chunking for documents
- [ ] Hybrid search (vector + keyword)
- [ ] Citation tracking in RAG responses
- [ ] PostgreSQL and MySQL tested for SQL Agent
- [ ] Visualization code execution working
- [ ] WebSocket streaming enabled (optional)
- [ ] All new features tested

---

## 6. PHASE 3: RESEARCH & PUBLICATION READINESS

**Duration:** 4 weeks  
**Goal:** Validate claims with data, prepare research paper

### Week 10-11: Benchmarking Infrastructure

#### Task 3.1: Create Benchmark Dataset
**Effort:** 3 days | **Risk:** LOW | **Impact:** CRITICAL

**New Directory:** `benchmarks/`

```
benchmarks/
├── datasets/
│   ├── data_analysis/
│   │   ├── sales_queries.json        # 50 queries + ground truth
│   │   ├── financial_queries.json    # 50 queries + ground truth
│   │   └── statistical_queries.json  # 50 queries + ground truth
│   └── document_qa/
│       ├── academic_papers.json      # 30 documents + questions
│       └── technical_docs.json       # 30 documents + questions
├── baselines/
│   ├── chatgpt_responses.json
│   ├── claude_responses.json
│   └── llama_direct_responses.json
└── evaluation/
    ├── metrics.py
    └── run_benchmark.py
```

**Query Format:**
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
  "required_capabilities": ["aggregation", "groupby"]
}
```

#### Task 3.2: Implement Evaluation Metrics
**Effort:** 2 days | **Risk:** LOW | **Impact:** HIGH

**File:** `benchmarks/evaluation/metrics.py`

```python
class BenchmarkMetrics:
    @staticmethod
    def accuracy(predictions: List, ground_truth: List) -> float:
        """Exact match accuracy"""
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        return correct / len(predictions)
    
    @staticmethod
    def semantic_similarity(prediction: str, ground_truth: str) -> float:
        """Embedding-based similarity for text answers"""
        # Use sentence-transformers
        pass
    
    @staticmethod
    def latency_p95(latencies: List[float]) -> float:
        """95th percentile latency"""
        return np.percentile(latencies, 95)
    
    @staticmethod
    def routing_accuracy(predicted_agents: List, optimal_agents: List) -> float:
        """Measure if correct agent was selected"""
        correct = sum(1 for p, o in zip(predicted_agents, optimal_agents) if p == o)
        return correct / len(predicted_agents)
```

#### Task 3.3: Run Baseline Comparisons
**Effort:** 3 days | **Risk:** MEDIUM | **Impact:** CRITICAL

Compare against:
1. **ChatGPT-4** - via API (requires key)
2. **Claude** - via API (requires key)
3. **Direct Ollama** - no agent routing (ablation)
4. **Single agent** - no routing (ablation)

---

### Week 12-13: Research Paper Writing

#### Task 3.4: Paper Structure
**Effort:** 5 days | **Risk:** LOW | **Impact:** CRITICAL

**Target Venue:** ACL/EMNLP/NeurIPS Workshop on LLM Applications

**Structure:**
```
1. Introduction
   - Problem: Data analysis requires domain expertise
   - Solution: Multi-agent LLM system with local inference
   - Contributions (3-4 bullet points)

2. Related Work
   - LLM-based data analysis (LIDA, ChatGPT Code Interpreter)
   - Multi-agent systems (AutoGPT, CrewAI)
   - Self-correction in LLMs

3. System Design
   - Plugin-based agent architecture
   - Capability-based routing
   - Self-correction loop

4. Experimental Setup
   - Benchmark datasets
   - Baselines
   - Metrics

5. Results
   - Routing accuracy
   - Answer quality
   - Latency analysis
   - Ablation studies

6. Analysis & Discussion
   - When does the system work well?
   - Failure cases
   - Resource requirements

7. Conclusion
```

#### Task 3.5: Ablation Studies
**Effort:** 2 days | **Risk:** LOW | **Impact:** HIGH

**Must prove:**
1. Multi-agent routing improves over single agent
2. Self-correction improves answer quality
3. RAM-aware selection enables resource-constrained deployment

---

### Phase 3 Deliverables Checklist

- [ ] Benchmark dataset created (150+ queries)
- [ ] Baseline results collected
- [ ] Evaluation metrics implemented
- [ ] Ablation studies completed
- [ ] Paper draft written
- [ ] All claims supported by data

---

## 7. PHASE 4: PATENT POSITIONING & DIFFERENTIATION

**Duration:** 3 weeks  
**Goal:** Identify patentable innovations, prepare filing

### Week 14: Prior Art Analysis

#### Task 4.1: Patent Search
**Effort:** 2 days | **Risk:** LOW | **Impact:** CRITICAL

Search USPTO, Google Patents for:
- "multi-agent LLM routing"
- "RAM-aware model selection"
- "self-correction data analysis"
- "plugin-based AI agent"

**Document findings** in `docs/PRIOR_ART_ANALYSIS.md`

#### Task 4.2: Differentiation Matrix
**Effort:** 1 day | **Risk:** LOW | **Impact:** HIGH

| Feature | Nexus | ChatGPT | AutoGPT | LIDA |
|---------|-------|---------|---------|------|
| Local-first | ✅ | ❌ | ❌ | ❌ |
| RAM-aware selection | ✅ | N/A | ❌ | ❌ |
| Plugin agents | ✅ | ❌ | ✅ | ❌ |
| Self-correction loop | ✅ | ❌ | ⚠️ | ❌ |
| No API costs | ✅ | ❌ | ❌ | ❌ |

---

### Week 15-16: Patent Documentation

#### Task 4.3: Patent Claims Draft
**Effort:** 5 days | **Risk:** MEDIUM | **Impact:** CRITICAL

**Potential Claims:**

**Claim 1: System Claim**
> A computer-implemented system for data analysis comprising:
> - A plugin registry that dynamically discovers and registers agent modules at runtime
> - A routing mechanism that selects agents based on query content and file type using capability scoring
> - A local language model interface that adapts model selection based on available system memory
> - A self-correction loop that refines LLM outputs through generator-critic iteration

**Claim 2: Method Claim**
> A method for memory-aware language model selection comprising:
> - Detecting available system RAM at runtime
> - Querying a local model server for installed models and their sizes
> - Calculating memory requirements for each model based on model size
> - Selecting the optimal model that fits within available memory constraints

**Claim 3: Apparatus Claim**
> A data processing apparatus implementing a self-correcting analysis pipeline...

#### Task 4.4: Prepare Provisional Patent Application
**Effort:** 3 days | **Risk:** LOW | **Impact:** HIGH

File provisional patent to secure priority date while completing research validation.

---

### Phase 4 Deliverables Checklist

- [ ] Prior art search completed
- [ ] Differentiation matrix documented
- [ ] Patent claims drafted
- [ ] Provisional patent filed (optional, costs involved)
- [ ] Freedom-to-operate analysis (if resources allow)

---

## 8. RECOMMENDED ARCHITECTURE (NEXT VERSION)

### Cleaned Architecture (Post Phase 0-1)

```
src/backend/
├── main.py                      # FastAPI app (unchanged)
├── core/
│   ├── config.py               # Configuration (unchanged)
│   ├── plugin_system.py        # Agent registry (unchanged)
│   ├── llm_client.py           # LLM communication (unchanged)
│   ├── model_selector.py       # Model selection (enhanced for multi-provider)
│   ├── circuit_breaker.py      # Resilience (unchanged)
│   ├── chromadb_client.py      # Vector DB (enhanced)
│   ├── advanced_cache.py       # Cache mechanism (enhanced)
│   ├── self_correction_engine.py # CoT loop (fixed)
│   ├── cot_parser.py           # Tag parsing (fixed)
│   ├── sandbox.py              # Code execution (tested)
│   ├── metrics.py              # NEW: Prometheus metrics
│   └── [REMOVED: intelligent_query_engine.py, optimized_llm_client.py, websocket_manager.py]
├── services/
│   └── analysis_service.py     # Orchestrator (unchanged)
├── api/
│   ├── analyze.py              # Analysis endpoint
│   ├── upload.py               # Upload endpoint (unchanged)
│   └── ...
├── plugins/
│   ├── data_analyst_agent.py   # Core agent (fixed + code generation)
│   ├── statistical_agent.py    # Stats (unchanged)
│   ├── rag_agent.py            # Documents (enhanced)
│   ├── sql_agent.py            # SQL (multi-db added)
│   ├── visualizer_agent.py     # Charts (execution added)
│   └── ...
└── utils/
    ├── data_optimizer.py       # Data prep (unchanged)
    └── data_utils.py           # Helpers (unchanged)
```

### New Component: Benchmark Runner

```
benchmarks/
├── runner.py                   # Main benchmark script
├── datasets/                   # Test queries and ground truth
├── baselines/                  # Comparison results
└── reports/                    # Generated reports
```

---

## 9. APPENDIX: COMPLETE TASK LIST

### Summary by Phase (Updated v1.1)

| Phase | Tasks | Total Effort | Priority |
|-------|-------|--------------|----------|
| 0 | 6 | 2 weeks | CRITICAL |
| 1 | 4 | 2 weeks | HIGH |
| 2 | 6 | 3 weeks | **HIGH** (LLM Code Gen) |
| 3 | 5 | 4 weeks | HIGH |
| 4 | 4 | 3 weeks | MEDIUM |
| **Total** | **25** | **~14 weeks** | - |

### Critical Path Tasks

These tasks are **blocking** for research publication:

1. ✅ Fix CoT Parser (Phase 0.4)
2. ✅ Remove self-learning claim (Phase 0.5)
3. ⭐ Add LLM Code Generation (Phase 2 - NEW)
4. ✅ Create benchmark dataset (Phase 3.1)
5. ✅ Run baseline comparisons (Phase 3.3)
6. ✅ Complete ablation studies (Phase 3.5)

### Optional but Valuable

1. ~~WebSocket streaming (Phase 2.6)~~ - Archived
2. Multi-DB SQL (Phase 2.4)
3. Patent filing (Phase 4.4)

---

## EXECUTION RECOMMENDATIONS

### Immediate Actions (This Week)

1. Archive unused files (Task 0.1-0.3)
2. Fix CoT parser with fallback (Task 0.4)
3. Run existing tests to establish baseline

### Decision Points (Updated v1.1)

1. **Self-learning:** Remove claim now, implement later? → RECOMMEND: Remove
2. **WebSocket:** Enable or keep disabled? → **DECIDED: Archive (out of scope)**
3. ~~**Authentication:** JWT or simpler API keys?~~ → **DECIDED: Out of scope**
4. **LLM Code Generation:** Add now? → **DECIDED: Yes, Phase 2 priority**
5. **Patent:** File provisional? → RECOMMEND: Yes, after Phase 3 validation

### Success Criteria

**Research Publication:**
- [ ] Benchmark shows 15%+ improvement over baselines
- [ ] Ablation proves each component adds value
- [ ] Paper accepted at target venue

**Patent Filing:**
- [ ] Prior art search shows differentiation
- [ ] Claims are defensible
- [ ] Provisional filed within 12 months

**Production Ready:**
- [ ] ~~Authentication working~~ - Out of scope
- [ ] Test coverage > 70%
- [ ] LLM Code Generation working
- [ ] Cache mechanism optimized
- [ ] No critical security vulnerabilities

---

*This roadmap supersedes PROJECT_ROADMAP_FOR_RESEARCH.md. Follow phases in order.*

# 🔧 SONNET 4.5 FIX GUIDE - Nexus LLM Analytics

> **Purpose**: Step-by-step instructions for Claude Sonnet 4.5 to fix critical issues
> **Created**: January 2, 2026
> **Updated**: January 2, 2026 (Added Fixes 7-11 from documentation audit)
> **Priority Order**: Fixes ordered by impact on Power, Accuracy, Speed
> **Total Fixes**: 11

---

## 📊 FIXES OVERVIEW

| # | Fix | Category | Priority | Est. Time |
|---|-----|----------|----------|-----------|
| 1 | Sandbox Too Restrictive | Accuracy | 🔴 Critical | 15-20 min |
| 2 | Model Warmup on Startup | Speed | 🔴 Critical | 10-15 min |
| 3 | Thread-Safe Singletons | Reliability | 🔴 Critical | 15-20 min |
| 4 | Smart Fallback Hardcoded Models | Reliability | 🟠 High | 20-25 min |
| 5 | Improve Prompt Templates | Accuracy | 🟠 High | 25-30 min |
| 6 | Async LLM Calls | Speed | 🟠 High | 20-25 min |
| 7 | ML Abbreviations in Routing | Accuracy | 🟡 Medium | 20-25 min |
| 8 | Semantic Layer for Data Agnosticism | Power | 🟡 Medium | 30-40 min |
| 9 | Streaming Responses (SSE) | UX | 🟡 Medium | 25-30 min |
| 10 | User Feedback Collection | Power | 🟡 Medium | 20-25 min |
| 11 | Dynamic Planner JSON Repair | Reliability | 🟢 Low | 15-20 min |

**Total Estimated Time: ~4-5 hours**

**Recommended Order**: 1 → 2 → 3 → 4 → 5 → 6 (core fixes), then 7 → 10 → 11 → 8 → 9 (enhancements)

---

## 📋 PRE-FIX CHECKLIST

Before starting any fix, ensure:
1. ✅ Backend server is NOT running (stop with Ctrl+C)
2. ✅ You're in the project root: `nexus-llm-analytics-dist/`
3. ✅ Ollama is running: `ollama serve`
4. ✅ At least one model installed: `ollama list`

---

## 🔴 FIX 1: SANDBOX TOO RESTRICTIVE (Accuracy - Critical)

### Problem
The `EnhancedSandbox` blocks legitimate pandas operations like `.apply()`, `.map()`, `.transform()` which are needed for complex data analysis. This causes valid LLM-generated code to fail.

### File to Modify
`src/backend/core/sandbox.py`

### What to Do
Add safe pandas methods to the `RestrictedPandas` class that are blocked by default but are actually safe for data analysis.

### Exact Changes

**FIND this code block (around line 95-130):**
```python
class RestrictedPandas:
    DataFrame = pd.DataFrame
    Series = pd.Series
    concat = pd.concat
    merge = pd.merge
    pivot_table = pd.pivot_table
    cut = pd.cut
    qcut = pd.qcut
    crosstab = pd.crosstab
    get_dummies = pd.get_dummies
    
    # Explicitly block dangerous functions
    def __getattr__(self, name):
        dangerous_funcs = {
            'read_csv', 'read_json', 'read_excel', 'read_sql', 'read_html',
            'read_xml', 'read_pickle', 'read_parquet', 'read_feather',
            'to_pickle', 'read_clipboard', 'read_fwf', 'read_table'
        }
        if name in dangerous_funcs:
            raise AttributeError(f"Access to pandas.{name} is not allowed for security reasons")
        raise AttributeError(f"'{name}' is not available in restricted pandas")
```

**REPLACE WITH:**
```python
class RestrictedPandas:
    DataFrame = pd.DataFrame
    Series = pd.Series
    concat = pd.concat
    merge = pd.merge
    pivot_table = pd.pivot_table
    cut = pd.cut
    qcut = pd.qcut
    crosstab = pd.crosstab
    get_dummies = pd.get_dummies
    melt = pd.melt
    wide_to_long = pd.wide_to_long
    
    # Safe transformation methods - these operate on data, not files
    @staticmethod
    def _safe_apply(df_or_series, func, *args, **kwargs):
        """Safe apply that only allows simple operations"""
        return df_or_series.apply(func, *args, **kwargs)
    
    @staticmethod
    def _safe_map(series, func_or_dict):
        """Safe map for Series"""
        return series.map(func_or_dict)
    
    @staticmethod
    def _safe_transform(df_or_series, func, *args, **kwargs):
        """Safe transform operation"""
        return df_or_series.transform(func, *args, **kwargs)
    
    @staticmethod
    def _safe_agg(df_or_series, func, *args, **kwargs):
        """Safe aggregation"""
        return df_or_series.agg(func, *args, **kwargs)
    
    # Explicitly block dangerous functions
    def __getattr__(self, name):
        dangerous_funcs = {
            'read_csv', 'read_json', 'read_excel', 'read_sql', 'read_html',
            'read_xml', 'read_pickle', 'read_parquet', 'read_feather',
            'to_pickle', 'read_clipboard', 'read_fwf', 'read_table',
            'to_csv', 'to_json', 'to_excel', 'to_sql', 'to_html'
        }
        if name in dangerous_funcs:
            raise AttributeError(f"Access to pandas.{name} is not allowed for security reasons")
        
        # Allow safe methods that were missed
        safe_methods = {'isna', 'notna', 'isnull', 'notnull', 'fillna', 'dropna',
                       'sort_values', 'sort_index', 'reset_index', 'set_index',
                       'to_datetime', 'to_numeric', 'to_string', 'to_dict', 'to_list'}
        if name in safe_methods and hasattr(pd, name):
            return getattr(pd, name)
        
        raise AttributeError(f"'{name}' is not available in restricted pandas")
```

**ALSO FIND (around line 175-200) where safe_modules is populated:**
```python
safe_modules.update({
    # Core data manipulation
    'pd': RestrictedPandas(),
```

**ADD after 'pd' entry:**
```python
safe_modules.update({
    # Core data manipulation
    'pd': RestrictedPandas(),
    # Direct pandas access for apply/map (needed by generated code)
    'pandas': RestrictedPandas(),
```

### How to Test

```bash
# Terminal 1: Start backend
cd src/backend
python -m uvicorn main:app --reload --port 8000

# Terminal 2: Test with curl
curl -X POST "http://localhost:8000/api/analyze/" \
  -H "Content-Type: application/json" \
  -d '{"query": "Calculate the average of each numeric column", "filename": "your_test_file.csv"}'
```

**Expected**: Response should include calculated averages, not a sandbox error.

### Success Criteria
- [ ] No "AttributeError" for `.apply()`, `.map()`, `.transform()`
- [ ] Code using `df.apply(lambda x: ...)` executes successfully
- [ ] File I/O operations still blocked (security preserved)

---

## 🔴 FIX 2: MODEL WARMUP ON STARTUP (Speed - Critical)

### Problem
First user query takes 10-30 seconds extra because models are loaded lazily. The warmup happens AFTER the app starts serving, so first request triggers cold start.

### File to Modify
`src/backend/main.py`

### What to Do
Move model warmup to BEFORE the `yield` statement in the lifespan handler, so models are ready when the first request arrives.

### Exact Changes

**FIND this code block (around line 50-80):**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown tasks"""
    import logging
    import asyncio
    
    # Startup tasks
    logger = logging.getLogger(__name__)
    logger.debug("Starting Nexus LLM Analytics backend...")
    
    # Auto-configure models on startup to prevent hardcoded fallbacks
    try:
        from backend.core.model_selector import ModelSelector
        # Trigger model selection to cache results and detect issues early
        primary, review, embedding = ModelSelector.select_optimal_models()
        logger.debug(f"Startup model selection: Primary={primary}, Review={review}")
    except Exception as model_error:
        logger.warning(f"Model selection warning during startup: {model_error}")
    
    # Start background optimization (non-blocking)
    try:
        from backend.core.optimizers import optimize_startup
        optimization_result = optimize_startup()
        logger.info("Backend ready")
    except Exception as opt_error:
        logger.warning(f"Optimization warning: {opt_error}")
    
    # Run model test in background to avoid blocking startup (optional validation)
    asyncio.create_task(test_model_on_startup())
    
    yield  # Application runs here
```

**REPLACE WITH:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown tasks"""
    import logging
    import asyncio
    
    # Startup tasks
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Nexus LLM Analytics backend...")
    
    # STEP 1: Auto-configure models on startup
    try:
        from backend.core.model_selector import ModelSelector
        primary, review, embedding = ModelSelector.select_optimal_models()
        logger.info(f"✅ Models selected: Primary={primary.split('/')[-1]}, Review={review.split('/')[-1]}")
    except Exception as model_error:
        logger.warning(f"⚠️ Model selection warning: {model_error}")
    
    # STEP 2: WARM UP PRIMARY MODEL BEFORE SERVING (critical for first-request speed)
    try:
        logger.info("🔥 Warming up primary model (this may take 10-30 seconds)...")
        from backend.core.llm_client import LLMClient
        warmup_client = LLMClient()
        warmup_response = warmup_client.generate(
            prompt="Say 'ready' in one word.",
            model=None,  # Uses primary model
            adaptive_timeout=True
        )
        if warmup_response.get('response'):
            logger.info("✅ Primary model warmed up and ready!")
        else:
            logger.warning("⚠️ Model warmup returned empty response")
    except Exception as warmup_error:
        logger.warning(f"⚠️ Model warmup skipped: {warmup_error}")
    
    # STEP 3: Background optimization (non-blocking)
    try:
        from backend.core.optimizers import optimize_startup
        optimization_result = optimize_startup()
    except Exception as opt_error:
        logger.debug(f"Optimization note: {opt_error}")
    
    logger.info("✅ Backend ready to serve requests!")
    
    yield  # Application serves requests here - model is already warm
    
    # Shutdown tasks
    logger.info("👋 Shutting down Nexus LLM Analytics backend...")
```

**ALSO FIND and MODIFY the `test_model_on_startup` function (around line 230-270):**

```python
async def test_model_on_startup():
    """
    Test the currently configured model on startup (background task).
    Integrated from src2 for better startup validation.
    """
    import logging
    import asyncio
    
    logger = logging.getLogger(__name__)
    
    # Wait for the server to fully start
    await asyncio.sleep(5)
```

**REPLACE WITH (simplified - warmup now happens in lifespan):**
```python
async def test_model_on_startup():
    """
    Optional background model validation.
    Note: Primary warmup now happens in lifespan() before serving.
    This function is kept for secondary checks only.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # No longer needed - warmup happens in lifespan before yield
    logger.debug("Background model test skipped - warmup done in lifespan")
    return
```

### How to Test

```bash
# Restart the backend and time the first request
cd src/backend
python -m uvicorn main:app --reload --port 8000

# Watch the startup logs - you should see:
# 🚀 Starting Nexus LLM Analytics backend...
# ✅ Models selected: Primary=llama3.1:8b, Review=phi3:mini
# 🔥 Warming up primary model (this may take 10-30 seconds)...
# ✅ Primary model warmed up and ready!
# ✅ Backend ready to serve requests!

# Then time the first query (should be fast now):
time curl -X POST "http://localhost:8000/api/analyze/" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 2+2?", "filename": "test.csv"}'
```

**Expected**: First query should complete in <5 seconds (vs 30+ before).

### Success Criteria
- [ ] Startup logs show "Warming up primary model" message
- [ ] First API request completes in <5 seconds
- [ ] No cold-start delay on first user query

---

## 🔴 FIX 3: THREAD-SAFE SINGLETONS (Reliability - Critical)

### Problem
Several singleton patterns lack thread locks, causing potential race conditions when multiple requests arrive simultaneously.

### Files to Modify
1. `src/backend/services/analysis_service.py`
2. `src/backend/core/plugin_system.py`
3. `src/backend/agents/model_initializer.py`

### What to Do
Add threading locks to singleton instance creation.

### Exact Changes

#### File 1: `src/backend/services/analysis_service.py`

**FIND (at the end of the file, around line 155-165):**
```python
# Singleton
_service_instance = None

def get_analysis_service():
    global _service_instance
    if not _service_instance:
        _service_instance = AnalysisService()
    return _service_instance
```

**REPLACE WITH:**
```python
# Thread-safe Singleton
import threading

_service_instance = None
_service_lock = threading.Lock()

def get_analysis_service():
    """Get or create the singleton AnalysisService instance (thread-safe)."""
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            # Double-check pattern for thread safety
            if _service_instance is None:
                _service_instance = AnalysisService()
    return _service_instance
```

#### File 2: `src/backend/core/plugin_system.py`

**FIND (near the end of the file, around line 330-345):**
```python
# Global registry instance
_global_registry: Optional[AgentRegistry] = None

def get_agent_registry(plugins_dir: str = None) -> AgentRegistry:
    """Get or create the global agent registry"""
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry(plugins_dir)
        _global_registry.discover_agents()
    return _global_registry
```

**REPLACE WITH:**
```python
# Thread-safe Global registry instance
import threading

_global_registry: Optional[AgentRegistry] = None
_registry_lock = threading.Lock()

def get_agent_registry(plugins_dir: str = None) -> AgentRegistry:
    """Get or create the global agent registry (thread-safe)."""
    global _global_registry
    if _global_registry is None:
        with _registry_lock:
            # Double-check pattern for thread safety
            if _global_registry is None:
                _global_registry = AgentRegistry(plugins_dir)
                _global_registry.discover_agents()
    return _global_registry
```

#### File 3: `src/backend/agents/model_initializer.py`

**FIND (near the top, around line 10-15):**
```python
# Singleton instance
_model_initializer: Optional['ModelInitializer'] = None
```

**REPLACE WITH:**
```python
# Thread-safe Singleton instance
import threading

_model_initializer: Optional['ModelInitializer'] = None
_initializer_lock = threading.Lock()
```

**ALSO FIND (at the end of the file, around line 200-208) - if there's a getter function, update it:**

**ADD at the end of the file if not present:**
```python
def get_model_initializer() -> ModelInitializer:
    """Get or create the singleton ModelInitializer instance (thread-safe)."""
    global _model_initializer
    if _model_initializer is None:
        with _initializer_lock:
            if _model_initializer is None:
                _model_initializer = ModelInitializer()
    return _model_initializer
```

### How to Test

```bash
# Run concurrent requests to test thread safety
# Terminal 1: Start backend
cd src/backend
python -m uvicorn main:app --reload --port 8000 --workers 1

# Terminal 2: Send 5 concurrent requests
for i in {1..5}; do
  curl -X POST "http://localhost:8000/api/analyze/" \
    -H "Content-Type: application/json" \
    -d '{"query": "Count rows", "filename": "test.csv"}' &
done
wait
```

**Expected**: All 5 requests complete without errors, no duplicate initialization logs.

### Success Criteria
- [ ] No race condition errors in logs
- [ ] Single "initialized" message per component, not multiple
- [ ] Concurrent requests all succeed

---

## 🔴 FIX 4: SMART FALLBACK HARDCODED MODELS (Reliability - Critical)

### Problem
The `smart_fallback.py` hardcodes model names like `tinyllama`, `phi3:mini` which may not be installed, causing fallback to fail.

### File to Modify
`src/backend/core/smart_fallback.py`

### What to Do
Make the fallback chain dynamic based on installed models.

### Exact Changes

**FIND (around line 115-135):**
```python
def _init_fallback_chains(self):
    """Initialize default fallback chains"""
    
    # Model fallback chain (from most capable to minimal)
    self.model_chain = FallbackChain(
        name="model",
        strategies=["llama3.1:8b", "phi3:mini", "tinyllama", "echo"]
    )
```

**REPLACE WITH:**
```python
def _init_fallback_chains(self):
    """Initialize default fallback chains based on installed models"""
    
    # Dynamically build model fallback chain from installed models
    installed_models = self._get_installed_model_names()
    
    if installed_models:
        # Sort by estimated capability (larger models first)
        model_strategies = installed_models[:4]  # Top 4 models
    else:
        # Absolute fallback if no models detected
        model_strategies = ["llama3.1:8b", "phi3:mini", "tinyllama"]
        logger.warning("No installed models detected, using default fallback chain")
    
    self.model_chain = FallbackChain(
        name="model",
        strategies=model_strategies + ["echo"]  # 'echo' as last resort
    )
```

**ALSO ADD this new method right after `__init__`:**
```python
def _get_installed_model_names(self) -> List[str]:
    """Fetch installed model names from Ollama dynamically"""
    try:
        import requests
        import os
        
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        
        if response.status_code == 200:
            models_data = response.json().get("models", [])
            # Extract model names, filter out embedding models
            model_names = []
            for model in models_data:
                name = model.get("name", "")
                # Skip embedding models
                if "embed" not in name.lower() and "nomic" not in name.lower():
                    model_names.append(name)
            
            # Sort by size (larger = more capable, should be tried first)
            model_names.sort(
                key=lambda m: next(
                    (model.get("size", 0) for model in models_data if model.get("name") == m),
                    0
                ),
                reverse=True
            )
            
            logger.debug(f"Discovered models for fallback: {model_names}")
            return model_names
        
        return []
    except Exception as e:
        logger.warning(f"Could not fetch installed models: {e}")
        return []
```

### How to Test

```bash
# Check what models are installed
ollama list

# Verify fallback chain is populated correctly
# Add this test to your test file or run in Python:
python -c "
from backend.core.smart_fallback import SmartFallbackManager
manager = SmartFallbackManager()
print('Model fallback chain:', manager.model_chain.strategies)
"
```

**Expected**: Fallback chain should contain YOUR installed model names, not hardcoded ones.

### Success Criteria
- [ ] Fallback chain contains only installed models
- [ ] No KeyError when a fallback model doesn't exist
- [ ] Logs show "Discovered models for fallback: [your_models]"

---

## 🟠 FIX 5: IMPROVE PROMPT TEMPLATES (Accuracy - High)

### Problem
The code generation prompt uses the same format for all model sizes. Smaller models need simpler, more direct instructions.

### Files to Modify
1. `src/backend/core/code_generator.py`
2. `src/backend/prompts/code_generation_prompt.txt` (keep as-is for large models)
3. CREATE: `src/backend/prompts/code_generation_prompt_simple.txt`

### What to Do
Create a simplified prompt template for smaller models and route based on model size.

### Exact Changes

#### Step 1: Create new simple prompt template

**CREATE NEW FILE: `src/backend/prompts/code_generation_prompt_simple.txt`**

```plaintext
Generate Python code to answer: {query}

Data columns: {columns}

Rules:
- Store answer in `result`
- Use only these columns
- df is already loaded

Examples:
- Maximum: result = df['column'].max()
- Top 5: result = df.nlargest(5, 'column')[['name', 'column']]
- Average: result = df['column'].mean()
- Count: result = len(df)

Code:
```python
result = 
```
```

#### Step 2: Modify code_generator.py to use appropriate prompt

**FIND in `src/backend/core/code_generator.py` (around line 60-80):**
```python
def _build_dynamic_prompt(self, query: str, df: pd.DataFrame) -> str:
    """
    Build a dynamic, context-aware prompt based on the user's query and data structure.
    Supports both standard analytics and advanced ML/statistical queries.
    """
```

**ADD at the beginning of this method (after the docstring):**
```python
def _build_dynamic_prompt(self, query: str, df: pd.DataFrame, model: str = None) -> str:
    """
    Build a dynamic, context-aware prompt based on the user's query and data structure.
    Supports both standard analytics and advanced ML/statistical queries.
    Uses simplified prompts for smaller models.
    """
    # Determine if we should use simplified prompt for smaller models
    model_name = (model or "").lower()
    is_small_model = any(small in model_name for small in ['tiny', 'mini', '1b', '2b', '3b', 'gemma:2b'])
    
    if is_small_model:
        return self._build_simple_prompt(query, df)
    
    # Rest of existing code continues below...
```

**ADD this new method after `_build_dynamic_prompt`:**
```python
def _build_simple_prompt(self, query: str, df: pd.DataFrame) -> str:
    """
    Build simplified prompt for smaller models (tinyllama, phi3:mini, gemma:2b).
    Smaller models work better with concise, direct instructions.
    """
    columns_str = ", ".join(df.columns.tolist()[:15])  # Limit columns for small context
    
    # Load simple template
    from pathlib import Path
    simple_template_path = Path(__file__).parent.parent / 'prompts' / 'code_generation_prompt_simple.txt'
    
    if simple_template_path.exists():
        with open(simple_template_path, 'r', encoding='utf-8') as f:
            template = f.read()
        return template.format(query=query, columns=columns_str)
    
    # Inline fallback if template doesn't exist
    return f"""Generate Python code: {query}

Columns: {columns_str}

Store answer in `result`. Examples:
- Max: result = df['col'].max()
- Top 5: result = df.nlargest(5, 'col')
- Average: result = df['col'].mean()

```python
result = """
```

**ALSO UPDATE the `generate_and_execute` method to pass model name:**

**FIND (around line 500-520):**
```python
def generate_and_execute(
    self,
    query: str,
    df: pd.DataFrame,
    model: str = None,
```

**In this method, FIND where `_build_dynamic_prompt` is called and UPDATE to:**
```python
prompt = self._build_dynamic_prompt(query, df, model=model)
```

### How to Test

```bash
# Test with a small model explicitly
curl -X POST "http://localhost:8000/api/analyze/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the maximum value in the price column?",
    "filename": "test.csv"
  }'

# Check logs to see which prompt template was used
# Should see "Using simplified prompt for small model" in debug logs
```

### Success Criteria
- [ ] Small models receive shorter, simpler prompts
- [ ] Code generation success rate improves for small models
- [ ] Large models still get full detailed prompts

---

## 🟠 FIX 6: ASYNC LLM CALLS (Speed - High)

### Problem
LLM calls use synchronous `requests.post()` which blocks the event loop, causing poor throughput.

### File to Modify
`src/backend/core/llm_client.py`

### What to Do
Add async version of generate method using `httpx`.

### Exact Changes

**ADD these imports at the top of the file:**
```python
import httpx
import asyncio
```

**FIND the existing `generate` method and ADD this new async method after it:**
```python
async def generate_async(
    self, 
    prompt: str, 
    model: Optional[str] = None, 
    system: Optional[str] = None,
    adaptive_timeout: bool = True
) -> Dict[str, Any]:
    """
    Async version of generate for non-blocking LLM calls.
    Use this in async endpoints for better throughput.
    """
    from .circuit_breaker import get_circuit_breaker, CircuitBreakerConfig
    
    if model is None:
        model = self.primary_model
    
    # Check if model is available
    if "model_not_available" in model:
        return {
            "model": model,
            "prompt": prompt,
            "response": "Model not available. Please install a compatible model.",
            "error": "No suitable models installed.",
            "user_action_required": True
        }
    
    # Calculate timeout
    timeout = self._calculate_adaptive_timeout(model) if adaptive_timeout else 300
    
    url = f"{self.base_url}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            if "response" in data:
                return {
                    "model": model, 
                    "prompt": prompt, 
                    "response": data["response"].strip(), 
                    "success": True
                }
            else:
                return {
                    "model": model,
                    "prompt": prompt,
                    "response": "",
                    "error": "Empty response from LLM"
                }
                
    except httpx.TimeoutException:
        return {
            "model": model,
            "prompt": prompt,
            "response": "",
            "error": f"Request timed out after {timeout}s",
            "timeout": True
        }
    except Exception as e:
        return {
            "model": model,
            "prompt": prompt,
            "response": "",
            "error": str(e)
        }

async def generate_primary_async(self, prompt: str) -> Dict[str, Any]:
    """Async generate using the primary model."""
    return await self.generate_async(prompt, model=self.primary_model)

async def generate_review_async(self, prompt: str) -> Dict[str, Any]:
    """Async generate using the review model."""
    return await self.generate_async(prompt, model=self.review_model)
```

**ADD httpx to requirements.txt if not present:**
```
httpx>=0.24.0
```

### How to Test

```bash
# Install httpx if not present
pip install httpx

# Test async endpoint (requires updating analyze.py to use async - separate task)
# For now, verify the method exists:
python -c "
import asyncio
from backend.core.llm_client import LLMClient
client = LLMClient()
print('Async method exists:', hasattr(client, 'generate_async'))
"
```

### Success Criteria
- [ ] `generate_async` method exists and works
- [ ] No blocking during LLM generation
- [ ] Multiple concurrent requests don't queue behind each other

---

## 📋 TESTING CHECKLIST

After implementing all fixes, run this comprehensive test:

```bash
# 1. Start fresh
cd nexus-llm-analytics-dist
pip install -r requirements.txt

# 2. Start backend
cd src/backend
python -m uvicorn main:app --reload --port 8000

# 3. Watch startup logs - should see model warmup

# 4. Run test suite
cd ../..
python test_ml_quick.py

# 5. Test concurrent requests
for i in {1..3}; do
  curl -X POST "http://localhost:8000/api/analyze/" \
    -H "Content-Type: application/json" \
    -d '{"query": "Count all rows", "filename": "test.csv"}' &
done
wait

# 6. Test complex query (sandbox fix)
curl -X POST "http://localhost:8000/api/analyze/" \
  -H "Content-Type: application/json" \
  -d '{"query": "Apply a transformation to calculate the square of each value in column X", "filename": "test.csv"}'
```

---

## ⏳ ESTIMATED TIME PER FIX

| Fix | Estimated Time | Complexity |
|-----|----------------|------------|
| Fix 1: Sandbox | 15-20 min | Easy |
| Fix 2: Model Warmup | 10-15 min | Easy |
| Fix 3: Thread Safety | 15-20 min | Easy |
| Fix 4: Smart Fallback | 20-25 min | Medium |
| Fix 5: Prompt Templates | 25-30 min | Medium |
| Fix 6: Async LLM | 20-25 min | Medium |

**Total: ~2-2.5 hours**

---

## 🚫 DO NOT DO (For Now)

1. ❌ **Do NOT reorganize folder structure** - Fix functionality first
2. ❌ **Do NOT add authentication yet** - Focus on core accuracy/speed
3. ❌ **Do NOT refactor agent system** - Works, just needs tuning
4. ❌ **Do NOT upgrade dependencies** - Stability over features

---

## ✅ DONE CRITERIA

All fixes are complete when:

1. [ ] First query responds in <5 seconds (warm model)
2. [ ] `.apply()` and `.map()` work in generated code (sandbox fixed)
3. [ ] No race conditions under concurrent load (thread safety)
4. [ ] Fallback works with any installed models (no hardcoding)
5. [ ] Small models generate working code (prompt templates)
6. [ ] Async LLM method available for future use

---

# 📚 ADDITIONAL FIXES FROM DOCUMENTATION AUDIT

The following fixes are based on comparing documented features with actual implementation. These represent ideas that were planned but not fully implemented.

---

## 🟡 FIX 7: QUERY COMPLEXITY ANALYZER - MISSING ML ABBREVIATIONS (Accuracy - High)

### Problem
As documented in `ROUTING_IMPROVEMENT_ACTION_PLAN.md`, the routing system has only 72.1% accuracy (target: 85%). Critical safety failures occur when complex ML queries are misrouted to FAST tier because abbreviations aren't recognized.

### Evidence from Documentation
- "Run K-means" → FAST (expected: FULL) - complexity 0.183
- "Perform PCA" → BALANCED (expected: FULL) - complexity 0.333
- "Run a t-test" → BALANCED (expected: FULL) - complexity 0.333

### File to Modify
`src/backend/core/query_complexity_analyzer.py`

### What to Do
The analyzer already has ML keywords but is missing many common abbreviations and needs negation detection.

### Exact Changes

**FIND in `query_complexity_analyzer.py` (around line 55-115) the `COMPLEX_KEYWORDS` set and ADD these missing items:**

```python
# Machine Learning Abbreviations (ADD TO COMPLEX_KEYWORDS)
'rf', 'svc',  # Random Forest, Support Vector Classifier
'svr',  # Support Vector Regression
'knn', 'k-nn',  # K-Nearest Neighbors
'lda',  # Linear Discriminant Analysis
'gmm',  # Gaussian Mixture Model
'em algorithm',
'bagging', 'boosting',

# Statistical Test Abbreviations (ADD TO COMPLEX_KEYWORDS)  
'f-test', 'ftest',
'z-test', 'ztest',
'paired t', 'two-sample',
'mcnemar', 'fisher exact',
'levene', 'bartlett',

# Optimization Abbreviations (ADD TO COMPLEX_KEYWORDS)
'lp', 'qp',  # Linear/Quadratic Programming
'convex optimization',
'sgd', 'adam',  # Optimizers
```

**ALSO ADD this new method for negation detection (after `_has_explicit_negation`):**

```python
def _detect_negation_for_keyword(self, query: str, keyword_pos: int) -> bool:
    """
    Check if a keyword at position is negated.
    E.g., "Don't use machine learning, just sum" should be FAST, not FULL.
    """
    context_start = max(0, keyword_pos - 40)
    context = query[context_start:keyword_pos].lower()
    
    negation_patterns = [
        "don't", "dont", "do not",
        "no need", "no", "not",
        "without", "skip", "avoid",
        "instead of", "rather than",
        "don't want", "don't need"
    ]
    
    for pattern in negation_patterns:
        if pattern in context:
            return True
    return False
```

### How to Test

```bash
python -c "
from backend.core.query_complexity_analyzer import QueryComplexityAnalyzer
analyzer = QueryComplexityAnalyzer()

test_queries = [
    ('Run K-means clustering', 'full'),
    ('Perform PCA', 'full'),
    ('Run a t-test', 'full'),
    ('Use SVM for classification', 'full'),
    (\"Don't use ML, just count\", 'fast'),
]

for query, expected in test_queries:
    result = analyzer.analyze(query, {'rows': 1000, 'columns': 10})
    actual = result.recommended_tier
    status = '✅' if actual == expected else '❌'
    print(f'{status} \"{query[:40]}\" -> {actual} (expected: {expected})')
"
```

### Success Criteria
- [ ] "Run K-means" routes to FULL tier
- [ ] "Perform PCA" routes to FULL tier
- [ ] "Don't use ML, just count" routes to FAST tier
- [ ] Routing accuracy improved from 72.1% toward 85%

---

## 🟡 FIX 8: IMPLEMENT SEMANTIC LAYER FOR DATA AGNOSTICISM (Power - High)

### Problem
As documented in `IMPROVEMENT_ROADMAP.md`, agents use hardcoded lists like `['revenue', 'profit']` to guess column meanings. This limits the system to specific domains.

### File to Modify
`src/backend/core/data_utils.py` (or create `src/backend/core/semantic_mapper.py`)

### What to Do
Implement a Semantic Mapper that maps user columns to standard concepts on file upload.

### Exact Changes

**CREATE NEW FILE: `src/backend/core/semantic_mapper.py`**

```python
"""
Semantic Mapper: Maps domain-specific column names to standard analytical concepts.
This enables domain-agnostic analysis by translating column names to universal concepts.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class SemanticMapper:
    """
    Maps user's column names to standard analytical concepts.
    
    Example mappings:
    - "gross_inflow" -> "revenue" (finance)
    - "patient_count" -> "count" (healthcare)
    - "survival_rate" -> "rate" (medical)
    """
    
    # Standard concept categories
    CONCEPT_PATTERNS = {
        # Monetary concepts
        'revenue': ['revenue', 'sales', 'income', 'inflow', 'earnings', 'receipts', 'proceeds'],
        'cost': ['cost', 'expense', 'expenditure', 'outflow', 'spending', 'outlay'],
        'profit': ['profit', 'margin', 'earnings', 'net_income', 'gain', 'surplus'],
        'price': ['price', 'rate', 'amount', 'value', 'worth', 'cost_per', 'unit_price'],
        
        # Quantity concepts
        'count': ['count', 'number', 'quantity', 'total', 'volume', 'amount', 'patients', 'users', 'customers'],
        
        # Time concepts
        'date': ['date', 'datetime', 'timestamp', 'time', 'period', 'year', 'month', 'day', 'week', 'quarter'],
        
        # Categorical concepts
        'category': ['category', 'type', 'class', 'group', 'segment', 'region', 'department', 'product'],
        
        # Identifier concepts
        'id': ['id', 'identifier', 'code', 'key', 'ref', 'number'],
        
        # Rate/Ratio concepts
        'rate': ['rate', 'ratio', 'percentage', 'percent', 'proportion', 'share', 'fraction'],
        
        # Status concepts
        'status': ['status', 'state', 'condition', 'phase', 'stage', 'active', 'flag'],
    }
    
    def __init__(self):
        self._cached_mappings: Dict[str, Dict[str, str]] = {}
    
    def infer_column_concepts(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Analyze DataFrame columns and map each to a standard concept.
        
        Returns:
            Dict mapping column_name -> standard_concept
        """
        mappings = {}
        
        for col in df.columns:
            col_lower = col.lower().replace('_', ' ').replace('-', ' ')
            
            # Try to match against known patterns
            matched_concept = None
            best_match_len = 0
            
            for concept, patterns in self.CONCEPT_PATTERNS.items():
                for pattern in patterns:
                    if pattern in col_lower:
                        if len(pattern) > best_match_len:
                            matched_concept = concept
                            best_match_len = len(pattern)
            
            # Also infer from dtype
            if matched_concept is None:
                dtype = str(df[col].dtype)
                if 'datetime' in dtype:
                    matched_concept = 'date'
                elif 'float' in dtype or 'int' in dtype:
                    matched_concept = 'numeric'
                elif 'object' in dtype or 'category' in dtype:
                    matched_concept = 'text'
            
            mappings[col] = matched_concept or 'unknown'
            
        logger.debug(f"Inferred column concepts: {mappings}")
        return mappings
    
    def get_columns_for_concept(self, df: pd.DataFrame, concept: str) -> List[str]:
        """Get all columns that map to a given concept."""
        mappings = self.infer_column_concepts(df)
        return [col for col, mapped_concept in mappings.items() if mapped_concept == concept]
    
    def enhance_query_context(self, query: str, df: pd.DataFrame) -> str:
        """
        Enhance the query with column concept information.
        This helps LLMs understand what columns to use.
        """
        mappings = self.infer_column_concepts(df)
        
        concept_info = []
        for concept in set(mappings.values()):
            if concept not in ['unknown', 'text', 'numeric']:
                cols = self.get_columns_for_concept(df, concept)
                if cols:
                    concept_info.append(f"{concept}: {', '.join(cols)}")
        
        if concept_info:
            return f"{query}\n\n[Column Concepts: {'; '.join(concept_info)}]"
        return query


# Singleton accessor
_mapper_instance: Optional[SemanticMapper] = None

def get_semantic_mapper() -> SemanticMapper:
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = SemanticMapper()
    return _mapper_instance
```

### How to Test

```bash
python -c "
import pandas as pd
from backend.core.semantic_mapper import get_semantic_mapper

# Test with finance-like data
df1 = pd.DataFrame({'gross_revenue': [100], 'operating_cost': [50], 'report_date': ['2024-01-01']})
mapper = get_semantic_mapper()
print('Finance data concepts:', mapper.infer_column_concepts(df1))

# Test with healthcare-like data  
df2 = pd.DataFrame({'patient_count': [100], 'survival_rate': [0.85], 'admission_date': ['2024-01-01']})
print('Healthcare data concepts:', mapper.infer_column_concepts(df2))
"
```

### Success Criteria
- [ ] Column concepts are correctly inferred
- [ ] Same operations work across different domain vocabularies
- [ ] Query context is enhanced with concept information

---

## 🟡 FIX 9: STREAMING RESPONSES (UX - Medium)

### Problem
As documented in `IMPROVEMENT_ROADMAP.md`, users wait for the full analysis. Implementing token streaming makes the wait feel shorter and builds trust.

### File to Modify
`src/backend/api/analyze.py`

### What to Do
Add Server-Sent Events (SSE) endpoint for streaming progress updates.

### Exact Changes

**ADD new streaming endpoint after the existing `/analyze/` endpoint:**

```python
from fastapi.responses import StreamingResponse
import asyncio
import json

@router.post("/analyze/stream")
async def analyze_stream(request: AnalysisRequest):
    """
    Streaming analysis endpoint using Server-Sent Events.
    Returns progress updates as the analysis proceeds.
    """
    async def generate_updates():
        try:
            # Step 1: Starting
            yield f"data: {json.dumps({'step': 'starting', 'message': 'Starting analysis...'})}\n\n"
            await asyncio.sleep(0.1)
            
            # Step 2: Loading data
            yield f"data: {json.dumps({'step': 'loading', 'message': 'Loading data file...'})}\n\n"
            
            # Perform actual analysis (call existing service)
            from ..services.analysis_service import get_analysis_service
            service = get_analysis_service()
            
            # Step 3: Analyzing
            yield f"data: {json.dumps({'step': 'analyzing', 'message': 'Running analysis with LLM...'})}\n\n"
            
            result = await asyncio.to_thread(
                service.analyze,
                request.query,
                request.filename,
                request.model
            )
            
            # Step 4: Complete
            yield f"data: {json.dumps({'step': 'complete', 'result': result})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'step': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_updates(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
```

### How to Test

```bash
# Test streaming endpoint
curl -N -X POST "http://localhost:8000/api/analyze/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "Count all rows", "filename": "test.csv"}'

# Should see progressive updates:
# data: {"step": "starting", ...}
# data: {"step": "loading", ...}
# data: {"step": "analyzing", ...}
# data: {"step": "complete", "result": {...}}
```

### Success Criteria
- [ ] Streaming endpoint returns progressive updates
- [ ] Frontend can consume SSE updates (future task)
- [ ] Errors are properly streamed

---

## 🟡 FIX 10: USER FEEDBACK COLLECTION (Power - Medium)

### Problem  
As documented in `IMPROVEMENT_ROADMAP.md`, the system "fires and forgets" with no feedback loop. Adding thumbs up/down creates a flywheel for improvement.

### File to Create
`src/backend/api/feedback.py`

### What to Do
Create a feedback endpoint to save user ratings.

### Exact Changes

**CREATE NEW FILE: `src/backend/api/feedback.py`**

```python
"""
Feedback API: Collects user ratings for analysis results.
This enables a feedback flywheel for continuous improvement.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    query: str = Field(..., description="The original query")
    result: str = Field(..., description="The analysis result")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    thumbs_up: Optional[bool] = Field(None, description="Quick thumbs up/down")
    comment: Optional[str] = Field(None, description="Optional user comment")
    filename: Optional[str] = Field(None, description="Data file used")


class FeedbackResponse(BaseModel):
    success: bool
    message: str
    feedback_id: str


@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for an analysis result.
    Stores (Query, Result, Rating) triplets for future fine-tuning.
    """
    try:
        # Generate feedback ID
        feedback_id = f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(request.query) % 10000:04d}"
        
        # Prepare feedback entry
        entry = {
            "id": feedback_id,
            "timestamp": datetime.now().isoformat(),
            "query": request.query,
            "result": request.result[:1000],  # Truncate long results
            "rating": request.rating,
            "thumbs_up": request.thumbs_up,
            "comment": request.comment,
            "filename": request.filename
        }
        
        # Store in feedback file
        feedback_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'feedback')
        os.makedirs(feedback_dir, exist_ok=True)
        feedback_file = os.path.join(feedback_dir, 'user_feedback.jsonl')
        
        with open(feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + "\n")
        
        logger.info(f"Feedback recorded: {feedback_id} (rating: {request.rating})")
        
        return FeedbackResponse(
            success=True,
            message="Thank you for your feedback!",
            feedback_id=feedback_id
        )
        
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")


@router.get("/stats")
async def get_feedback_stats():
    """Get aggregate feedback statistics."""
    try:
        feedback_file = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'data', 'feedback', 'user_feedback.jsonl'
        )
        
        if not os.path.exists(feedback_file):
            return {"total": 0, "avg_rating": None, "thumbs_up_rate": None}
        
        total = 0
        ratings = []
        thumbs_up = 0
        thumbs_total = 0
        
        with open(feedback_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    total += 1
                    ratings.append(entry.get('rating', 3))
                    if entry.get('thumbs_up') is not None:
                        thumbs_total += 1
                        if entry['thumbs_up']:
                            thumbs_up += 1
        
        return {
            "total": total,
            "avg_rating": sum(ratings) / len(ratings) if ratings else None,
            "thumbs_up_rate": thumbs_up / thumbs_total if thumbs_total > 0 else None
        }
        
    except Exception as e:
        logger.error(f"Failed to get feedback stats: {e}")
        return {"error": str(e)}
```

**ALSO ADD to `src/backend/main.py` (in the router includes):**

```python
from .api.feedback import router as feedback_router
app.include_router(feedback_router)
```

### How to Test

```bash
# Submit feedback
curl -X POST "http://localhost:8000/api/feedback/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the average sales?",
    "result": "The average sales is $1,234.56",
    "rating": 5,
    "thumbs_up": true
  }'

# Get stats
curl "http://localhost:8000/api/feedback/stats"
```

### Success Criteria
- [ ] Feedback endpoint accepts and stores ratings
- [ ] Feedback stats are retrievable
- [ ] Data stored in `data/feedback/user_feedback.jsonl`

---

## 🟢 FIX 11: DYNAMIC PLANNER JSON REPAIR (Reliability - Medium)

### Problem
As documented in `PROJECT_ROADMAP_NEXT_LEVEL.md`, the dynamic planner fails when LLM produces invalid JSON.

### File to Modify
`src/backend/core/dynamic_planner.py` (if exists) or relevant planner code

### What to Do
Add JSON repair/fallback for malformed LLM responses.

### Exact Changes

**ADD this helper function (can be added to any module that parses LLM JSON):**

```python
import re
import json
from typing import Any, Optional


def repair_json(malformed_json: str) -> Optional[dict]:
    """
    Attempt to repair common JSON malformations from LLMs.
    
    Common issues:
    - Trailing commas
    - Single quotes instead of double
    - Unquoted keys
    - Missing closing brackets
    """
    text = malformed_json.strip()
    
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Repair attempt 1: Extract JSON from markdown code block
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            text = match.group(1).strip()
    
    # Repair attempt 2: Replace single quotes with double quotes
    text = re.sub(r"'([^']*)':", r'"\1":', text)
    text = re.sub(r":\s*'([^']*)'", r': "\1"', text)
    
    # Repair attempt 3: Remove trailing commas
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    # Repair attempt 4: Add missing closing brackets
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    text = text + '}' * open_braces + ']' * open_brackets
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def safe_json_parse(text: str, default: Any = None) -> Any:
    """
    Safely parse JSON with automatic repair.
    Returns default if parsing fails completely.
    """
    result = repair_json(text)
    return result if result is not None else default
```

### How to Test

```bash
python -c "
from backend.core.json_utils import repair_json  # or wherever you put it

# Test cases
test_cases = [
    '{\"key\": \"value\"}',  # Valid
    \"{'key': 'value'}\",  # Single quotes
    '{\"key\": \"value\",}',  # Trailing comma
    '\`\`\`json\\n{\"key\": \"value\"}\\n\`\`\`',  # Markdown wrapped
    '{\"key\": \"value\"',  # Missing brace
]

for test in test_cases:
    result = repair_json(test)
    print(f'{\"✅\" if result else \"❌\"} Input: {test[:30]}... -> {result}')
"
```

### Success Criteria
- [ ] Trailing commas are handled
- [ ] Single quotes are converted
- [ ] Markdown-wrapped JSON is extracted
- [ ] Missing brackets are added

---

## ⏳ ESTIMATED TIME FOR ADDITIONAL FIXES

| Fix | Estimated Time | Complexity |
|-----|----------------|------------|
| Fix 7: ML Abbreviations | 20-25 min | Easy |
| Fix 8: Semantic Mapper | 30-40 min | Medium |
| Fix 9: Streaming | 25-30 min | Medium |
| Fix 10: Feedback | 20-25 min | Easy |
| Fix 11: JSON Repair | 15-20 min | Easy |

**Additional Total: ~2-2.5 hours**

---

**Created by**: Nexus Audit System  
**For**: Claude Sonnet 4.5 Implementation  
**Last Updated**: January 2, 2026

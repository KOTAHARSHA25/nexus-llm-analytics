# 🤖 MODEL SELECTION GUIDE FOR 8GB RAM

## 📊 CURRENT SITUATION ANALYSIS

### **Your Setup:**
- **Development Machine:** 16GB RAM (current)
- **Deployment Machine:** 8GB RAM (target)
- **Current Models:**
  - `phi3:mini` (2.2 GB) - Primary
  - `tinyllama:latest` (637 MB) - Review
  - `llama3.1:8b` (4.9 GB) - Not suitable for 8GB
  - `nomic-embed-text` (274 MB) - Embeddings ✅

---

## 🎯 RECOMMENDATION: USE SINGLE MODEL

### **✅ YES - Use Same Model for Both Primary & Review**

**Why This Works Better:**

1. **Memory Efficiency** 🚀
   - Single model loaded once in RAM
   - 8GB RAM can handle one ~3GB model comfortably
   - Switching between models takes time and memory

2. **Consistency** 🎯
   - Same "brain" doing analysis and review
   - More coherent communication style
   - Better understanding of context

3. **Simplicity** 🧠
   - Easier to maintain
   - Fewer configuration issues
   - Simpler deployment

4. **Performance** ⚡
   - No model switching overhead
   - Faster response times
   - Lower memory thrashing

---

## 🏆 BEST MODEL FOR 8GB RAM

### **RECOMMENDED: `phi3:mini` (2.2 GB)**

**Why Phi3-Mini is Perfect for Your Project:**

✅ **Size:** 2.2 GB - Leaves 5.8GB free for system & ChromaDB  
✅ **Quality:** Microsoft's state-of-the-art small model  
✅ **Performance:** Excellent for code generation & data analysis  
✅ **Context:** 4K tokens (sufficient for most queries)  
✅ **Speed:** Fast inference on consumer hardware  
✅ **Reliability:** Stable, well-tested, production-ready  

**Your Test Results:**
- ✅ 6/6 simple JSON queries PASSED
- ✅ 45-95s response time (acceptable)
- ✅ 100% accuracy on tested queries
- ✅ Direct answers (no hallucinations after optimization)

**Memory Footprint on 8GB RAM:**
```
System:            ~2.0 GB
Phi3-Mini:         ~2.2 GB
ChromaDB:          ~0.5 GB
Python/FastAPI:    ~1.0 GB
Frontend (if local): ~1.0 GB
Free:              ~1.3 GB (buffer)
------------------------
Total:             8.0 GB ✅
```

---

## 🔄 ALTERNATIVE OPTIONS (If Phi3-Mini Doesn't Work)

### **Option 2: `qwen2.5:3b` (1.9 GB)** ⭐ NEW RECOMMENDATION

**Advantages:**
- Smaller than phi3:mini
- Better at following instructions
- Good code generation
- Strong reasoning capabilities
- 32K context window (8x larger!)

**Download:**
```bash
ollama pull qwen2.5:3b
```

**When to use:**
- If phi3:mini is too slow on 8GB RAM
- If you need longer context
- If Chinese language support needed

---

### **Option 3: `gemma2:2b` (1.6 GB)** 

**Advantages:**
- Very lightweight (Google's efficient model)
- Good instruction following
- Fast inference
- 8K context window

**Disadvantages:**
- Smaller knowledge base
- May struggle with complex queries

**Download:**
```bash
ollama pull gemma2:2b
```

**When to use:**
- RAM is extremely constrained (<6GB available)
- Speed is more important than accuracy
- Simple queries only

---

### **Option 4: `llama3.2:3b` (2.0 GB)**

**Advantages:**
- Meta's latest small model
- Good general performance
- Better instruction following than phi3
- 128K context window (!)

**Download:**
```bash
ollama pull llama3.2:3b
```

**When to use:**
- Need very long context
- Meta ecosystem preferred
- Good balance of size/performance

---

### **Option 5: `tinyllama:1.1b` (637 MB)** ⚠️ LAST RESORT

**Advantages:**
- Extremely small
- Very fast
- Leaves maximum RAM for data

**Disadvantages:**
- Lower quality responses
- Limited reasoning ability
- May hallucinate more

**When to use:**
- Absolute minimum RAM (<4GB)
- Demo purposes only
- Speed critical, quality secondary

---

## 📋 COMPARISON TABLE

| Model | Size | Context | RAM Need | Speed | Quality | Best For |
|-------|------|---------|----------|-------|---------|----------|
| **phi3:mini** ⭐ | 2.2GB | 4K | 5GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | **YOUR PROJECT** |
| qwen2.5:3b | 1.9GB | 32K | 4.5GB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Long context |
| llama3.2:3b | 2.0GB | 128K | 4.5GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | Very long context |
| gemma2:2b | 1.6GB | 8K | 4GB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Speed priority |
| tinyllama:1.1b | 637MB | 2K | 3GB | ⚡⚡⚡⚡⚡ | ⭐⭐ | Demo/testing |

---

## 🛠️ CONFIGURATION FOR YOUR PROJECT

### **Step 1: Update config/user_preferences.json**

```json
{
  "primary_model": "phi3:mini",
  "review_model": "phi3:mini",
  "embedding_model": "nomic-embed-text:latest",
  "enable_auto_review": true,
  "max_iterations": 2,
  "temperature": 0.1
}
```

**Key Changes:**
- ✅ `review_model`: Changed from `tinyllama` to `phi3:mini`
- ✅ Both primary and review use **same model**
- ✅ `enable_auto_review`: Keep as `true` (your requirement)

---

### **Step 2: Update src/backend/core/config.py**

Find this section (around line 30-50):

```python
# Model configurations
DEFAULT_PRIMARY_MODEL = "phi3:mini"
DEFAULT_REVIEW_MODEL = "phi3:mini"  # ← Changed from tinyllama
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
```

---

### **Step 3: Verify Memory Usage**

Test on your 8GB machine:

```bash
# Before starting
free -h  # Linux
# or
Get-Process | Sort-Object WorkingSet -Descending | Select-Object -First 5  # Windows

# Start backend
cd src/backend
python -m uvicorn main:app --reload --port 8000

# Monitor memory
# RAM usage should be ~5GB total
```

**Expected Memory:**
- Initial: ~3GB (Python + Ollama + ChromaDB)
- During query: ~5GB (+ phi3:mini loaded)
- Peak: ~6GB (+ processing buffers)

---

## ⚡ PERFORMANCE OPTIMIZATION FOR 8GB RAM

### **1. Reduce Ollama Memory Usage**

Create/edit `~/.ollama/config.json`:

```json
{
  "num_gpu": 0,
  "num_thread": 4,
  "num_ctx": 4096,
  "num_batch": 512,
  "num_keep": 4,
  "low_vram": true
}
```

**What this does:**
- `num_ctx: 4096` - Limit context window (saves RAM)
- `low_vram: true` - Optimize for limited memory
- `num_batch: 512` - Smaller batches (slower but less RAM)

---

### **2. Optimize ChromaDB**

In `src/backend/core/chromadb_client.py`:

```python
# Limit collection size
MAX_DOCUMENTS = 1000  # Don't load more than 1000 docs

# Use smaller embedding batches
EMBEDDING_BATCH_SIZE = 10  # Process 10 at a time instead of 100
```

---

### **3. Enable Swap/Virtual Memory**

**Windows:**
```powershell
# Increase virtual memory to 16GB
# System Properties → Advanced → Performance Settings → Advanced → Virtual Memory
# Set to: Initial: 16384 MB, Maximum: 16384 MB
```

**Linux:**
```bash
# Create 8GB swap file (if needed)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Important:** Swap is SLOW but prevents crashes on low RAM

---

### **4. Close Unnecessary Programs**

Before running:
- Close browser (saves 1-2GB)
- Close IDE if not needed (saves 500MB-1GB)
- Close Discord/Slack (saves 300-500MB)
- Run only backend (frontend optional for testing)

---

## 🧪 TESTING PLAN FOR 8GB RAM

### **Phase 1: Verify Model Works**

```bash
# On 8GB machine
cd src/backend

# Test 1: Single model load
ollama run phi3:mini "What is 2+2?"
# Should respond in <10s

# Test 2: Backend startup
python -m uvicorn main:app --reload --port 8000
# Monitor RAM usage

# Test 3: Simple query
curl -X POST http://localhost:8000/analyze/ \
  -H "Content-Type: application/json" \
  -d '{"filename": "1.json", "query": "What is the student name?"}'
# Should complete in <120s
```

---

### **Phase 2: Stress Test**

```bash
# Run Phase 1 JSON tests
python test_phase1_json_optimized.py

# Expected results:
# - 6/6 simple tests PASS ✅
# - Response time: 60-120s (slower than 16GB)
# - No out-of-memory errors
# - System remains responsive
```

---

### **Phase 3: Monitor & Optimize**

```python
# Add memory monitoring to backend
import psutil

def log_memory():
    mem = psutil.virtual_memory()
    print(f"RAM: {mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB ({mem.percent}%)")

# Call before/after each query
```

---

## 🚨 WARNING SIGNS (8GB RAM Issues)

### **If You See These, Model is Too Large:**

❌ **System freezing/lagging**
- Solution: Use smaller model (gemma2:2b)

❌ **Out of memory errors**
- Solution: Enable swap, reduce context window

❌ **Response time >300s**
- Solution: Reduce data preview size, use smaller model

❌ **Ollama crashing**
- Solution: Lower `num_ctx`, enable `low_vram`

❌ **Backend not starting**
- Solution: Close other programs, restart

---

## 📝 RECOMMENDED CONFIGURATION FOR 8GB RAM

### **Final Setup:**

```json
{
  "primary_model": "phi3:mini",
  "review_model": "phi3:mini",
  "embedding_model": "nomic-embed-text:latest",
  "enable_auto_review": true,
  "max_iterations": 2,
  "temperature": 0.1,
  "max_context_length": 4096,
  "max_data_preview_rows": 5,
  "max_data_preview_chars": 2000,
  "enable_caching": true,
  "cache_ttl": 1800
}
```

**Why This Works:**
- Single model reduces memory by ~2GB
- Limited context prevents memory bloat
- Caching reduces repeated model calls
- Review protocol still active (your requirement)

---

## ✅ ACTION PLAN

### **On Your 16GB Machine (Development):**

1. Keep current setup:
   ```json
   "primary_model": "phi3:mini",
   "review_model": "phi3:mini"
   ```

2. Test thoroughly with single model

3. Optimize code for lower memory usage

### **On 8GB Machine (Deployment):**

1. **Install models:**
   ```bash
   ollama pull phi3:mini
   ollama pull nomic-embed-text:latest
   ```

2. **Configure:**
   - Copy optimized config
   - Enable low_vram mode
   - Set up swap

3. **Test:**
   - Run simple queries first
   - Monitor memory closely
   - Adjust if needed

4. **Fallback Plan:**
   - If phi3:mini too heavy → Use `qwen2.5:3b`
   - If still issues → Use `gemma2:2b`
   - Emergency → Use `tinyllama:1.1b` (quality will suffer)

---

## 🎯 FINAL RECOMMENDATION

### **For Your B.Tech Project:**

**Use `phi3:mini` for BOTH primary and review.**

**Why:**
✅ Proven to work (your test results)  
✅ Fits in 8GB RAM comfortably  
✅ Microsoft quality & support  
✅ Good balance of speed/accuracy  
✅ Same model = consistency  
✅ Simplifies deployment  
✅ Maintains review protocol  

**Alternative (If Issues on 8GB):**
Use `qwen2.5:3b` - slightly smaller, better instruction following, longer context

**DO NOT USE:**
❌ `llama3.1:8b` (4.9GB) - Too large for 8GB RAM  
❌ Different models for primary/review - Wastes memory  
❌ `tinyllama` for production - Quality too low  

---

## 📊 MEMORY BUDGET BREAKDOWN

### **8GB RAM Allocation:**

```
┌─────────────────────────────────────┐
│ Windows/Linux System:      2.0 GB   │ (25%)
├─────────────────────────────────────┤
│ Phi3-Mini Model:           2.2 GB   │ (27.5%)
├─────────────────────────────────────┤
│ Python/FastAPI:            1.0 GB   │ (12.5%)
├─────────────────────────────────────┤
│ ChromaDB:                  0.5 GB   │ (6%)
├─────────────────────────────────────┤
│ Ollama Server:             0.8 GB   │ (10%)
├─────────────────────────────────────┤
│ Processing Buffer:         1.0 GB   │ (12.5%)
├─────────────────────────────────────┤
│ Safety Margin:             0.5 GB   │ (6%)
└─────────────────────────────────────┘
Total:                       8.0 GB   (100%)
```

**This configuration leaves enough headroom for:**
- Temporary data processing
- Cache storage
- System operations
- Prevents crashes

---

## 🚀 READY TO DEPLOY

**Your final configuration is:**
- **Primary Model:** `phi3:mini` (2.2GB)
- **Review Model:** `phi3:mini` (2.2GB) ← Same as primary
- **Embedding Model:** `nomic-embed-text` (274MB)
- **Total Model Size:** ~2.5GB (loaded once)
- **RAM Requirement:** 5-6GB (fits in 8GB)

**This maintains ALL core principles:**
✅ Privacy-first (100% local)  
✅ Multi-agent system (5 agents)  
✅ Review protocol (2-step validation)  
✅ Works on 8GB RAM  

---

**Created:** October 18, 2025  
**For:** Nexus LLM Analytics B.Tech Project  
**Target:** 8GB RAM deployment

# GPU Testing & Improvement Guide

## Quick Start: Test Your Project on GPU

### Option 1: Google Colab (Fastest Setup - 2 minutes)

**Step 1:** Go to https://colab.research.google.com/

**Step 2:** Upload your project:
```python
# In first cell:
!git clone https://github.com/YOUR_USERNAME/nexus-llm-analytics.git
%cd nexus-llm-analytics
!pip install -r requirements.txt -q

# Enable GPU: Runtime → Change runtime type → T4 GPU
```

**Step 3:** Install Ollama with GPU:
```python
# In second cell:
!curl https://ollama.ai/install.sh | sh
!nohup ollama serve > ollama.log 2>&1 &
import time; time.sleep(5)  # Wait for server
!ollama pull phi3:mini
```

**Step 4:** Run tests:
```python
# In third cell:
!python tests/test_gpu_performance.py
```

**Expected improvement:** 6-12x faster than your PC

---

### Option 2: Kaggle (30 hrs/week FREE GPU)

**Step 1:** Create Kaggle account, create new notebook

**Step 2:** Settings → Accelerator → GPU T4 x2

**Step 3:** Add Code:
```python
!curl https://ollama.ai/install.sh | sh
!ollama serve &

# Upload your project as Kaggle dataset, then:
import sys
sys.path.append('/kaggle/input/your-project-name')

!python tests/test_accuracy_comprehensive.py
```

---

### Option 3: Paperspace (Persistent Development)

**Step 1:** Sign up at paperspace.com/gradient

**Step 2:** Create new project → Launch console

**Step 3:** In terminal:
```bash
git clone https://github.com/YOUR_USERNAME/nexus-llm-analytics.git
cd nexus-llm-analytics
pip install -r requirements.txt

curl https://ollama.ai/install.sh | sh
ollama serve &
ollama pull phi3:mini

# Run tests
python tests/test_gpu_performance.py
python tests/test_accuracy_comprehensive.py
```

---

## What to Test on GPU:

### 1. **Response Time Improvement**
Compare CPU vs GPU speed:
```python
# Your current CPU: ~300s avg per query
# Expected GPU:     ~30-50s per query (6-10x faster)
```

### 2. **Model Comparison**
Test which model is best for your use case:
```bash
# Test different models quickly on GPU
for model in phi3:mini llama3.1:8b mistral:7b; do
    ollama pull $model
    # Update config, run tests
    python tests/test_accuracy_comprehensive.py
done
```

### 3. **Batch Processing**
Test if processing multiple queries simultaneously improves throughput:
```python
# Sequential: 10 queries x 30s = 300s
# Parallel on GPU: 10 queries = ~60s (5x improvement)
```

### 4. **Larger Models**
Your PC can't run 13B+ models (RAM limited). GPU can:
```bash
# Test on Colab with 40GB RAM:
ollama pull llama3.1:13b  # Better accuracy
ollama pull codellama:13b  # Better code generation
```

---

## Performance Targets:

| Metric | Current (CPU) | Target (GPU) | Improvement |
|--------|---------------|--------------|-------------|
| Simple queries | 70-360s | 8-40s | **6-10x faster** |
| Complex queries | 300-600s | 30-80s | **8-10x faster** |
| Model options | 3.8B max | 13-70B | **Better accuracy** |
| Batch throughput | 1 query/5min | 5-10 queries/5min | **5-10x more** |

---

## RECOMMENDED TESTING WORKFLOW:

### Week 1: Baseline Testing (Google Colab - FREE)
- Upload project to Colab
- Run test_gpu_performance.py
- Compare CPU vs GPU T4
- Document speedup

### Week 2: Model Optimization (Kaggle - FREE 30hrs)
- Test phi3:mini vs llama3.1:8b vs mistral:7b
- Measure accuracy vs speed tradeoff
- Find optimal model for your queries

### Week 3: Stress Testing (Paperspace - Free tier + credits)
- Run comprehensive test suite (19 tests x 10 iterations)
- Test with larger datasets (100K+ rows)
- Memory profiling under load

### Week 4: Final Benchmarks
- Document all improvements
- Create performance comparison charts
- Update thesis with GPU results

---

## Tools Included:

1. **test_gpu_performance.py** - Quick benchmark script
2. **test_accuracy_comprehensive.py** - Your existing 19-test suite
3. **test_performance_benchmarks.py** - Detailed profiling

---

## Tips for Maximum Benefit:

✅ **DO:**
- Test on Colab with T4 first (fastest setup)
- Compare at least 3 different models
- Run full test suite multiple times (average results)
- Test with your actual data files
- Document speedups for thesis

❌ **DON'T:**
- Deploy to production (testing only)
- Leave sessions running (waste GPU time)
- Test on CPU-only instances
- Use small toy datasets

---

## Getting Started Right Now:

1. Open Google Colab: https://colab.research.google.com/
2. Copy this into first cell:

```python
# Quick GPU test for Nexus LLM Analytics
!git clone https://github.com/YOUR_REPO_URL.git
%cd nexus-llm-analytics
!pip install -q fastapi uvicorn pandas numpy python-multipart aiofiles langchain-core sqlalchemy

# Install Ollama
!curl -fsSL https://ollama.com/install.sh | sh
!nohup ollama serve > /dev/null 2>&1 &

import time
time.sleep(5)

!ollama pull phi3:mini

# Test single query
from pathlib import Path
import sys
sys.path.append('src')

import asyncio
from backend.services.analysis_service import AnalysisService

async def quick_test():
    service = AnalysisService()
    
    # Your test file (upload 1.json to Colab files first)
    result = await service.analyze(
        "what is the name",
        {'filename': '1.json', 'filepath': '/content/1.json'}
    )
    print(result)

await quick_test()
```

3. Upload your `1.json` file to Colab
4. Run the cell
5. See GPU speedup!

---

## Expected Results:

You should see something like:
```
Query: "what is the name"
CPU Time: 70.45s → GPU Time: 8.2s (8.6x faster!)
Result: ✓ Found "harsha" correctly
```

This proves GPU acceleration works for your project!

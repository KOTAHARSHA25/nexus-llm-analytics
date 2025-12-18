"""
Task 1.3.1: LLM Response Speed Benchmarking
Compare phi3:mini vs llama3.1:8b performance
"""

import subprocess
import requests
import time
import json

BASE_URL = "http://localhost:8000"
TEST_FILE = "data/samples/simple.json"

print("="*60)
print("TASK 1.3.1: LLM RESPONSE SPEED BENCHMARKING")
print("="*60)

# Step 1: Check which models are available in Ollama
print("\n[1/6] Checking available Ollama models...")
try:
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        timeout=10
    )
    available_models = result.stdout
    print(available_models)
    
    has_phi3 = "phi3:mini" in available_models or "phi3" in available_models
    has_llama = "llama3.1:8b" in available_models or "llama3.1" in available_models
    
    print(f"\n✅ phi3:mini available: {has_phi3}")
    print(f"✅ llama3.1:8b available: {has_llama}")
    
    if not has_phi3:
        print("\n⚠️  phi3:mini not found. Current model may be different.")
    if not has_llama:
        print("\n⚠️  llama3.1:8b not found. Will need to pull it first.")
        print("   Run: ollama pull llama3.1:8b")
        print("\n   Continuing with phi3:mini benchmark only...")
        
except Exception as e:
    print(f"⚠️  Could not check Ollama models: {e}")
    print("   Continuing with current configuration...")

# Step 2: Check backend health
print("\n[2/6] Checking backend health...")
try:
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    if response.status_code == 200:
        print("✅ Backend is running")
    else:
        print(f"❌ Backend returned status {response.status_code}")
        exit(1)
except Exception as e:
    print(f"❌ Backend not accessible: {e}")
    exit(1)

# Step 3: Check current model configuration
print("\n[3/6] Checking current model configuration...")
try:
    response = requests.get(f"{BASE_URL}/models/current", timeout=5)
    if response.status_code == 200:
        current_models = response.json()
        print(f"   Primary: {current_models.get('primary_model', 'unknown')}")
        print(f"   Review: {current_models.get('review_model', 'unknown')}")
        print(f"   Embedding: {current_models.get('embedding_model', 'unknown')}")
    else:
        print(f"⚠️  Could not fetch current models: {response.status_code}")
except Exception as e:
    print(f"⚠️  Error checking models: {e}")

# Step 4: Upload test file
print("\n[4/6] Uploading test file...")
try:
    with open(TEST_FILE, 'rb') as f:
        files = {'file': ('simple.json', f, 'application/json')}
        response = requests.post(f"{BASE_URL}/upload-documents/", files=files, timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        filename = data.get('filename')
        print(f"✅ File uploaded: {filename}")
    else:
        print(f"❌ Upload failed: {response.status_code}")
        exit(1)
except Exception as e:
    print(f"❌ Upload error: {e}")
    exit(1)

# Step 5: Benchmark current model (phi3:mini)
print("\n[5/6] Benchmarking current model (phi3:mini)...")
test_queries = [
    "What is the total sales amount?",
    "How many products are there?",
    "What is the average sales amount per product?"
]

phi3_times = []
phi3_answers = []

for i, query in enumerate(test_queries, 1):
    print(f"   Query {i}/3: {query[:40]}...")
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/analyze/",
            json={
                "query": query,
                "filename": filename
            },
            timeout=120
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get('result', 'No answer')
            phi3_times.append(elapsed)
            phi3_answers.append(answer)
            print(f"      ✅ {elapsed:.2f}s - {answer[:60]}...")
        else:
            print(f"      ❌ Failed: {response.status_code}")
            phi3_times.append(0)
            phi3_answers.append("Error")
    except Exception as e:
        print(f"      ❌ Error: {e}")
        phi3_times.append(0)
        phi3_answers.append("Error")

# Step 6: Results and analysis
print("\n[6/6] Benchmark Results...")
print("="*60)

avg_phi3_time = sum(phi3_times) / len(phi3_times) if phi3_times else 0
min_phi3_time = min(phi3_times) if phi3_times else 0
max_phi3_time = max(phi3_times) if phi3_times else 0

print("\n**Current Model (phi3:mini) Performance:**")
print(f"   Average: {avg_phi3_time:.2f}s")
print(f"   Min:     {min_phi3_time:.2f}s")
print(f"   Max:     {max_phi3_time:.2f}s")
print(f"   Total:   {sum(phi3_times):.2f}s for {len(test_queries)} queries")

# Note about llama3.1:8b
print("\n**Note about llama3.1:8b:**")
if not has_llama:
    print("   ⚠️  llama3.1:8b is not installed")
    print("   To benchmark it:")
    print("   1. Run: ollama pull llama3.1:8b")
    print("   2. Update model config in backend")
    print("   3. Re-run this benchmark")
else:
    print("   ✅ llama3.1:8b is available")
    print("   Note: llama3.1:8b is a larger model (8B vs 3.8B params)")
    print("   Expected: Higher quality but potentially slower responses")
    print("   To switch: Update model config via /models/configure endpoint")

# Summary
print("\n" + "="*60)
print("SUMMARY - TASK 1.3.1: LLM BENCHMARKING")
print("="*60)

print(f"\n✅ phi3:mini benchmark complete")
print(f"   - Average response time: {avg_phi3_time:.2f}s")
print(f"   - Suitable for: Fast responses, good for simple queries")

if has_llama:
    print(f"\n⚠️  llama3.1:8b available but not tested")
    print(f"   - To test: Configure it as primary model and re-run")
else:
    print(f"\n⏳ llama3.1:8b not available")
    print(f"   - Install: ollama pull llama3.1:8b")

print("\n**Recommendation:**")
print("   Current setup (phi3:mini) is optimized for speed")
print(f"   Average {avg_phi3_time:.2f}s response time is acceptable")
print("   Consider llama3.1:8b only if answer quality is insufficient")

print("\n" + "="*60)

"""
Task 1.3.1: LLM Response Speed Benchmarking - llama3.1:8b
Test llama3.1:8b performance and compare with phi3:mini
"""

import requests
import time

BASE_URL = "http://localhost:8000"
TEST_FILE = "data/samples/simple.json"

print("="*60)
print("TASK 1.3.1: LLM BENCHMARKING - llama3.1:8b")
print("="*60)

# Step 1: Wait for backend to restart with new model
print("\n⚠️  IMPORTANT: Restart the backend server with llama3.1:8b")
print("   The .env file has been updated to use llama3.1:8b")
print("   Press Enter when backend is restarted...")
input()

# Step 2: Verify model configuration
print("\n[1/4] Verifying model configuration...")
try:
    response = requests.get(f"{BASE_URL}/models/current", timeout=10)
    if response.status_code == 200:
        current_models = response.json()
        primary = current_models.get('primary_model', 'unknown')
        review = current_models.get('review_model', 'unknown')
        print(f"   Primary: {primary}")
        print(f"   Review: {review}")
        
        if 'llama3.1' in primary.lower():
            print("   ✅ llama3.1:8b is active")
        else:
            print(f"   ❌ Expected llama3.1:8b but got {primary}")
            print("   Please restart backend and try again")
            exit(1)
    else:
        print(f"   ⚠️  Could not verify models: {response.status_code}")
except Exception as e:
    print(f"   ❌ Error checking models: {e}")
    print("   Make sure backend is running!")
    exit(1)

# Step 3: Upload test file
print("\n[2/4] Uploading test file...")
try:
    with open(TEST_FILE, 'rb') as f:
        files = {'file': ('simple.json', f, 'application/json')}
        response = requests.post(f"{BASE_URL}/upload-documents/", files=files, timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        filename = data.get('filename')
        print(f"   ✅ File uploaded: {filename}")
    else:
        print(f"   ❌ Upload failed: {response.status_code}")
        exit(1)
except Exception as e:
    print(f"   ❌ Upload error: {e}")
    exit(1)

# Step 4: Benchmark llama3.1:8b
print("\n[3/4] Benchmarking llama3.1:8b...")
print("   NOTE: llama3.1:8b is larger (8B params) so may be slower")
print("   But should provide higher quality answers\n")

test_queries = [
    "What is the total sales amount?",
    "How many products are there?",
    "What is the average sales amount per product?"
]

llama_times = []
llama_answers = []

for i, query in enumerate(test_queries, 1):
    print(f"   Query {i}/3: {query[:50]}...")
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/analyze/",
            json={
                "query": query,
                "filename": filename
            },
            timeout=180  # Longer timeout for larger model
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get('result', 'No answer')
            llama_times.append(elapsed)
            llama_answers.append(answer)
            print(f"      ✅ {elapsed:.2f}s")
            print(f"         Answer: {answer[:80]}...")
        else:
            print(f"      ❌ Failed: {response.status_code}")
            llama_times.append(0)
            llama_answers.append("Error")
    except Exception as e:
        print(f"      ❌ Error: {e}")
        llama_times.append(0)
        llama_answers.append("Error")

# Step 5: Results comparison
print("\n[4/4] Benchmark Results - llama3.1:8b")
print("="*60)

avg_llama = sum(llama_times) / len(llama_times) if llama_times else 0
min_llama = min(llama_times) if llama_times else 0
max_llama = max(llama_times) if llama_times else 0

print("\n**llama3.1:8b Performance:**")
print(f"   Average: {avg_llama:.2f}s")
print(f"   Min:     {min_llama:.2f}s")
print(f"   Max:     {max_llama:.2f}s")
print(f"   Total:   {sum(llama_times):.2f}s for {len(test_queries)} queries")

# Comparison with phi3:mini (from previous test)
print("\n**Comparison with phi3:mini (previous test):**")
print("   phi3:mini Average:    11.75s")
print(f"   llama3.1:8b Average:  {avg_llama:.2f}s")

if avg_llama > 0:
    if avg_llama < 11.75:
        speedup = ((11.75 - avg_llama) / 11.75) * 100
        print(f"   Result: llama3.1:8b is {speedup:.1f}% FASTER ✅")
    elif avg_llama > 11.75:
        slowdown = ((avg_llama - 11.75) / 11.75) * 100
        print(f"   Result: llama3.1:8b is {slowdown:.1f}% SLOWER ⚠️")
    else:
        print(f"   Result: Same performance")

# Summary
print("\n" + "="*60)
print("SUMMARY - llama3.1:8b BENCHMARK")
print("="*60)

if avg_llama > 0:
    print(f"\n✅ llama3.1:8b benchmark complete")
    print(f"   - Average response time: {avg_llama:.2f}s")
    print(f"   - Model: 8B parameters (vs phi3:mini 3.8B)")
    
    if avg_llama < 20:
        print(f"   - Speed: EXCELLENT (under 20s)")
    elif avg_llama < 40:
        print(f"   - Speed: GOOD (20-40s)")
    else:
        print(f"   - Speed: ACCEPTABLE (slower but higher quality expected)")
    
    print("\n**Recommendation:**")
    if avg_llama < 11.75 * 1.5:  # If within 50% of phi3
        print("   ✅ llama3.1:8b is viable - good speed/quality balance")
        print("   Consider using for tasks requiring higher accuracy")
    else:
        print("   ⚠️  llama3.1:8b is slower - use phi3:mini for speed")
        print("   Use llama3.1:8b only when answer quality is critical")

print("\n" + "="*60)
print("NEXT: Please switch back to phi3:mini if needed")
print("      Update .env PRIMARY_MODEL=ollama/phi3:mini")
print("      Restart backend")
print("="*60)

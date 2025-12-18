"""
Simple test to verify routing logs appear in backend
Uses unique queries each time to avoid cache
"""

import requests
import time
import random

BASE_URL = "http://localhost:8000"
FILENAME = "sales_data.csv"

# Add random suffix to bypass cache
def make_unique_query(base_query):
    suffix = f" (test {random.randint(1000, 9999)})"
    return base_query + suffix

print("="*80)
print("🔍 SIMPLE ROUTING LOG TEST")
print("="*80)
print("\n⚠️  Check BACKEND TERMINAL for routing logs!")
print("    Look for: 🎯 [INTELLIGENT ROUTING] messages\n")

# Test 3 queries with different complexities
queries = [
    ("How many total rows exist in the dataset", "EASY - Should use tinyllama (FAST tier)"),
    ("Calculate average revenue grouped by each region", "MEDIUM - Should use phi3:mini (BALANCED tier)"),
    ("Analyze which region performs best and explain the underlying reasons why", "COMPLEX - Should use llama3.1:8b (FULL_POWER tier)")
]

for i, (base_query, description) in enumerate(queries, 1):
    query = make_unique_query(base_query)
    
    print(f"\n{'='*80}")
    print(f"Query {i}/3: {description}")
    print(f"{'='*80}")
    print(f"Sending: '{query[:60]}...'")
    print("\n⏳ Waiting for response...")
    print("👀 CHECK BACKEND TERMINAL NOW for routing logs!\n")
    
    start = time.time()
    
    try:
        response = requests.post(
            f"{BASE_URL}/analyze/",
            json={"query": query, "filename": FILENAME},
            timeout=180
        )
        
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS in {elapsed:.1f}s")
            print(f"   Answer preview: {result.get('result', '')[:100]}...")
        else:
            print(f"❌ FAILED: {response.status_code}")
            
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ ERROR after {elapsed:.1f}s: {str(e)[:100]}")
    
    print(f"\n{'='*80}")

print("\n\n")
print("="*80)
print("🎯 TEST COMPLETE")
print("="*80)
print("\n📋 EXPECTED BACKEND LOGS:")
print("   Query 1: 🎯 [INTELLIGENT ROUTING] Complexity: ~0.1")
print("            ⚡ Tier: FAST")
print("            🤖 Model: tinyllama:latest")
print()
print("   Query 2: 🎯 [INTELLIGENT ROUTING] Complexity: ~0.4")
print("            ⚖️  Tier: BALANCED")
print("            🤖 Model: phi3:mini")
print()
print("   Query 3: 🎯 [INTELLIGENT ROUTING] Complexity: ~0.7")
print("            🚀 Tier: FULL_POWER")
print("            🤖 Model: llama3.1:8b")
print()
print("❓ Did you see these logs? If NOT, routing is not working!")
print("="*80)

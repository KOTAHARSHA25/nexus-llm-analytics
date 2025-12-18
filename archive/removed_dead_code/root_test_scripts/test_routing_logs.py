"""
Quick test to verify routing logs appear in backend terminal
"""
import requests
import time

print("Testing routing logs...")
print("-" * 60)

# Test queries with different complexity levels
queries = [
    ("Simple query", "What is the total?"),
    ("Medium query", "Compare sales by region"),
    ("Complex query", "Analyze trends and predict future patterns with detailed insights"),
]

for i, (label, query) in enumerate(queries, 1):
    print(f"\nQuery {i} ({label}): {query}")
    start = time.time()
    
    try:
        result = requests.post(
            'http://localhost:8000/analyze/',
            json={
                'query': query,
                'filename': 'sales.csv'
            },
            timeout=120  # 2 minutes timeout
        )
        elapsed = time.time() - start
        
        if result.status_code == 200:
            data = result.json()
            print(f"  ✅ Completed in {elapsed:.2f}s")
            print(f"  Success: {data.get('success', False)}")
            if data.get('result'):
                preview = str(data['result'])[:100]
                print(f"  Preview: {preview}...")
        else:
            print(f"  ❌ HTTP {result.status_code}")
            
    except requests.Timeout:
        elapsed = time.time() - start
        print(f"  ⏱️  Timeout after {elapsed:.2f}s (query still running in backend)")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print(f"  💡 Check backend terminal for routing decision!")
    time.sleep(1)  # Brief pause between queries

print("\n" + "=" * 60)
print("✅ TEST COMPLETE - CHECK YOUR BACKEND TERMINAL!")
print("=" * 60)
print("\nYou should see routing logs like:")
print("  [INTELLIGENT ROUTING] Complexity: 0.XXX")
print("  Tier: FAST / BALANCED / FULL_POWER")
print("  Model: tinyllama:latest / phi3:mini / llama3.1:8b")
print("  Expected: 1-3s / 3-6s / 8-15s")
print("=" * 60)

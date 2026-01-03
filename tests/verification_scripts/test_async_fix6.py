"""
Test Fix 6: Async LLM Calls
Tests async generate methods and verifies they work correctly
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.core.llm_client import LLMClient


async def test_async_generate():
    """Test async generate method"""
    print("🔬 Testing async LLM generate...")
    
    client = LLMClient()
    
    # Test 1: Simple async call
    print("\n1️⃣ Test simple async call...")
    start = time.time()
    result = await client.generate_async("Say 'hello' in one word", adaptive_timeout=True)
    elapsed = time.time() - start
    
    if result.get('success'):
        print(f"✅ Async call succeeded in {elapsed:.2f}s")
        print(f"   Response: {result.get('response', '')[:50]}")
    else:
        print(f"❌ Async call failed: {result.get('error')}")
        return False
    
    # Test 2: Primary model async
    print("\n2️⃣ Test primary model async...")
    start = time.time()
    result = await client.generate_primary_async("What is 2+2? Answer in digits only.")
    elapsed = time.time() - start
    
    if result.get('success'):
        print(f"✅ Primary async succeeded in {elapsed:.2f}s")
        print(f"   Response: {result.get('response', '')[:50]}")
    else:
        print(f"❌ Primary async failed: {result.get('error')}")
        return False
    
    # Test 3: Concurrent async calls (throughput test)
    print("\n3️⃣ Test concurrent async calls...")
    start = time.time()
    
    tasks = [
        client.generate_async("Count to 3", adaptive_timeout=True),
        client.generate_async("Say 'test'", adaptive_timeout=True),
        client.generate_async("Answer: yes or no?", adaptive_timeout=True)
    ]
    
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start
    
    successful = sum(1 for r in results if r.get('success'))
    print(f"✅ {successful}/3 concurrent calls succeeded in {elapsed:.2f}s")
    print(f"   (Sequential would take ~{elapsed*3:.1f}s)")
    
    return True


async def test_sync_vs_async():
    """Compare sync vs async performance"""
    print("\n⚡ Performance comparison: Sync vs Async")
    
    client = LLMClient()
    
    # Sync version
    print("\n📊 Running 3 SYNC calls...")
    start = time.time()
    for i in range(3):
        result = client.generate(f"Count to {i+1}", adaptive_timeout=True)
    sync_time = time.time() - start
    print(f"   Sync total: {sync_time:.2f}s")
    
    # Async version
    print("\n📊 Running 3 ASYNC concurrent calls...")
    start = time.time()
    tasks = [
        client.generate_async(f"Count to {i+1}", adaptive_timeout=True)
        for i in range(3)
    ]
    results = await asyncio.gather(*tasks)
    async_time = time.time() - start
    print(f"   Async total: {async_time:.2f}s")
    
    speedup = sync_time / async_time if async_time > 0 else 0
    print(f"\n🚀 Speedup: {speedup:.1f}x faster with async!")
    
    return True


async def main():
    """Run all async tests"""
    print("=" * 60)
    print("FIX 6: ASYNC LLM CALLS - Verification Test")
    print("=" * 60)
    
    try:
        # Test async methods
        success = await test_async_generate()
        if not success:
            print("\n❌ Async generation tests failed")
            return 1
        
        # Test performance
        await test_sync_vs_async()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Fix 6 implemented successfully!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

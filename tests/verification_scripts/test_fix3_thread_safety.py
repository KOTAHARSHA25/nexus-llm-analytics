"""
Test Fix 3: Thread-Safe Singletons
Tests concurrent access to singleton instances to ensure no race conditions
"""
import threading
import time
from typing import List

print("="*80)
print("🧪 TESTING FIX 3: THREAD-SAFE SINGLETONS")
print("="*80)

# Test 1: AnalysisService Singleton
print("\n📝 Test 1: AnalysisService thread safety")
print("-"*80)

from src.backend.services.analysis_service import get_analysis_service

instances1 = []
def get_service():
    instance = get_analysis_service()
    instances1.append(id(instance))
    time.sleep(0.01)  # Small delay to increase race condition chance

threads = []
for i in range(10):
    t = threading.Thread(target=get_service)
    threads.append(t)
    t.start()

for t in threads:
    t.join()

unique_ids1 = set(instances1)
if len(unique_ids1) == 1:
    print(f"✅ PASS: All 10 threads got the same instance (ID: {list(unique_ids1)[0]})")
else:
    print(f"❌ FAIL: Got {len(unique_ids1)} different instances (race condition!)")
    print(f"   IDs: {unique_ids1}")

# Test 2: AgentRegistry Singleton
print("\n📝 Test 2: AgentRegistry thread safety")
print("-"*80)

from src.backend.core.plugin_system import get_agent_registry

instances2 = []
def get_registry():
    instance = get_agent_registry()
    instances2.append(id(instance))
    time.sleep(0.01)

threads = []
for i in range(10):
    t = threading.Thread(target=get_registry)
    threads.append(t)
    t.start()

for t in threads:
    t.join()

unique_ids2 = set(instances2)
if len(unique_ids2) == 1:
    print(f"✅ PASS: All 10 threads got the same instance (ID: {list(unique_ids2)[0]})")
else:
    print(f"❌ FAIL: Got {len(unique_ids2)} different instances (race condition!)")
    print(f"   IDs: {unique_ids2}")

# Test 3: ModelManager Singleton
print("\n📝 Test 3: ModelManager thread safety")
print("-"*80)

from src.backend.agents.model_manager import get_model_manager

instances3 = []
def get_initializer():
    instance = get_model_manager()
    instances3.append(id(instance))
    time.sleep(0.01)

threads = []
for i in range(10):
    t = threading.Thread(target=get_initializer)
    threads.append(t)
    t.start()

for t in threads:
    t.join()

unique_ids3 = set(instances3)
if len(unique_ids3) == 1:
    print(f"✅ PASS: All 10 threads got the same instance (ID: {list(unique_ids3)[0]})")
else:
    print(f"❌ FAIL: Got {len(unique_ids3)} different instances (race condition!)")
    print(f"   IDs: {unique_ids3}")

# Summary
print("\n" + "="*80)
print("📊 FIX 3 TEST SUMMARY")
print("="*80)

all_passed = len(unique_ids1) == 1 and len(unique_ids2) == 1 and len(unique_ids3) == 1

if all_passed:
    print("✅ ALL TESTS PASSED - Thread-safe singletons working correctly!")
    print("\n🎯 Impact:")
    print("  - No race conditions under concurrent load")
    print("  - True singleton pattern (single instance guaranteed)")
    print("  - Production-ready for multi-threaded FastAPI server")
else:
    print("❌ SOME TESTS FAILED - Race conditions detected")
    print("   Check the threading locks implementation")

print("="*80)

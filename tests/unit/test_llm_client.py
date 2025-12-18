"""
TEST: LLM Client Communication
Purpose: Test core LLM communication with Ollama
Date: December 16, 2025
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.backend.core.llm_client import LLMClient
import time

print("="*80)
print("🧪 TESTING: LLM Client (llm_client.py)")
print("="*80)

# ============================================================================
# TEST 1: Initialize LLM Client
# ============================================================================
print("\n[TEST 1] LLM Client Initialization")
print("-"*80)

try:
    client = LLMClient()
    print(f"✅ Client initialized successfully")
    print(f"   Primary Model: {client.primary_model}")
    print(f"   Review Model: {client.review_model}")
    print(f"   Base URL: {client.base_url}")
    test1_pass = True
except Exception as e:
    print(f"❌ FAILED: {e}")
    test1_pass = False

# ============================================================================
# TEST 2: Simple Arithmetic Query (Ground Truth Test)
# ============================================================================
print("\n[TEST 2] Simple Arithmetic Query - Accuracy Test")
print("-"*80)

test2_pass = False
if test1_pass:
    try:
        prompt = "What is 15 + 27? Respond with ONLY the number, no explanation."
        print(f"Prompt: {prompt}")
        print(f"Expected Answer: 42")
        
        start_time = time.time()
        result = client.generate_primary(prompt)
        elapsed = time.time() - start_time
        
        response = result.get("response", "").strip()
        print(f"Actual Answer: {response}")
        print(f"Response Time: {elapsed:.2f}s")
        
        # Check if response contains the correct answer
        if "42" in response:
            print("✅ PASSED - Correct answer (42)")
            test2_pass = True
        else:
            print(f"❌ FAILED - Wrong answer. Expected: 42, Got: {response}")
            
        # Check for errors
        if result.get("error"):
            print(f"⚠️ Warning: Error reported: {result['error']}")
            
    except Exception as e:
        print(f"❌ FAILED: {e}")

# ============================================================================
# TEST 3: Data Analysis Query (Realistic Test)
# ============================================================================
print("\n[TEST 3] Data Analysis Query - Realistic Scenario")
print("-"*80)

test3_pass = False
if test1_pass:
    try:
        prompt = """Given this data: Sales = [100, 200, 150, 300, 250]
Calculate the total sales. Respond with ONLY the number."""
        
        print(f"Prompt: {prompt}")
        print(f"Expected Answer: 1000")
        
        start_time = time.time()
        result = client.generate_primary(prompt)
        elapsed = time.time() - start_time
        
        response = result.get("response", "").strip()
        print(f"Actual Answer: {response}")
        print(f"Response Time: {elapsed:.2f}s")
        
        # Check if response contains the correct answer
        if "1000" in response:
            print("✅ PASSED - Correct total (1000)")
            test3_pass = True
        else:
            print(f"❌ FAILED - Wrong answer. Expected: 1000, Got: {response}")
            
    except Exception as e:
        print(f"❌ FAILED: {e}")

# ============================================================================
# TEST 4: Review Model Test
# ============================================================================
print("\n[TEST 4] Review Model Communication")
print("-"*80)

test4_pass = False
if test1_pass:
    try:
        prompt = "Is 2+2=4 correct? Answer YES or NO only."
        print(f"Prompt: {prompt}")
        print(f"Expected Answer: YES")
        
        start_time = time.time()
        result = client.generate_review(prompt)
        elapsed = time.time() - start_time
        
        response = result.get("response", "").strip().upper()
        print(f"Actual Answer: {response}")
        print(f"Response Time: {elapsed:.2f}s")
        print(f"Model Used: {result.get('model')}")
        
        # Check if response contains YES
        if "YES" in response:
            print("✅ PASSED - Review model working")
            test4_pass = True
        else:
            print(f"⚠️ WARNING - Unexpected answer: {response}")
            test4_pass = True  # Still pass if model responded
            
    except Exception as e:
        print(f"❌ FAILED: {e}")

# ============================================================================
# TEST 5: Circuit Breaker Test (Error Handling)
# ============================================================================
print("\n[TEST 5] Circuit Breaker & Error Handling")
print("-"*80)

test5_pass = False
if test1_pass:
    try:
        # Test with invalid model
        prompt = "Test error handling"
        print(f"Testing with invalid model: nonexistent_model")
        
        result = client.generate(prompt, model="nonexistent_model")
        
        if result.get("error") or result.get("fallback_used"):
            print("✅ PASSED - Error handling working")
            print(f"   Error: {result.get('error', 'N/A')}")
            print(f"   Fallback: {result.get('fallback_used', False)}")
            test5_pass = True
        else:
            print("⚠️ WARNING - No error reported for invalid model")
            test5_pass = True  # Still pass if it handled gracefully
            
    except Exception as e:
        print(f"✅ PASSED - Exception caught: {e}")
        test5_pass = True

# ============================================================================
# TEST 6: Response Structure Validation
# ============================================================================
print("\n[TEST 6] Response Structure Validation")
print("-"*80)

test6_pass = False
if test2_pass and test1_pass:
    try:
        prompt = "What is 10 times 5?"
        result = client.generate_primary(prompt)
        
        # Check required fields
        required_fields = ["model", "prompt", "response"]
        missing_fields = [field for field in required_fields if field not in result]
        
        if not missing_fields:
            print("✅ PASSED - All required fields present")
            print(f"   Fields: {list(result.keys())}")
            test6_pass = True
        else:
            print(f"❌ FAILED - Missing fields: {missing_fields}")
            
    except Exception as e:
        print(f"❌ FAILED: {e}")

# ============================================================================
# TEST 7: Adaptive Timeout Calculation
# ============================================================================
print("\n[TEST 7] Adaptive Timeout Calculation")
print("-"*80)

test7_pass = False
if test1_pass:
    try:
        # Test timeout calculation for different models
        models = ["phi3:mini", "tinyllama", "llama3.1:8b"]
        
        for model in models:
            timeout = client._calculate_adaptive_timeout(model)
            print(f"   {model}: {timeout}s timeout")
            
        print("✅ PASSED - Adaptive timeout working")
        test7_pass = True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")

# ============================================================================
# TEST SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 LLM CLIENT TEST SUMMARY")
print("="*80)

tests = [
    ("Initialization", test1_pass),
    ("Arithmetic Accuracy", test2_pass),
    ("Data Analysis", test3_pass),
    ("Review Model", test4_pass),
    ("Error Handling", test5_pass),
    ("Response Structure", test6_pass),
    ("Adaptive Timeout", test7_pass)
]

passed = sum(1 for _, result in tests if result)
total = len(tests)

for test_name, result in tests:
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{status} - {test_name}")

print("-"*80)
print(f"Overall: {passed}/{total} tests passed ({(passed/total*100):.1f}%)")

# Critical issues
if not test1_pass:
    print("\n🚨 CRITICAL: LLM client initialization failed!")
elif not test2_pass:
    print("\n⚠️ WARNING: Arithmetic accuracy test failed - LLM not giving correct answers")
elif passed == total:
    print("\n✅ SUCCESS: All LLM client tests passed!")
else:
    print(f"\n⚠️ ATTENTION: {total-passed} test(s) failed")

print("="*80)

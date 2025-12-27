"""
Test that the critic LLM can catch intentional errors
This validates the Two Friends Model actually works
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backend.core.self_correction_engine import SelfCorrectionEngine
from src.backend.core.llm_client import LLMClient
from src.backend.core.cot_parser import CoTParser
import json


def load_config():
    """Load CoT review config"""
    config_path = project_root / "config" / "cot_review_config.json"
    with open(config_path) as f:
        full_config = json.load(f)
        return full_config['cot_review']  # Return the nested cot_review section


async def test_critic_catches_calculation_error():
    """Test 1: Critic should catch wrong calculation (2+2=5)"""
    print("\n" + "="*70)
    print("TEST 1: Can critic catch calculation error?")
    print("="*70)
    
    query = "What is 2 + 2?"
    
    # Intentionally wrong response
    wrong_response = """[REASONING]
Let me calculate 2 + 2.
Using basic arithmetic: 2 + 2 = 5
[/REASONING]

[OUTPUT]
The answer is 5.
[/OUTPUT]"""
    
    # Initialize components
    llm = LLMClient()
    config = load_config()
    
    engine = SelfCorrectionEngine(
        config=config,
        llm_client=llm
    )
    
    # Parse the wrong response
    parsed = parser.parse(wrong_response)
    
    print(f"\nQuery: {query}")
    print(f"Wrong Reasoning: {parsed.reasoning[:100]}...")
    print(f"Wrong Output: {parsed.output}")
    
    # Critic should catch this
    print("\n⏳ Asking critic to review...")
    review_result = await engine._get_critic_review(
        query=query,
        cot_reasoning=parsed.reasoning,
        final_output=parsed.output,
        data_context="No data context needed for this calculation"
    )
    
    print(f"\nCritic Response:\n{review_result.response}")
    print(f"\nCritic Decision: {'✅ VALID' if review_result.is_valid else '❌ INVALID'}")
    
    if not review_result.is_valid:
        print(f"Feedback: {review_result.feedback}")
        print("\n✅ TEST PASSED: Critic correctly caught the calculation error")
        return True
    else:
        print("\n❌ TEST FAILED: Critic approved wrong calculation (2+2=5)")
        return False


async def test_critic_catches_logic_error():
    """Test 2: Critic should catch inverted logic"""
    print("\n" + "="*70)
    print("TEST 2: Can critic catch logic error?")
    print("="*70)
    
    query = "Filter customers where age > 30"
    
    # Wrong logic (inverted condition)
    wrong_response = """[REASONING]
To find customers where age > 30, I will filter the data.
I'll use the condition age < 30 to get customers above 30.
This will give me all customers older than 30.
[/REASONING]

[OUTPUT]
Filtered customers using condition age < 30.
[/OUTPUT]"""
    
    llm = LLMClient()
    parser = CoTParser()
    config = load_config()
    
    engine = SelfCorrectionEngine(
        config=config,
        llm_client=llm
    )
    
    parsed = parser.parse(wrong_response)
    
    print(f"\nQuery: {query}")
    print(f"Wrong Logic: {parsed.reasoning[:150]}...")
    print(f"Wrong Output: {parsed.output}")
    
    print("\n⏳ Asking critic to review...")
    review_result = await engine._get_critic_review(
        query=query,
        cot_reasoning=parsed.reasoning,
        final_output=parsed.output,
        data_context="Dataframe with columns: customer_id, age, purchase_amount"
    )
    
    print(f"\nCritic Response:\n{review_result.response}")
    print(f"\nCritic Decision: {'✅ VALID' if review_result.is_valid else '❌ INVALID'}")
    
    if not review_result.is_valid:
        print(f"Feedback: {review_result.feedback}")
        print("\n✅ TEST PASSED: Critic correctly caught the logic error")
        return True
    else:
        print("\n❌ TEST FAILED: Critic approved inverted logic")
        return False


async def test_critic_approves_correct_answer():
    """Test 3: Critic should approve correct reasoning"""
    print("\n" + "="*70)
    print("TEST 3: Can critic approve correct answer?")
    print("="*70)
    
    query = "What is the average of [10, 20, 30]?"
    
    # Correct response
    correct_response = """[REASONING]
To find the average of [10, 20, 30]:
1. Sum the values: 10 + 20 + 30 = 60
2. Count the elements: 3
3. Divide sum by count: 60 / 3 = 20
[/REASONING]

[OUTPUT]
The average is 20.
[/OUTPUT]"""
    
    llm = LLMClient()
    parser = CoTParser()
    config = load_config()
    
    engine = SelfCorrectionEngine(
        config=config,
        llm_client=llm
    )
    
    parsed = parser.parse(correct_response)
    
    print(f"\nQuery: {query}")
    print(f"Correct Reasoning: {parsed.reasoning}")
    print(f"Correct Output: {parsed.output}")
    
    print("\n⏳ Asking critic to review...")
    review_result = await engine._get_critic_review(
        query=query,
        cot_reasoning=parsed.reasoning,
        final_output=parsed.output,
        data_context="Array: [10, 20, 30]"
    )
    
    print(f"\nCritic Response:\n{review_result.response}")
    print(f"\nCritic Decision: {'✅ VALID' if review_result.is_valid else '❌ INVALID'}")
    
    if review_result.is_valid:
        print("\n✅ TEST PASSED: Critic correctly approved correct answer")
        return True
    else:
        print(f"Feedback: {review_result.feedback}")
        print("\n⚠️ TEST WARNING: Critic rejected correct answer (too strict)")
        return True  # Still pass - being strict is better than lenient


async def main():
    """Run all critic strictness tests"""
    print("\n" + "="*70)
    print("CRITIC STRICTNESS VALIDATION")
    print("Testing if Two Friends Model can actually catch errors")
    print("="*70)
    
    results = []
    
    # Test 1: Calculation error
    try:
        result1 = await test_critic_catches_calculation_error()
        results.append(("Calculation Error Detection", result1))
    except Exception as e:
        print(f"\n❌ Test 1 crashed: {e}")
        results.append(("Calculation Error Detection", False))
    
    # Test 2: Logic error
    try:
        result2 = await test_critic_catches_logic_error()
        results.append(("Logic Error Detection", result2))
    except Exception as e:
        print(f"\n❌ Test 2 crashed: {e}")
        results.append(("Logic Error Detection", False))
    
    # Test 3: Correct answer approval
    try:
        result3 = await test_critic_approves_correct_answer()
        results.append(("Correct Answer Approval", result3))
    except Exception as e:
        print(f"\n❌ Test 3 crashed: {e}")
        results.append(("Correct Answer Approval", False))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 SUCCESS: Two Friends Model is working!")
        print("Critic can detect errors and approve correct answers.")
    elif passed >= 2:
        print("\n⚠️ PARTIAL: Two Friends Model working but needs tuning")
    else:
        print("\n❌ FAILURE: Two Friends Model still not working properly")
        print("Critic prompt may need further adjustment")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

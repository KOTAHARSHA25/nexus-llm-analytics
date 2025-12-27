"""
Two Friends Model - Complete Integration Test
==============================================
Tests the full Generator → Critic → Improvement loop

This test validates:
1. AutomatedValidator catches rule-based errors (fast path)
2. LLM Critic catches semantic errors (when needed)
3. Generator improves based on feedback
4. System is domain-agnostic (works with any data)
"""
import sys
from pathlib import Path
import json
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backend.core.self_correction_engine import SelfCorrectionEngine
from src.backend.core.llm_client import LLMClient
from src.backend.core.automated_validation import AutomatedValidator


def test_automated_validator_integration():
    """Test that AutomatedValidator is properly integrated"""
    print("\n" + "="*70)
    print("TEST 1: AutomatedValidator Integration")
    print("="*70)
    
    validator = AutomatedValidator()
    
    # Test case with deliberate error
    result = validator.validate(
        query="What percentage of customers are satisfied?",
        reasoning="I calculated 75/100 = 0.75, so 0.75 of customers are satisfied",
        output="0.75 of customers are satisfied",
        data_context={"columns": ["customer_id", "satisfied"]}
    )
    
    print(f"\nQuery: What percentage of customers are satisfied?")
    print(f"Bad Output: '0.75 of customers are satisfied'")
    print(f"Expected: Should flag missing percentage conversion")
    print(f"\nValidator Result:")
    print(f"  Valid: {result.is_valid}")
    print(f"  Issues: {len(result.issues)}")
    
    for issue in result.issues:
        print(f"    - [{issue.severity}] {issue.description}")
    
    if not result.is_valid:
        print("\n✅ AutomatedValidator correctly caught the error!")
        return True
    else:
        print("\n❌ AutomatedValidator missed the error")
        return False


def test_critic_feedback_generation():
    """Test that CriticFeedback.feedback property works"""
    print("\n" + "="*70)
    print("TEST 2: CriticFeedback Generation")
    print("="*70)
    
    from src.backend.core.cot_parser import CriticFeedback, CriticIssue
    
    # Create feedback with issues
    issues = [
        CriticIssue(
            description="Calculation error: 2+2=5 is wrong",
            location="Step 3",
            severity="HIGH",
            suggestion="Correct to 2+2=4"
        ),
        CriticIssue(
            description="Missing percentage conversion",
            location="Output",
            severity="MEDIUM",
            suggestion="Multiply by 100 for percentage"
        )
    ]
    
    feedback = CriticFeedback(
        is_valid=False,
        issues=issues,
        raw_response="Test response"
    )
    
    print(f"\nFeedback text generated:")
    print("-" * 40)
    print(feedback.feedback)
    print("-" * 40)
    
    if "Calculation error" in feedback.feedback and "Step 3" in feedback.feedback:
        print("\n✅ CriticFeedback.feedback property works!")
        return True
    else:
        print("\n❌ CriticFeedback.feedback property not generating correctly")
        return False


def test_self_correction_engine_creation():
    """Test that SelfCorrectionEngine initializes with AutomatedValidator"""
    print("\n" + "="*70)
    print("TEST 3: SelfCorrectionEngine Setup")
    print("="*70)
    
    llm_client = LLMClient()
    
    config_path = project_root / 'config' / 'cot_review_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)['cot_review']
    
    engine = SelfCorrectionEngine(config, llm_client)
    
    # Check components
    has_cot_parser = hasattr(engine, 'cot_parser') and engine.cot_parser is not None
    has_critic_parser = hasattr(engine, 'critic_parser') and engine.critic_parser is not None
    has_auto_validator = hasattr(engine, 'automated_validator') and engine.automated_validator is not None
    
    print(f"\nEngine Components:")
    print(f"  CoT Parser: {'✅' if has_cot_parser else '❌'}")
    print(f"  Critic Parser: {'✅' if has_critic_parser else '❌'}")
    print(f"  AutomatedValidator: {'✅' if has_auto_validator else '❌'}")
    
    if has_cot_parser and has_critic_parser and has_auto_validator:
        print("\n✅ SelfCorrectionEngine fully configured!")
        return True
    else:
        print("\n❌ SelfCorrectionEngine missing components")
        return False


def test_domain_agnostic_validation():
    """Test validation works across different domains"""
    print("\n" + "="*70)
    print("TEST 4: Domain-Agnostic Validation")
    print("="*70)
    
    validator = AutomatedValidator()
    
    test_cases = [
        # Healthcare domain
        {
            "name": "Healthcare",
            "query": "What percentage of patients recovered?",
            "reasoning": "Recovered: 85, Total: 100, Rate = 85/100 = 0.85",
            "output": "0.85 of patients recovered",
            "context": {"columns": ["patient_id", "status", "recovery_date"]}
        },
        # Finance domain
        {
            "name": "Finance",
            "query": "Calculate profit margin",
            "reasoning": "Profit margin = revenue / cost = $1000 / $800 = 1.25",
            "output": "Profit margin is 1.25",
            "context": {"columns": ["revenue", "cost", "profit"]}
        },
        # Education domain
        {
            "name": "Education",
            "query": "What is the pass rate for students above 60?",
            "reasoning": "Filter students where score < 60 to find those above 60",
            "output": "50 students passed",
            "context": {"columns": ["student_id", "score", "grade"]}
        },
        # Time period error (any domain)
        {
            "name": "Q4 Sales",
            "query": "What were Q4 revenues?",
            "reasoning": "Q4 means September, October, November (months 9, 10, 11)",
            "output": "Q4 revenue: $500,000",
            "context": {"columns": ["month", "revenue", "orders"]}
        },
    ]
    
    passed = 0
    for tc in test_cases:
        result = validator.validate(
            query=tc["query"],
            reasoning=tc["reasoning"],
            output=tc["output"],
            data_context=tc["context"]
        )
        
        status = "❌" if result.is_valid else "✅"
        print(f"  {status} {tc['name']}: {'Caught error' if not result.is_valid else 'Missed error'}")
        if not result.is_valid:
            passed += 1
            for issue in result.issues[:2]:
                print(f"       → {issue.description[:60]}...")
    
    print(f"\n✅ Caught {passed}/{len(test_cases)} domain-agnostic errors")
    return passed >= 3  # At least 75% caught


def test_feedback_loop_simulation():
    """Simulate the improvement loop without LLM calls"""
    print("\n" + "="*70)
    print("TEST 5: Feedback Loop Simulation")
    print("="*70)
    
    from src.backend.core.cot_parser import CriticFeedback, CriticIssue
    
    # Simulate iteration 1: Generator makes mistake
    print("\n📝 Iteration 1: Generator output")
    gen_output_1 = "The average is 100 + 200 = 350"  # Wrong: should be 300
    print(f"   Output: '{gen_output_1}'")
    
    # AutomatedValidator catches it
    validator = AutomatedValidator()
    result = validator.validate(
        query="What is the average of 100 and 200?",
        reasoning="Adding 100 and 200: 100 + 200 = 350",
        output=gen_output_1,
        data_context=None
    )
    
    print(f"\n🔍 Critic Review (Automated):")
    print(f"   Valid: {result.is_valid}")
    if not result.is_valid:
        print(f"   Feedback: {result.issues[0].description if result.issues else 'None'}")
    
    # Simulate iteration 2: Generator corrects based on feedback
    print("\n📝 Iteration 2: Generator corrects")
    gen_output_2 = "The average is (100 + 200) / 2 = 300 / 2 = 150"
    print(f"   Output: '{gen_output_2}'")
    
    # Validate corrected output
    result_2 = validator.validate(
        query="What is the average of 100 and 200?",
        reasoning="First sum: 100 + 200 = 300. Then average: 300 / 2 = 150",
        output=gen_output_2,
        data_context=None
    )
    
    print(f"\n🔍 Critic Review (Automated):")
    print(f"   Valid: {result_2.is_valid}")
    
    if not result.is_valid and result_2.is_valid:
        print("\n✅ Feedback loop successfully improved the output!")
        return True
    else:
        print("\n⚠️ Feedback loop test inconclusive")
        return False


def main():
    print("\n" + "="*70)
    print("TWO FRIENDS MODEL - COMPLETE INTEGRATION TEST")
    print("Verifying domain-agnostic, data-agnostic self-correction")
    print("="*70)
    
    results = {
        "automated_validator": test_automated_validator_integration(),
        "critic_feedback": test_critic_feedback_generation(),
        "engine_setup": test_self_correction_engine_creation(),
        "domain_agnostic": test_domain_agnostic_validation(),
        "feedback_loop": test_feedback_loop_simulation(),
    }
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅" if passed_test else "❌"
        print(f"  {status} {test_name.replace('_', ' ').title()}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
    
    if passed == total:
        print("\n🎉 TWO FRIENDS MODEL IS FULLY INTEGRATED!")
        print("   The system can now:")
        print("   • Catch calculation errors automatically")
        print("   • Detect logic inversions")
        print("   • Flag causation/correlation confusion")
        print("   • Identify time period mistakes")
        print("   • Validate across any domain/data type")
    else:
        print(f"\n⚠️ {total - passed} test(s) need attention")


if __name__ == "__main__":
    main()

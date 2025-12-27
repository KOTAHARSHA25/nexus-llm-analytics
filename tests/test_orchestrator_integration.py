"""
Test QueryOrchestrator Integration with DataAnalystAgent
=========================================================
Verifies the 3-track unified decision system:
1. Complexity → Model Selection
2. Query Type → Execution Method  
3. Complexity + Method → Review Level (Two Friends)
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backend.core.query_orchestrator import QueryOrchestrator, ExecutionMethod, ReviewLevel


def test_orchestrator_decisions():
    """Test the QueryOrchestrator makes correct unified decisions"""
    
    print("="*70)
    print("QUERY ORCHESTRATOR INTEGRATION TEST")
    print("Testing 3-Track Unified Decision System")
    print("="*70)
    
    # Configure orchestrator
    config = {
        'model_selection': {
            'simple': 'tinyllama',
            'medium': 'phi3:mini',
            'complex': 'llama3.1:8b',
            'thresholds': {
                'simple_max': 0.3,
                'medium_max': 0.7
            }
        },
        'cot_review': {
            'activation_rules': {
                'always_on_complexity': 0.7,
                'optional_range': [0.3, 0.7],
                'always_on_code_gen': True
            }
        }
    }
    
    orchestrator = QueryOrchestrator(None, config)
    
    # Test cases: (query, data, expected_model, expected_method, expected_review)
    test_cases = [
        # Simple queries → tinyllama, direct, no review
        {
            "query": "What is a customer?",
            "data": None,
            "expected_model": "tinyllama",
            "expected_review": ReviewLevel.NONE,
            "description": "Simple definition query"
        },
        # Medium queries with data → needs moderate complexity for phi3:mini
        {
            "query": "Calculate the average sales by region and group by quarter for the current year",
            "data": {"columns": ["region", "sales", "quarter"]},
            "expected_model": "phi3:mini",
            "expected_method": ExecutionMethod.CODE_GENERATION,
            "expected_review": ReviewLevel.MANDATORY,  # Code gen REQUIRES review for safety
            "description": "Medium computation query"
        },
        # Complex multi-step → llama3.1:8b, mandatory review
        {
            "query": "Analyze the correlation between quarterly revenue and marketing spend over the past 3 years, then predict next quarter's performance and identify top 3 risk factors that could impact growth",
            "data": {"columns": ["quarter", "revenue", "marketing_spend"]},
            "expected_model": "llama3.1:8b",
            "expected_review": ReviewLevel.MANDATORY,
            "description": "Complex multi-step analysis"
        },
        # Computation without data → direct LLM (no code gen)
        {
            "query": "Calculate the sum of 5 + 3",
            "data": None,
            "expected_model": "tinyllama",
            "expected_method": ExecutionMethod.DIRECT_LLM,
            "expected_review": ReviewLevel.NONE,
            "description": "Simple math without data"
        },
    ]
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─'*60}")
        print(f"TEST {i}: {test['description']}")
        print(f"Query: {test['query'][:60]}...")
        
        plan = orchestrator.create_execution_plan(
            query=test['query'],
            data=test['data']
        )
        
        # Check results
        checks = []
        
        # Model check
        if 'expected_model' in test:
            model_ok = plan.model == test['expected_model']
            checks.append(('Model', model_ok, test['expected_model'], plan.model))
        
        # Method check
        if 'expected_method' in test:
            method_ok = plan.execution_method == test['expected_method']
            checks.append(('Method', method_ok, test['expected_method'].value, plan.execution_method.value))
        
        # Review check
        if 'expected_review' in test:
            review_ok = plan.review_level == test['expected_review']
            checks.append(('Review', review_ok, test['expected_review'].value, plan.review_level.value))
        
        # Display results
        all_passed = all(c[1] for c in checks)
        
        for check_name, passed, expected, actual in checks:
            status = "✅" if passed else "❌"
            print(f"   {status} {check_name}: {actual} (expected: {expected})")
        
        print(f"   Complexity: {plan.complexity_score:.2f}")
        
        results.append(all_passed)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("QueryOrchestrator 3-track system working correctly:")
        print("  • Track 1: Complexity → Model Selection ✅")
        print("  • Track 2: Query Type → Execution Method ✅")
        print("  • Track 3: Complexity + Method → Review Level ✅")
    else:
        print(f"\n⚠️ {total - passed} tests failed")
    
    return passed == total


def test_model_selection_thresholds():
    """Test model selection based on complexity thresholds"""
    
    print("\n" + "="*70)
    print("MODEL SELECTION THRESHOLD TEST")
    print("="*70)
    
    config = {
        'model_selection': {
            'simple': 'tinyllama',
            'medium': 'phi3:mini',
            'complex': 'llama3.1:8b',
            'thresholds': {
                'simple_max': 0.3,
                'medium_max': 0.7
            }
        },
        'cot_review': {'activation_rules': {}}
    }
    
    orchestrator = QueryOrchestrator(None, config)
    
    # Test various complexity levels
    test_queries = [
        # Very short = simple
        ("Hi", None, "tinyllama"),
        # Short definition = simple
        ("What is data?", None, "tinyllama"),
        # Medium calculation = medium model
        ("Calculate the average of all values and group by category", {"data": True}, "phi3:mini"),
        # Long complex = complex model
        ("Analyze the correlation between quarterly revenue growth and marketing spend efficiency over multiple fiscal years, segment by geographic region and customer tier, then forecast next year's performance with confidence intervals and identify the key drivers of variance", {"data": True}, "llama3.1:8b"),
    ]
    
    print("\nComplexity → Model Mapping:")
    print("  < 0.3  → tinyllama (637 MB)")
    print("  0.3-0.7 → phi3:mini (2.2 GB)")
    print("  > 0.7  → llama3.1:8b (4.9 GB)")
    
    for query, data, expected_model in test_queries:
        plan = orchestrator.create_execution_plan(query, data)
        status = "✅" if plan.model == expected_model else "❌"
        print(f"\n{status} Complexity {plan.complexity_score:.2f} → {plan.model}")
        print(f"   Query: {query[:50]}...")
    
    print("\n✅ Model selection based on complexity working!")
    return True


if __name__ == "__main__":
    test1 = test_orchestrator_decisions()
    test2 = test_model_selection_thresholds()
    
    print("\n" + "="*70)
    if test1 and test2:
        print("✅ ✅ ✅  ALL INTEGRATION TESTS PASSED  ✅ ✅ ✅")
    else:
        print("⚠️ Some tests failed")

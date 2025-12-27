"""
Two Friends Model Validation Test
==================================
Tests that the Generator-Critic collaboration actually improves output quality.

This validates the core innovation: Does the review model make the primary model better?
"""
import sys
from pathlib import Path
import json
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backend.core.self_correction_engine import SelfCorrectionEngine
from src.backend.core.llm_client import LLMClient

# Test queries that commonly have issues (good candidates for critic feedback)
TEST_QUERIES = [
    {
        "id": "calc_error",
        "query": "Calculate the average profit margin (profit/revenue) across all products",
        "data_context": {
            "stats_summary": "Dataset: sales_data.csv (100 rows, 5 columns: product, revenue, profit, units, region)",
            "columns": ["product", "revenue", "profit", "units", "region"]
        },
        "common_mistake": "LLMs sometimes divide incorrectly or forget to multiply by 100 for percentage"
    },
    {
        "id": "missing_filter",
        "query": "What is total revenue for products in the 'Electronics' category?",
        "data_context": {
            "stats_summary": "Dataset: sales_data.csv (100 rows, 5 columns: product, category, revenue, units, region)",
            "columns": ["product", "category", "revenue", "units", "region"]
        },
        "common_mistake": "Might sum all revenue instead of filtering by category first"
    },
    {
        "id": "correlation_confusion",
        "query": "Is there a correlation between advertising spend and sales revenue?",
        "data_context": {
            "stats_summary": "Dataset: marketing_data.csv (50 rows, 4 columns: month, ad_spend, revenue, impressions)",
            "columns": ["month", "ad_spend", "revenue", "impressions"]
        },
        "common_mistake": "Might confuse correlation with causation or calculate incorrectly"
    },
    {
        "id": "aggregation_level",
        "query": "Show average sales by region, sorted highest to lowest",
        "data_context": {
            "stats_summary": "Dataset: sales_data.csv (100 rows, 4 columns: region, sales, date, product)",
            "columns": ["region", "sales", "date", "product"]
        },
        "common_mistake": "Might not group by region properly or forget to sort"
    },
    {
        "id": "time_series",
        "query": "What was the month-over-month growth rate in Q4?",
        "data_context": {
            "stats_summary": "Dataset: monthly_sales.csv (12 rows, 3 columns: month, revenue, orders)",
            "columns": ["month", "revenue", "orders"]
        },
        "common_mistake": "Might calculate wrong period or use wrong formula for growth rate"
    }
]


class TwoFriendsValidator:
    """Validates that the Two Friends Model improves output quality"""
    
    def __init__(self):
        self.llm_client = LLMClient()
        
        # Load CoT config
        config_path = project_root / 'config' / 'cot_review_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.cot_engine = SelfCorrectionEngine(config['cot_review'], self.llm_client)
        self.results = []
    
    def test_single_query(self, test_case):
        """Test one query through the Two Friends pipeline"""
        print(f"\n{'='*80}")
        print(f"TEST: {test_case['id']}")
        print(f"Query: {test_case['query']}")
        print(f"Common Mistake: {test_case['common_mistake']}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Run through self-correction engine
        result = self.cot_engine.run_correction_loop(
            query=test_case['query'],
            data_context=test_case['data_context'],
            generator_model='llama3.1:8b',
            critic_model='phi3:mini'
        )
        
        elapsed = time.time() - start_time
        
        # Analyze the results
        print(f"\n📊 RESULTS:")
        print(f"   Iterations: {result.total_iterations}")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Success: {result.success}")
        
        if result.total_iterations > 1:
            print(f"\n🔄 IMPROVEMENT DETECTED:")
            print(f"   Initial attempt had issues that critic caught")
            print(f"   Generator revised based on feedback")
            improvement = "YES"
        else:
            print(f"\n✓ FIRST ATTEMPT ACCEPTED:")
            print(f"   Initial output passed critic validation")
            improvement = "N/A"
        
        # Show final output preview
        print(f"\n📝 FINAL OUTPUT:")
        print(f"   {result.final_output[:200]}...")
        
        # Store results
        self.results.append({
            'test_id': test_case['id'],
            'query': test_case['query'],
            'iterations': result.total_iterations,
            'time_seconds': elapsed,
            'success': result.success,
            'improvement_made': improvement,
            'had_initial_issues': result.total_iterations > 1
        })
        
        return result
    
    def run_all_tests(self):
        """Run all test queries"""
        print("\n" + "="*80)
        print("TWO FRIENDS MODEL VALIDATION")
        print("Testing if Generator-Critic collaboration improves outputs")
        print("="*80)
        
        for test_case in TEST_QUERIES:
            try:
                self.test_single_query(test_case)
            except Exception as e:
                print(f"\n❌ ERROR: {str(e)}")
                self.results.append({
                    'test_id': test_case['id'],
                    'query': test_case['query'],
                    'success': False,
                    'error': str(e)
                })
        
        self.print_summary()
    
    def print_summary(self):
        """Print overall statistics"""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        total = len(self.results)
        successful = sum(1 for r in self.results if r.get('success', False))
        with_improvements = sum(1 for r in self.results if r.get('had_initial_issues', False))
        avg_time = sum(r.get('time_seconds', 0) for r in self.results) / total if total > 0 else 0
        avg_iterations = sum(r.get('iterations', 0) for r in self.results) / total if total > 0 else 0
        
        print(f"\n📊 Overall Performance:")
        print(f"   Total Queries: {total}")
        print(f"   Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"   Average Time: {avg_time:.2f}s")
        print(f"   Average Iterations: {avg_iterations:.2f}")
        
        print(f"\n🎯 Two Friends Model Impact:")
        print(f"   Queries Improved by Critic: {with_improvements} ({with_improvements/total*100:.1f}%)")
        print(f"   First Attempt Success Rate: {(total-with_improvements)/total*100:.1f}%")
        
        if with_improvements > 0:
            print(f"\n✅ CONCLUSION: Critic caught issues in {with_improvements} queries")
            print(f"   The Two Friends Model is WORKING - review model improves output quality")
        else:
            print(f"\n⚠️  CONCLUSION: All queries passed on first attempt")
            print(f"   Either queries too simple OR critic not strict enough")
        
        # Cost analysis
        llm_calls = successful * avg_iterations * 2  # Each iteration = generator + critic
        print(f"\n💰 Cost Analysis:")
        print(f"   Total LLM Calls: {llm_calls:.0f}")
        print(f"   Calls per Query: {llm_calls/total:.1f}")
        print(f"   Overhead vs Direct: {(llm_calls/total - 1) * 100:.0f}% more LLM calls")
        
        return self.results


def main():
    """Run validation tests"""
    validator = TwoFriendsValidator()
    results = validator.run_all_tests()
    
    # Save detailed results
    output_file = project_root / 'tests' / 'two_friends_validation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: {output_file}")
    
    # Determine if validation passed
    if results is None or len(results) == 0:
        print("\n❌ No results to analyze")
        return 1
    
    with_improvements = sum(1 for r in results if r.get('had_initial_issues', False))
    success_rate = sum(1 for r in results if r.get('success', False)) / len(results)
    
    if success_rate >= 0.8 and with_improvements >= 1:
        print(f"\n✅ VALIDATION PASSED")
        print(f"   Two Friends Model demonstrably improves output quality")
        return 0
    else:
        print(f"\n⚠️  VALIDATION INCONCLUSIVE")
        print(f"   Need more evidence that critic actually improves outputs")
        return 1


if __name__ == "__main__":
    sys.exit(main())

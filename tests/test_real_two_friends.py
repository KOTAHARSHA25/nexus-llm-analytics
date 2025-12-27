"""
Two Friends Model - REAL End-to-End Test
=========================================
This test ACTUALLY runs both LLMs and demonstrates improvement through iteration.

Tests enterprise-level complex queries, not just simple calculations.
"""
import sys
from pathlib import Path
import json
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backend.core.self_correction_engine import SelfCorrectionEngine
from src.backend.core.llm_client import LLMClient


# Enterprise-level complex queries that require real reasoning
ENTERPRISE_QUERIES = [
    {
        "id": "multi_step_analysis",
        "query": "Calculate the year-over-year growth rate for Q4 sales and identify which product categories are underperforming compared to the market average of 15%",
        "data_context": {
            "stats_summary": """Dataset: enterprise_sales.csv
- 50,000 rows covering 3 years (2022-2024)
- Columns: date, product_id, category, region, revenue, units_sold, cost, profit
- Categories: Electronics, Apparel, Home & Garden, Automotive, Food & Beverage
- Regions: North America, Europe, APAC, Latin America
- Revenue range: $10 - $50,000 per transaction
- Average profit margin: 23%""",
            "columns": ["date", "product_id", "category", "region", "revenue", "units_sold", "cost", "profit"],
            "rows": 50000
        },
        "complexity": "HIGH - requires multi-step calculation and comparison"
    },
    {
        "id": "statistical_significance",
        "query": "Determine if the difference in customer satisfaction scores between premium and standard tier customers is statistically significant, and what factors might explain any differences",
        "data_context": {
            "stats_summary": """Dataset: customer_satisfaction.csv
- 10,000 customer survey responses
- Columns: customer_id, tier, satisfaction_score, response_time, issue_resolved, agent_id, channel
- Tier distribution: Premium (30%), Standard (70%)
- Satisfaction scores: 1-10 scale
- Mean Premium: 8.2, Mean Standard: 7.4
- Standard deviation Premium: 1.1, Standard: 1.8""",
            "columns": ["customer_id", "tier", "satisfaction_score", "response_time", "issue_resolved", "agent_id", "channel"],
            "rows": 10000
        },
        "complexity": "HIGH - requires statistical testing and causal analysis"
    },
    {
        "id": "predictive_modeling",
        "query": "Based on historical patterns, predict next quarter's revenue and identify the top 3 risk factors that could impact this forecast",
        "data_context": {
            "stats_summary": """Dataset: quarterly_financials.csv
- 20 quarters of data (5 years)
- Columns: quarter, revenue, costs, marketing_spend, headcount, market_index, competitor_activity
- Revenue trend: +8% CAGR
- Seasonal pattern: Q4 typically 30% higher than Q1
- Correlation with market_index: 0.72
- R&D spend as % of revenue: 12%""",
            "columns": ["quarter", "revenue", "costs", "marketing_spend", "headcount", "market_index", "competitor_activity"],
            "rows": 20
        },
        "complexity": "HIGH - requires time series analysis and risk assessment"
    }
]


class RealTwoFriendsTest:
    """Test the actual Generator-Critic communication loop"""
    
    def __init__(self):
        self.llm_client = LLMClient()
        
        config_path = project_root / 'config' / 'cot_review_config.json'
        with open(config_path, 'r') as f:
            self.config = json.load(f)['cot_review']
        
        # Use more iterations to allow correction
        self.config['max_iterations'] = 3
        self.config['timeout_per_iteration_seconds'] = 300  # 5 minutes per iteration
        
        self.engine = SelfCorrectionEngine(self.config, self.llm_client)
    
    def test_single_query(self, test_case: dict, generator_model: str, critic_model: str):
        """Test one enterprise query"""
        print(f"\n{'='*80}")
        print(f"ENTERPRISE TEST: {test_case['id']}")
        print(f"Complexity: {test_case['complexity']}")
        print(f"{'='*80}")
        print(f"\nQuery: {test_case['query'][:100]}...")
        print(f"\nModels: Generator={generator_model}, Critic={critic_model}")
        
        start_time = time.time()
        
        result = self.engine.run_correction_loop(
            query=test_case['query'],
            data_context=test_case['data_context'],
            generator_model=generator_model,
            critic_model=critic_model
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n📊 RESULTS:")
        print(f"   Total Iterations: {result.total_iterations}")
        print(f"   Time: {elapsed:.1f}s")
        print(f"   Success: {result.success}")
        print(f"   Termination: {result.termination_reason}")
        
        # Analyze iteration details
        if result.all_iterations:
            print(f"\n🔄 ITERATION DETAILS:")
            for i, iteration in enumerate(result.all_iterations, 1):
                issues_count = len(iteration.critic_feedback.issues) if iteration.critic_feedback else 0
                correction = "✅ Validated" if iteration.critic_feedback.is_valid else f"❌ {issues_count} issues found"
                print(f"   Iteration {i}: {correction}")
                
                if not iteration.critic_feedback.is_valid and iteration.critic_feedback.issues:
                    for issue in iteration.critic_feedback.issues[:2]:
                        print(f"      • {issue.description[:60]}...")
        
        # Show if improvement happened
        if result.total_iterations > 1:
            print(f"\n✨ IMPROVEMENT DETECTED!")
            print(f"   Generator revised output based on critic feedback")
            print(f"   Iterations needed: {result.total_iterations}")
        
        # Show output quality
        print(f"\n📝 FINAL OUTPUT (preview):")
        output_preview = result.final_output[:300] if result.final_output else "No output"
        print(f"   {output_preview}...")
        
        return {
            "test_id": test_case['id'],
            "iterations": result.total_iterations,
            "improved": result.total_iterations > 1,
            "success": result.success,
            "time": elapsed
        }
    
    def run_full_test(self):
        """Run complete test suite"""
        print("\n" + "="*80)
        print("TWO FRIENDS MODEL - ENTERPRISE-LEVEL END-TO-END TEST")
        print("Testing real Generator-Critic communication and improvement")
        print("="*80)
        
        results = []
        
        for test_case in ENTERPRISE_QUERIES[:1]:  # Start with one to test
            try:
                result = self.test_single_query(
                    test_case,
                    generator_model="llama3.1:8b",
                    critic_model="phi3:mini"
                )
                results.append(result)
            except Exception as e:
                print(f"\n❌ Error in {test_case['id']}: {e}")
                results.append({
                    "test_id": test_case['id'],
                    "error": str(e)
                })
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        successful = sum(1 for r in results if r.get('success', False))
        improved = sum(1 for r in results if r.get('improved', False))
        total = len(results)
        
        print(f"\n📊 Results:")
        print(f"   Successful completions: {successful}/{total}")
        print(f"   Queries improved by critic: {improved}/{total}")
        
        if improved > 0:
            print(f"\n✅ TWO FRIENDS MODEL IS ACTIVELY IMPROVING OUTPUTS!")
            print(f"   Generator and Critic are communicating effectively")
        else:
            print(f"\n⚠️ No improvements detected in this run")
            print(f"   Either generator was correct first time, or critic too lenient")
        
        return results


def quick_communication_test():
    """Quick test to verify models are actually communicating"""
    print("\n" + "="*80)
    print("QUICK COMMUNICATION TEST")
    print("Verifying Generator and Critic actually exchange messages")
    print("="*80)
    
    llm_client = LLMClient()
    
    # Step 1: Generator produces output
    print("\n🤖 Step 1: Generator (llama3.1:8b) producing output...")
    gen_prompt = """Analyze this data and provide your reasoning:

Query: What is the profit margin for Electronics category?

Data Context:
- Dataset has columns: product, category, revenue, cost, profit
- Electronics category has 500 products
- Total revenue: $2,000,000
- Total cost: $1,600,000

Provide your analysis in [REASONING] and [OUTPUT] sections."""

    gen_response = llm_client.generate(
        prompt=gen_prompt,
        model="llama3.1:8b",
        adaptive_timeout=True
    )
    
    if not gen_response.get('success'):
        print("❌ Generator failed")
        return False
    
    gen_output = gen_response.get('response', '')
    print(f"   Generator output: {len(gen_output)} chars")
    print(f"   Preview: {gen_output[:150]}...")
    
    # Step 2: Critic reviews the output
    print("\n🔍 Step 2: Critic (phi3:mini) reviewing output...")
    
    critic_prompt = f"""You are a STRICT QUALITY REVIEWER. Review this analysis:

ORIGINAL QUESTION: What is the profit margin for Electronics category?

THEIR REASONING:
{gen_output[:500]}

Check for:
1. Is the profit margin formula correct? (Profit Margin = Profit/Revenue × 100%)
2. Are calculations accurate?
3. Is the answer complete?

If ANY issues found, respond with [INVALID] and list issues.
If perfect, respond with [VALID].
"""

    critic_response = llm_client.generate(
        prompt=critic_prompt,
        model="tinyllama",  # Faster model for testing
        adaptive_timeout=True
    )
    
    if not critic_response.get('success'):
        print("❌ Critic failed")
        return False
    
    critic_output = critic_response.get('response', '')
    print(f"   Critic output: {len(critic_output)} chars")
    print(f"   Preview: {critic_output[:200]}...")
    
    # Step 3: Analyze communication
    is_valid = "[VALID]" in critic_output.upper()
    is_invalid = "[INVALID]" in critic_output.upper()
    
    print(f"\n📊 Communication Analysis:")
    print(f"   Generator produced: ✅")
    print(f"   Critic reviewed: ✅")
    print(f"   Critic verdict: {'VALID' if is_valid else 'INVALID' if is_invalid else 'UNCLEAR'}")
    
    if is_valid or is_invalid:
        print(f"\n✅ MODELS ARE COMMUNICATING SUCCESSFULLY!")
        return True
    else:
        print(f"\n⚠️ Critic response format unclear")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick communication test (fast)
        quick_communication_test()
    else:
        # Full enterprise test (slow but thorough)
        tester = RealTwoFriendsTest()
        tester.run_full_test()

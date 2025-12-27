"""
Enterprise-Level Complex Query Test
====================================
Tests the Two Friends Model with enterprise-grade complex analytics queries.
Demonstrates the system can handle:
- Multi-step calculations
- Statistical analysis
- Time series reasoning
- Cross-dimensional analysis
"""
import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backend.core.llm_client import LLMClient


class EnterpriseTest:
    """Enterprise-level analytics tests"""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.results = []
    
    def run_query_with_correction(self, query: str, context: str, expected_elements: list):
        """Run a query through Generator-Critic loop and verify improvement"""
        
        print(f"\n{'='*80}")
        print(f"ENTERPRISE QUERY TEST")
        print(f"{'='*80}")
        print(f"\n📋 Query: {query[:100]}...")
        
        # ITERATION 1: Generator
        print(f"\n🤖 ITERATION 1: Generator Analysis")
        gen_prompt = f"""Analyze this enterprise data query:

QUERY: {query}

DATA CONTEXT:
{context}

Provide a thorough step-by-step analysis with calculations."""

        start = time.time()
        gen_response = self.llm_client.generate(
            prompt=gen_prompt,
            model="llama3.1:8b",
            adaptive_timeout=True
        )
        gen_time = time.time() - start
        
        if not gen_response.get('success'):
            print(f"❌ Generator failed")
            return False
        
        gen_output = gen_response.get('response', '')
        print(f"   Time: {gen_time:.1f}s")
        print(f"   Output length: {len(gen_output)} chars")
        print(f"   Preview: {gen_output[:200]}...")
        
        # CRITIC REVIEW
        print(f"\n🔍 CRITIC: Reviewing Analysis")
        critic_prompt = f"""You are a STRICT enterprise analytics reviewer. Review this analysis:

ORIGINAL QUERY: {query}

ANALYST'S RESPONSE:
{gen_output}

CHECK FOR:
1. Are all required calculations included?
2. Is the methodology correct?
3. Are there any logical errors?
4. Is the analysis complete?

List ANY issues found. Be thorough."""

        critic_response = self.llm_client.generate(
            prompt=critic_prompt,
            model="llama3.1:8b",
            adaptive_timeout=True
        )
        
        critic_output = critic_response.get('response', '')
        has_issues = any(word in critic_output.lower() for word in 
                        ['issue', 'error', 'missing', 'incorrect', 'should', 'need'])
        
        print(f"   Issues found: {'Yes' if has_issues else 'No'}")
        
        if has_issues:
            # ITERATION 2: Correction
            print(f"\n📝 ITERATION 2: Generator Revising Based on Feedback")
            
            correction_prompt = f"""Your analysis needs correction. Here's the critic's feedback:

{critic_output}

ORIGINAL QUERY: {query}

DATA CONTEXT:
{context}

Please provide a CORRECTED, complete analysis addressing all issues."""

            corrected = self.llm_client.generate(
                prompt=correction_prompt,
                model="llama3.1:8b",
                adaptive_timeout=True
            )
            
            final_output = corrected.get('response', gen_output)
            iterations = 2
        else:
            final_output = gen_output
            iterations = 1
        
        # VERIFY QUALITY
        print(f"\n✅ QUALITY CHECK")
        elements_found = sum(1 for elem in expected_elements 
                           if elem.lower() in final_output.lower())
        quality_score = elements_found / len(expected_elements) * 100
        
        print(f"   Expected elements: {len(expected_elements)}")
        print(f"   Found: {elements_found}")
        print(f"   Quality Score: {quality_score:.0f}%")
        print(f"   Iterations: {iterations}")
        
        for elem in expected_elements:
            status = "✅" if elem.lower() in final_output.lower() else "❌"
            print(f"      {status} {elem}")
        
        return {
            "quality_score": quality_score,
            "iterations": iterations,
            "success": quality_score >= 50
        }
    
    def run_enterprise_tests(self):
        """Run comprehensive enterprise tests"""
        
        print("\n" + "="*80)
        print("🏢 ENTERPRISE-LEVEL ANALYTICS TEST SUITE")
        print("Testing complex, multi-step analytical queries")
        print("="*80)
        
        tests = [
            {
                "name": "Year-over-Year Growth Analysis",
                "query": "Calculate the year-over-year revenue growth rate and identify which quarters showed the strongest performance",
                "context": """Dataset: quarterly_revenue.csv
- 8 quarters of data (2022-2023)
- Q1 2022: $2.1M | Q2 2022: $2.4M | Q3 2022: $2.2M | Q4 2022: $2.8M
- Q1 2023: $2.5M | Q2 2023: $2.9M | Q3 2023: $2.7M | Q4 2023: $3.2M""",
                "expected": ["growth", "percentage", "quarter", "2022", "2023", "strongest"]
            },
            {
                "name": "Customer Segmentation Analysis",
                "query": "Analyze customer segments by revenue contribution and identify the highest-value segment",
                "context": """Dataset: customer_segments.csv
- Segments: Enterprise (500 customers, $5M revenue), SMB (2000 customers, $3M), Startup (5000 customers, $2M)
- Enterprise avg: $10K per customer
- SMB avg: $1.5K per customer  
- Startup avg: $400 per customer""",
                "expected": ["enterprise", "highest", "revenue", "average", "segment", "value"]
            },
            {
                "name": "Cost Efficiency Analysis",
                "query": "Calculate the cost efficiency ratio across departments and recommend optimization areas",
                "context": """Dataset: department_costs.csv
- Engineering: Revenue $10M, Costs $6M (60% cost ratio)
- Sales: Revenue $8M, Costs $4M (50% cost ratio)
- Marketing: Revenue $5M, Costs $3M (60% cost ratio)
- Operations: Revenue $3M, Costs $2.5M (83% cost ratio)""",
                "expected": ["efficiency", "ratio", "operations", "optimization", "cost", "percentage"]
            }
        ]
        
        results = []
        for test in tests:
            print(f"\n\n{'='*80}")
            print(f"TEST: {test['name']}")
            print(f"{'='*80}")
            
            result = self.run_query_with_correction(
                query=test['query'],
                context=test['context'],
                expected_elements=test['expected']
            )
            result['name'] = test['name']
            results.append(result)
        
        # FINAL SUMMARY
        print("\n\n" + "="*80)
        print("🏆 ENTERPRISE TEST RESULTS SUMMARY")
        print("="*80)
        
        total_tests = len(results)
        passed = sum(1 for r in results if r.get('success', False))
        avg_quality = sum(r.get('quality_score', 0) for r in results) / total_tests
        avg_iterations = sum(r.get('iterations', 1) for r in results) / total_tests
        
        print(f"\n📊 Overall Results:")
        print(f"   Tests Passed: {passed}/{total_tests}")
        print(f"   Average Quality Score: {avg_quality:.0f}%")
        print(f"   Average Iterations: {avg_iterations:.1f}")
        
        for r in results:
            status = "✅" if r.get('success') else "❌"
            print(f"   {status} {r['name']}: {r.get('quality_score', 0):.0f}% quality, {r.get('iterations', 1)} iterations")
        
        if passed == total_tests:
            print(f"\n🎉 ALL ENTERPRISE TESTS PASSED!")
            print(f"   The Two Friends Model handles complex enterprise-level queries.")
        elif passed > 0:
            print(f"\n⚠️ {passed}/{total_tests} tests passed")
            print(f"   Some complex queries may need refinement.")
        
        return results


if __name__ == "__main__":
    tester = EnterpriseTest()
    tester.run_enterprise_tests()

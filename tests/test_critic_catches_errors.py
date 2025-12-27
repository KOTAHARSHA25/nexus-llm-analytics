"""
Critic Validation Test - Does the Critic Actually Catch Errors?
================================================================
This test INJECTS deliberate errors to verify the critic works.

Unlike test_two_friends_validation.py which tests end-to-end,
this test isolates the critic to verify it catches known errors.
"""
import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backend.core.llm_client import LLMClient
from src.backend.core.cot_parser import CoTParser
from src.backend.core.automated_validation import AutomatedValidator


class CriticErrorCatchTest:
    """Test if critic catches deliberately injected errors"""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.cot_parser = CoTParser()
        self.auto_validator = AutomatedValidator()
        
        # Load config
        config_path = project_root / 'config' / 'cot_review_config.json'
        with open(config_path, 'r') as f:
            self.config = json.load(f)['cot_review']
        
        self.critic_model = 'phi3:mini'
        self.results = {"passed": 0, "failed": 0, "errors": []}
    
    def _build_critic_prompt(self, query: str, data_context: dict, 
                             reasoning: str, output: str) -> str:
        """Build critic prompt"""
        prompt_path = project_root / 'src' / 'backend' / 'prompts' / 'cot_critic_prompt.txt'
        with open(prompt_path, 'r') as f:
            template = f.read()
        
        data_str = json.dumps(data_context, indent=2) if data_context else "No specific data"
        
        return template.format(
            query=query,
            data_context=data_str,
            cot_reasoning=reasoning,
            final_output=output
        )
    
    def test_critic(self, test_case: dict) -> dict:
        """Test if critic catches the injected error"""
        print(f"\n{'='*70}")
        print(f"TEST: {test_case['name']}")
        print(f"Expected: Critic should return [INVALID]")
        print(f"Error Type: {test_case['error_type']}")
        print(f"{'='*70}")
        
        # First: Test automated validator
        auto_result = self.auto_validator.validate(
            query=test_case['query'],
            reasoning=test_case['bad_reasoning'],
            output=test_case['bad_output'],
            data_context=test_case.get('data_context')
        )
        
        print(f"\n🔧 Automated Validator:")
        print(f"   Valid: {auto_result.is_valid}")
        if auto_result.issues:
            for issue in auto_result.issues:
                print(f"   - {issue.severity}: {issue.description}")
        
        # If automated validator catches it, great!
        if not auto_result.is_valid:
            print(f"✅ AUTOMATED VALIDATOR caught the error!")
            self.results["passed"] += 1
            return {"test": test_case['name'], "caught_by": "automated", "passed": True}
        
        # Second: Test LLM critic
        prompt = self._build_critic_prompt(
            query=test_case['query'],
            data_context=test_case.get('data_context', {}),
            reasoning=test_case['bad_reasoning'],
            output=test_case['bad_output']
        )
        
        print(f"\n🤖 Calling Critic ({self.critic_model})...")
        
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                model=self.critic_model,
                adaptive_timeout=True
            )
            
            if not response.get('success'):
                print(f"❌ Critic call failed")
                self.results["failed"] += 1
                return {"test": test_case['name'], "error": "critic_call_failed", "passed": False}
            
            critic_output = response.get('response', '')
            
            # Check if critic returned INVALID
            is_invalid = '[INVALID]' in critic_output.upper()
            is_valid = '[VALID]' in critic_output.upper()
            
            print(f"\n📋 Critic Response Preview:")
            print(f"   {critic_output[:200]}...")
            
            if is_invalid:
                print(f"\n✅ CRITIC correctly identified the error!")
                self.results["passed"] += 1
                return {"test": test_case['name'], "caught_by": "critic", "passed": True}
            elif is_valid:
                print(f"\n❌ CRITIC MISSED THE ERROR - returned [VALID]")
                self.results["failed"] += 1
                self.results["errors"].append({
                    "test": test_case['name'],
                    "error_type": test_case['error_type'],
                    "critic_output": critic_output[:500]
                })
                return {"test": test_case['name'], "missed": True, "passed": False}
            else:
                print(f"\n⚠️ CRITIC gave unclear response (no [VALID] or [INVALID])")
                self.results["failed"] += 1
                return {"test": test_case['name'], "unclear": True, "passed": False}
                
        except Exception as e:
            print(f"❌ Error: {e}")
            self.results["failed"] += 1
            return {"test": test_case['name'], "error": str(e), "passed": False}
    
    def run_all_tests(self):
        """Run all error injection tests"""
        
        print("\n" + "="*70)
        print("CRITIC ERROR DETECTION TEST")
        print("Testing if the critic catches deliberately injected errors")
        print("="*70)
        
        test_cases = [
            # Test 1: Basic Calculation Error
            {
                "name": "calculation_error_simple",
                "error_type": "Wrong arithmetic",
                "query": "What is the total revenue?",
                "data_context": {"columns": ["product", "revenue"], "stats": "5 products"},
                "bad_reasoning": """
Step 1: I need to sum the revenue column
Step 2: Revenue values are: 100, 200, 300, 400, 500
Step 3: Adding them: 100 + 200 = 300, 300 + 300 = 600, 600 + 400 = 1000, 1000 + 500 = 1600
Step 4: Total revenue is 1600
                """,
                "bad_output": "The total revenue is $1600"
                # REAL ANSWER: 100+200+300+400+500 = 1500, not 1600
            },
            
            # Test 2: Logic Inversion Error
            {
                "name": "logic_inversion",
                "error_type": "Inverted filter logic",
                "query": "Find all customers with age greater than 30",
                "data_context": {"columns": ["name", "age", "city"]},
                "bad_reasoning": """
Step 1: I need to filter customers by age
Step 2: The query asks for age greater than 30
Step 3: I'll use the condition age < 30 to filter
Step 4: This gives me customers above 30
                """,
                "bad_output": "Filtered using age < 30 to get customers above 30"
                # ERROR: age < 30 gets customers BELOW 30, not above
            },
            
            # Test 3: Wrong Column Reference
            {
                "name": "wrong_column",
                "error_type": "Using wrong column",
                "query": "What is the average profit margin?",
                "data_context": {"columns": ["product", "revenue", "cost", "profit"]},
                "bad_reasoning": """
Step 1: To calculate profit margin, I need profit and revenue
Step 2: Profit margin = revenue / cost
Step 3: Let me calculate this for each product
Step 4: The average profit margin is 42%
                """,
                "bad_output": "Average profit margin is 42%"
                # ERROR: Profit margin = profit/revenue, NOT revenue/cost
            },
            
            # Test 4: Missing Filter Step
            {
                "name": "missing_filter",
                "error_type": "Forgot to apply filter",
                "query": "What is total sales for Electronics category?",
                "data_context": {"columns": ["product", "category", "sales"]},
                "bad_reasoning": """
Step 1: I need to find total sales
Step 2: I'll sum the sales column
Step 3: Sum of all sales = $50,000
Step 4: Total sales is $50,000
                """,
                "bad_output": "Total sales for Electronics: $50,000"
                # ERROR: Summed ALL sales, not filtered by Electronics first
            },
            
            # Test 5: Percentage Without Multiply by 100
            {
                "name": "percentage_error",
                "error_type": "Forgot to multiply by 100 for percentage",
                "query": "What percentage of products are profitable?",
                "data_context": {"columns": ["product", "is_profitable"], "stats": "100 products, 75 profitable"},
                "bad_reasoning": """
Step 1: Count profitable products = 75
Step 2: Count total products = 100
Step 3: Percentage = 75 / 100 = 0.75
Step 4: 0.75 of products are profitable
                """,
                "bad_output": "0.75 of products are profitable"
                # ERROR: Should say "75%" not "0.75" for percentage
            },
            
            # Test 6: Causation vs Correlation
            {
                "name": "causation_confusion",
                "error_type": "Confusing correlation with causation",
                "query": "Does advertising spend cause higher sales?",
                "data_context": {"columns": ["month", "ad_spend", "sales"]},
                "bad_reasoning": """
Step 1: I'll check the correlation between ad_spend and sales
Step 2: Correlation coefficient = 0.82 (strong positive)
Step 3: Since correlation is high, ad_spend CAUSES higher sales
Step 4: Yes, advertising causes higher sales
                """,
                "bad_output": "Yes, advertising spend causes higher sales (correlation = 0.82)"
                # ERROR: Correlation doesn't prove causation!
            },
            
            # Test 7: Group By Error
            {
                "name": "groupby_error",
                "error_type": "Aggregate without proper grouping",
                "query": "What is average sales per region?",
                "data_context": {"columns": ["region", "store", "sales"]},
                "bad_reasoning": """
Step 1: I need average sales by region
Step 2: Total sales = $100,000
Step 3: Number of rows = 1000
Step 4: Average = $100,000 / 1000 = $100 per region
                """,
                "bad_output": "Average sales per region: $100"
                # ERROR: Calculated overall average, not grouped by region
            },
            
            # Test 8: Off by One Month Error
            {
                "name": "timeframe_error",
                "error_type": "Wrong time period",
                "query": "What was Q4 revenue?",
                "data_context": {"columns": ["month", "revenue"], "stats": "12 months of data"},
                "bad_reasoning": """
Step 1: Q4 means the fourth quarter
Step 2: Q4 includes months 9, 10, 11 (September, October, November)
Step 3: I'll sum revenue for those 3 months
Step 4: Q4 revenue is $300,000
                """,
                "bad_output": "Q4 revenue: $300,000"
                # ERROR: Q4 is months 10, 11, 12 (Oct, Nov, Dec), not 9, 10, 11
            },
        ]
        
        results = []
        for test_case in test_cases:
            result = self.test_critic(test_case)
            results.append(result)
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        passed = self.results["passed"]
        failed = self.results["failed"]
        total = passed + failed
        
        print(f"\n✅ Passed: {passed}/{total} ({100*passed/total:.1f}%)")
        print(f"❌ Failed: {failed}/{total} ({100*failed/total:.1f}%)")
        
        if self.results["errors"]:
            print(f"\n⚠️ Errors the critic MISSED:")
            for err in self.results["errors"]:
                print(f"   - {err['test']}: {err['error_type']}")
        
        # Verdict
        print("\n" + "="*70)
        if passed / total >= 0.8:
            print("🎉 CRITIC IS WORKING WELL (≥80% detection rate)")
        elif passed / total >= 0.5:
            print("⚠️ CRITIC NEEDS IMPROVEMENT (50-80% detection rate)")
        else:
            print("❌ CRITIC IS TOO LENIENT (<50% detection rate)")
        print("="*70)
        
        return results


def main():
    tester = CriticErrorCatchTest()
    tester.run_all_tests()


if __name__ == "__main__":
    main()

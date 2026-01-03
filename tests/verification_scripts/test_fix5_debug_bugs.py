"""
Debug Fix 5 Template Bugs - Reproduce Undefined Variables
Run specific failing scenarios to see generated code
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import pandas as pd
from backend.core.code_generator import CodeGenerator
from backend.core.llm_client import LLMClient

print("="*80)
print("FIX 5 BUG DEBUG: Reproducing Undefined Variable Errors")
print("="*80)

# Load real sample data
df = pd.read_csv('data/samples/sales_data.csv')
print(f"\nDataset: {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {df.columns.tolist()}")

# Initialize
llm_client = LLMClient()
generator = CodeGenerator(llm_client)

# Test the specific failing scenarios from comprehensive test
failing_scenarios = [
    {
        "id": 6,
        "query": "Calculate the standard deviation of revenue",
        "error_reported": "StringIO not defined"
    },
    {
        "id": 7,
        "query": "How many products have sales greater than 5000?",
        "error_reported": "undefined variables"
    }
]

print("\n" + "="*80)
print("REPRODUCING BUGS FROM FAILING SCENARIOS")
print("="*80)

for scenario in failing_scenarios:
    print(f"\n{'='*80}")
    print(f"Scenario {scenario['id']}: {scenario['query']}")
    print(f"Reported Error: {scenario['error_reported']}")
    print(f"{'='*80}")
    
    # Generate with phi3:mini (small model using simple template)
    print("\nGenerating with phi3:mini (using simple template)...")
    
    try:
        generated = generator.generate_code(scenario['query'], df, "phi3:mini")
        
        if generated.is_valid:
            print(f"Code generated successfully")
            print(f"\nGENERATED CODE:")
            print("-" * 80)
            print(generated.code)
            print("-" * 80)
            
            # Try to execute it
            print(f"\nExecuting code...")
            result = generator.execute_code(generated.code, df)
            
            if result.success:
                print(f"EXECUTION SUCCESS")
                print(f"Result: {result.result}")
            else:
                print(f"EXECUTION FAILED")
                print(f"Error: {result.error}")
                
                # Analyze the error
                error_str = str(result.error)
                if "not defined" in error_str.lower() or "nameerror" in error_str.lower():
                    print(f"\nUNDEFINED VARIABLE DETECTED!")
                    print(f"This is the bug we need to fix.")
                    
                    # Try to extract what variable is undefined
                    import re
                    match = re.search(r"name '(\w+)' is not defined", error_str)
                    if match:
                        undefined_var = match.group(1)
                        print(f"Undefined variable: '{undefined_var}'")
                        
                        # Check if it appears in the code
                        if undefined_var in generated.code:
                            print(f"Variable '{undefined_var}' found in generated code")
                            print(f"This means LLM hallucinated this variable name")
        else:
            print(f"CODE GENERATION FAILED")
            print(f"Error: {generated.error_message}")
            
    except Exception as e:
        print(f"EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("DEBUG SUMMARY")
print("="*80)
print("""
Next Steps:
1. Review generated code above to identify undefined variables
2. Check if simple template causes confusion
3. Test if issue is in prompt or LLM hallucination
4. Implement fix (add validation or improve template)
""")
print("="*80)

"""
Debug Scenario 6 specifically - std deviation failures
Run it 10 times to catch the intermittent failures
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
print("DEBUG: Scenario 6 - Standard Deviation (10 runs)")
print("="*80)

df = pd.read_csv('data/samples/sales_data.csv')
print(f"\nDataset: {len(df)} rows, {len(df.columns)} columns")

# Expected answer
expected_std = df['revenue'].std()
print(f"Expected std dev of revenue: {expected_std:.2f}")

llm_client = LLMClient()
generator = CodeGenerator(llm_client)

query = "Calculate the standard deviation of revenue"

print(f"\nRunning query 10 times: '{query}'")
print("="*80)

successes = 0
failures = 0
errors = []

for run in range(1, 11):
    print(f"\nRun {run}/10...", end=" ")
    
    try:
        generated = generator.generate_code(query, df, "phi3:mini")
        
        if not generated.is_valid:
            print(f"GENERATION FAILED")
            failures += 1
            errors.append({
                "run": run,
                "stage": "generation",
                "error": generated.error_message
            })
            continue
        
        result = generator.execute_code(generated.code, df)
        
        if result.success and result.result is not None:
            actual = result.result
            if isinstance(actual, dict) and 'result' in actual:
                actual = actual['result']
            
            # Check if correct
            try:
                if abs(float(actual) - expected_std) < 1.0:
                    print(f"SUCCESS (result: {float(actual):.2f})")
                    successes += 1
                else:
                    print(f"WRONG VALUE (got {actual}, expected {expected_std:.2f})")
                    failures += 1
                    errors.append({
                        "run": run,
                        "stage": "validation",
                        "code": generated.code,
                        "result": actual,
                        "error": "Incorrect value"
                    })
            except:
                print(f"VALIDATION ERROR")
                failures += 1
                errors.append({
                    "run": run,
                    "stage": "validation",
                    "code": generated.code,
                    "result": actual,
                    "error": "Could not validate result"
                })
        else:
            print(f"EXECUTION FAILED")
            failures += 1
            errors.append({
                "run": run,
                "stage": "execution",
                "code": generated.code,
                "error": result.error
            })
    except Exception as e:
        print(f"EXCEPTION: {e}")
        failures += 1
        errors.append({
            "run": run,
            "stage": "exception",
            "error": str(e)
        })

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Successes: {successes}/10 ({successes*10}%)")
print(f"Failures: {failures}/10 ({failures*10}%)")

if errors:
    print(f"\n{len(errors)} ERROR(S) ENCOUNTERED:")
    for err in errors:
        print(f"\n--- Run {err['run']} ---")
        print(f"Stage: {err['stage']}")
        if 'code' in err:
            print(f"Generated code:")
            print(err['code'])
        print(f"Error: {err['error']}")

print("="*80)

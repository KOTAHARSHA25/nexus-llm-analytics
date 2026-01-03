"""
Test Fix 5: Result Accuracy with Real Sample Data
Focus on OUTPUT CORRECTNESS, not just code execution
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
print("🎯 FIX 5 ACCURACY TEST: Verifying Result Correctness")
print("="*80)

# Load real sample data
df = pd.read_csv('data/samples/sales_data.csv')
print(f"\n📊 Loaded: {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 3 rows:")
print(df.head(3))

# Calculate expected answers manually
print("\n" + "="*80)
print("📐 CALCULATING EXPECTED ANSWERS")
print("="*80)

expected_answers = {
    "highest_sales": {
        "value": df['sales'].max(),
        "product": df.loc[df['sales'].idxmax(), 'product']
    },
    "total_revenue": df['revenue'].sum(),
    "average_price": df['price'].mean(),
    "top_3_sales": df.nlargest(3, 'sales')['product'].tolist(),
    "region_count": df['region'].nunique()
}

print(f"\n✓ Highest sales: {expected_answers['highest_sales']['value']} ({expected_answers['highest_sales']['product']})")
print(f"✓ Total revenue: {expected_answers['total_revenue']}")
print(f"✓ Average price: {expected_answers['average_price']:.2f}")
print(f"✓ Top 3 by sales: {expected_answers['top_3_sales']}")
print(f"✓ Region count: {expected_answers['region_count']}")

# Initialize code generator
try:
    llm_client = LLMClient()
    generator = CodeGenerator(llm_client)
    print("\n✅ CodeGenerator initialized")
except Exception as e:
    print(f"\n❌ Error initializing: {e}")
    print("Make sure Ollama is running!")
    sys.exit(1)

# Get available models
import requests
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    models = [m['name'] for m in response.json().get('models', []) if 'embed' not in m['name'].lower()]
    print(f"✅ Found {len(models)} models: {models}")
except:
    print("⚠️  Using default models")
    models = ["phi3:mini", "llama3.1:8b"]

# Test queries with expected answers
test_cases = [
    {
        "query": "What is the highest sales value?",
        "expected": expected_answers['highest_sales']['value'],
        "check": lambda result: abs(float(result) - expected_answers['highest_sales']['value']) < 0.01,
        "description": "Maximum sales"
    },
    {
        "query": "Which product has the highest sales?",
        "expected": expected_answers['highest_sales']['product'],
        "check": lambda result: expected_answers['highest_sales']['product'] in str(result),
        "description": "Product with max sales"
    },
    {
        "query": "What is the total revenue?",
        "expected": expected_answers['total_revenue'],
        "check": lambda result: abs(float(result) - expected_answers['total_revenue']) < 0.01,
        "description": "Sum of revenue"
    },
    {
        "query": "What is the average price?",
        "expected": expected_answers['average_price'],
        "check": lambda result: abs(float(result) - expected_answers['average_price']) < 0.01,
        "description": "Mean price"
    },
    {
        "query": "Show me the top 3 products by sales",
        "expected": expected_answers['top_3_sales'],
        "check": lambda result: any(p in str(result) for p in expected_answers['top_3_sales'][:2]),
        "description": "Top 3 ranking"
    }
]

# Test with different models
results_summary = []

for model in ["phi3:mini", "llama3.1:8b"]:  # Test key models for Fix 5
    print("\n" + "="*80)
    print(f"🤖 TESTING WITH MODEL: {model}")
    print("="*80)
    
    model_results = {"model": model, "correct": 0, "total": len(test_cases)}
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}/{len(test_cases)}: {test['description']}")
        print(f"   Query: {test['query']}")
        print(f"   Expected: {test['expected']}")
        
        try:
            # Generate code
            generated = generator.generate_code(test['query'], df, model)
            
            if not generated.is_valid:
                print(f"   ❌ GENERATION FAILED - {generated.error_message}")
                continue
            
            # Execute the generated code
            result = generator.execute_code(generated.code, df)
            
            if result.success and result.result is not None:
                # Extract actual value from result
                actual = result.result
                if isinstance(actual, dict) and 'result' in actual:
                    actual = actual['result']
                    
                print(f"   Got: {actual}")
                
                # Check if answer is correct
                try:
                    is_correct = test['check'](actual)
                except Exception as check_err:
                    print(f"   ⚠️  Check error: {check_err}, trying alternative...")
                    # Handle pandas/numpy types
                    try:
                        if hasattr(actual, 'item'):  # numpy scalar
                            is_correct = test['check'](actual.item())
                        elif isinstance(actual, pd.DataFrame):
                            is_correct = test['check'](actual)
                        else:
                            is_correct = False
                    except:
                        is_correct = False
                
                if is_correct:
                    print(f"   ✅ CORRECT - Answer matches expected value")
                    model_results["correct"] += 1
                else:
                    print(f"   ❌ WRONG - Answer does not match")
                    print(f"   Generated code:\n{generated.code[:300]}...")
            else:
                print(f"   ❌ EXECUTION FAILED - {result.error}")
                if generated.code:
                    print(f"   Code was:\n{generated.code[:200]}...")
                
        except Exception as e:
            print(f"   ❌ ERROR - {e}")
    
    # Model summary
    accuracy = (model_results["correct"] / model_results["total"]) * 100
    print(f"\n📊 {model} Accuracy: {model_results['correct']}/{model_results['total']} ({accuracy:.1f}%)")
    results_summary.append({
        "model": model,
        "correct": model_results["correct"],
        "total": model_results["total"],
        "accuracy": accuracy
    })

# Final summary
print("\n" + "="*80)
print("🎯 ACCURACY SUMMARY - Fix 5 Results")
print("="*80)

for result in results_summary:
    status = "✅" if result["accuracy"] >= 80 else "⚠️" if result["accuracy"] >= 60 else "❌"
    print(f"{status} {result['model']:20s}: {result['correct']}/{result['total']} correct ({result['accuracy']:.1f}%)")

avg_accuracy = sum(r["accuracy"] for r in results_summary) / len(results_summary)
print(f"\n📈 Average Accuracy: {avg_accuracy:.1f}%")

if avg_accuracy >= 80:
    print("\n✅ EXCELLENT: Fix 5 produces accurate results!")
elif avg_accuracy >= 60:
    print("\n⚠️  GOOD: Fix 5 works but needs improvement")
else:
    print("\n❌ POOR: Fix 5 needs debugging")

print("="*80)

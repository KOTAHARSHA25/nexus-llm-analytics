"""Phase 2 Real LLM Test - Uses actual Ollama model"""
import sys
sys.path.insert(0, 'src/backend')
import pandas as pd

print('='*70)
print('PHASE 2: REAL LLM CODE GENERATION TEST')
print('='*70)

# Load real data
df = pd.read_csv('data/samples/sales_data.csv')
print(f'Data: {len(df)} rows, {len(df.columns)} cols')
print(f'Columns: {list(df.columns)}\n')

from core.code_generator import CodeGenerator
from core.llm_client import LLMClient

# Create generator with real LLM
llm = LLMClient()
gen = CodeGenerator(llm_client=llm)

# Test queries - comprehensive coverage
queries = [
    # Simple aggregations
    "What is the total sales?",
    "What is the average revenue?",
    # Grouped operations
    "Show average revenue by region",
    "Total sales by product",
    # Filtering/selection
    "What product has the highest sales?",
    "Which region has the lowest revenue?",
    # Multiple operations
    "Show top 5 products by revenue",
]

for query in queries:
    print(f'--- Query: "{query}" ---')
    
    result = gen.generate_and_execute(
        query=query,
        df=df,
        model="phi3:mini",
        max_retries=3
    )
    
    if result.success:
        print(f'✅ Success ({result.execution_time_ms:.1f}ms)')
        print(f'Result: {result.result}')
        print(f'Code: {result.code[:100]}...')
    else:
        print(f'❌ Failed: {result.error}')
    print()

print('='*70)
print('PHASE 2 REAL LLM TEST COMPLETE')
print('='*70)

"""Test Code Execution History System - Comprehensive"""
import sys
sys.path.insert(0, 'src/backend')
import pandas as pd
from core.code_generator import get_code_generator
from core.code_execution_history import get_execution_history

print('='*70)
print('CODE EXECUTION HISTORY - COMPREHENSIVE TEST')
print('='*70)

# Create sample data
df = pd.read_csv('data/samples/sales_data.csv')
print(f'Data: {len(df)} rows, {len(df.columns)} cols\n')

# Initialize generator with history
gen = get_code_generator()

# Test queries
queries = [
    "What is the total sales?",
    "Show average revenue by region",
    "Which product has the highest sales?"
]

print('Executing queries and saving to history...\n')
for i, query in enumerate(queries, 1):
    print(f'[{i}] "{query}"')
    result = gen.generate_and_execute(
        query=query,
        df=df,
        model='phi3:mini',
        max_retries=2,
        data_file='sales_data.csv',
        save_history=True
    )
    status = "✅" if result.success else "❌"
    print(f'    {status} ID: {result.execution_id} | Time: {result.execution_time_ms:.1f}ms')
    if result.success:
        result_str = str(result.result)
        print(f'    Result: {result_str[:60]}...' if len(result_str) > 60 else f'    Result: {result_str}')
    else:
        print(f'    Error: {result.error}')
    print()

# Check history
print('='*70)
print('HISTORY RECORDS')
print('='*70)
history = get_execution_history()
records = history.get_recent_executions(limit=10)
print(f'Total records: {len(records)}\n')

for r in records:
    status = "✅" if r.success else "❌"
    print(f'{status} [{r.execution_id}] {r.query[:50]}')
    print(f'   Model: {r.model_used} | Time: {r.execution_time_ms:.1f}ms | Attempts: {r.attempt_count}')
    print(f'   Result type: {r.result_type}')
    if r.cleaned_code:
        code_preview = r.cleaned_code.replace('\n', ' ')[:80]
        print(f'   Code: {code_preview}...')
    print()

# Summary stats
print('='*70)
print('EXECUTION SUMMARY')
print('='*70)
summary = history.get_execution_summary()
print(f'Total executions: {summary["total_executions"]}')
print(f'Successful: {summary["successful"]} | Failed: {summary["failed"]}')
print(f'Success rate: {summary["success_rate"]:.1f}%')
print(f'Average execution time: {summary["avg_execution_time_ms"]:.1f}ms')
print(f'Models used: {", ".join(summary["models_used"])}')
print(f'Unique queries: {summary["unique_queries"]}')

print('\n' + '='*70)
print('✅ CODE EXECUTION HISTORY SYSTEM COMPLETE!')
print('='*70)

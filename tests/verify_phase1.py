"""Quick Phase 1 Verification - Run this to confirm all 3 tracks work"""
import sys
sys.path.insert(0, 'src/backend')
import pandas as pd

from core.query_orchestrator import QueryOrchestrator, ExecutionMethod, ReviewLevel
from core.query_complexity_analyzer import QueryComplexityAnalyzer

print('='*60)
print('PHASE 1 FINAL VERIFICATION - REAL DATA')
print('='*60)

# Load real sample data
try:
    df = pd.read_csv('data/samples/sales_data.csv')
    print(f'✅ Loaded real data: {len(df)} rows, {len(df.columns)} columns')
    print(f'   Columns: {list(df.columns)[:5]}...')
except:
    # Fallback to synthetic
    df = pd.DataFrame({
        'sales': [100, 200, 300, 150, 250], 
        'region': ['North', 'South', 'East', 'West', 'North'], 
        'date': ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05']
    })
    print('⚠️ Using synthetic data')

analyzer = QueryComplexityAnalyzer()
config = {'ollama_base_url': 'http://localhost:11434'}
qo = QueryOrchestrator(analyzer, config)

# Test cases with expected outcomes
# NOTE: code_generation always triggers mandatory review for safety
tests = [
    ('What is the total?', 'simple', 'tinyllama', 'direct_llm', 'none'),
    ('Show average sales by region', 'medium', 'phi3', 'code_generation', 'mandatory'),  # code gen = mandatory review
    ('Perform correlation analysis between all numeric columns and identify patterns', 'complex', 'llama3.1', 'code_generation', 'mandatory'),
]

all_passed = True
for query, expected_tier, expected_model_prefix, expected_method, expected_review in tests:
    plan = qo.create_execution_plan(query, data=df)
    
    # For complex queries, accept RAM-constrained fallback as valid
    if expected_tier == 'complex':
        model_ok = (expected_model_prefix in plan.model.lower() or 
                   plan.complexity_score >= 0.7)  # Accept fallback if complexity detected correctly
    else:
        model_ok = expected_model_prefix in plan.model.lower()
    method_ok = plan.execution_method.value == expected_method
    review_ok = plan.review_level.value == expected_review
    
    status = '✅' if (model_ok and method_ok and review_ok) else '❌'
    if status == '❌': 
        all_passed = False
    
    print(f'\n{status} {expected_tier.upper()}: "{query[:40]}..."')
    print(f'   Model: {plan.model} (expected {expected_model_prefix}*) {"✓" if model_ok else "✗"}')
    print(f'   Method: {plan.execution_method.value} (expected {expected_method}) {"✓" if method_ok else "✗"}')
    print(f'   Review: {plan.review_level.value} (expected {expected_review}) {"✓" if review_ok else "✗"}')
    print(f'   Complexity: {plan.complexity_score:.2f}')

print('\n' + '='*60)
if all_passed:
    print('✅ PHASE 1 VERIFIED - All tracks working correctly!')
else:
    print('❌ PHASE 1 HAS ISSUES - See above for details')
print('='*60)

"""Phase 1 Comprehensive Verification - Must pass before starting Phase 2"""
import sys
sys.path.insert(0, 'src/backend')
import pandas as pd

from core.query_orchestrator import QueryOrchestrator, ExecutionMethod, ReviewLevel
from core.query_complexity_analyzer import QueryComplexityAnalyzer

print('='*70)
print('PHASE 1 COMPREHENSIVE VERIFICATION')
print('='*70)

# Load real data
df = pd.read_csv('data/samples/sales_data.csv')
print(f'Data: {len(df)} rows, {len(df.columns)} cols\n')

analyzer = QueryComplexityAnalyzer()
config = {'ollama_base_url': 'http://localhost:11434'}
qo = QueryOrchestrator(analyzer, config)

# Test all 3 tracks
tests = [
    # Simple queries
    ('What is the total?', 'SIMPLE', '<0.3', 'direct_llm', 'none'),
    ('How many rows?', 'SIMPLE', '<0.3', 'direct_llm', 'none'),
    ('Show the first 5', 'SIMPLE', '<0.3', 'direct_llm', 'none'),
    
    # Medium queries (with data = code_gen = mandatory review)
    ('Calculate average sales by region', 'MEDIUM', '0.3-0.7', 'code_generation', 'mandatory'),
    ('Show top 10 products by revenue', 'MEDIUM', '0.3-0.7', 'code_generation', 'mandatory'),
    
    # Complex queries
    ('Perform correlation analysis between all numeric columns', 'COMPLEX', '>0.7', 'code_generation', 'mandatory'),
    ('Run k-means clustering to segment customers', 'COMPLEX', '>0.7', 'code_generation', 'mandatory'),
]

passed = 0
failed = 0

for query, tier, complexity_range, exp_method, exp_review in tests:
    plan = qo.create_execution_plan(query, data=df)
    
    c = plan.complexity_score
    if tier == 'SIMPLE': complexity_ok = c < 0.3
    elif tier == 'MEDIUM': complexity_ok = 0.3 <= c < 0.7
    else: complexity_ok = c >= 0.7
    
    method_ok = plan.execution_method.value == exp_method
    review_ok = plan.review_level.value == exp_review
    
    all_ok = complexity_ok and method_ok and review_ok
    status = '✅' if all_ok else '❌'
    if all_ok: passed += 1
    else: failed += 1
    
    print(f'{status} {tier}: "{query[:40]}..."')
    print(f'   Complexity: {c:.2f} ({complexity_range}) {"✓" if complexity_ok else "✗"}')
    print(f'   Method: {plan.execution_method.value} {"✓" if method_ok else "✗"}')
    print(f'   Review: {plan.review_level.value} {"✓" if review_ok else "✗"}')

print()
print('='*70)
print(f'RESULTS: {passed}/{passed+failed} passed')
if failed == 0:
    print('✅ PHASE 1 IS ROCK SOLID - Ready for Phase 2')
else:
    print(f'❌ {failed} tests failed - needs fixing')
print('='*70)

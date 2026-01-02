"""Phase 2 Code Generation Test"""
import sys
sys.path.insert(0, 'src/backend')
import pandas as pd

print('='*70)
print('PHASE 2: CODE GENERATION TEST')
print('='*70)

# Load real data
df = pd.read_csv('data/samples/sales_data.csv')
print(f'Data: {len(df)} rows, {len(df.columns)} cols')
print(f'Columns: {list(df.columns)}')
print()

# Test 1: Code Generator directly
print('--- TEST 1: Code Generator ---')
try:
    from core.code_generator import CodeGenerator
    
    # Create generator with mock LLM for testing
    class MockLLM:
        def generate(self, prompt, model=None):
            # Return a simple code response
            return '''```python
# Calculate average sales by region
result = df.groupby('region')['sales'].mean()
```'''
    
    gen = CodeGenerator(llm_client=MockLLM())
    
    # Generate code
    code_result = gen.generate_code(
        query="What is the average sales by region?",
        df=df,
        model="phi3:mini"
    )
    
    print(f'Code generated: {code_result.is_valid}')
    if code_result.is_valid:
        print(f'Code:\n{code_result.code}')
    else:
        print(f'Error: {code_result.error_message}')
    print()
    
    # Execute code
    print('--- TEST 2: Code Execution ---')
    exec_result = gen.execute_code(code_result.code, df)
    print(f'Execution success: {exec_result.success}')
    if exec_result.success:
        print(f'Result:\n{exec_result.result}')
        print(f'Time: {exec_result.execution_time_ms:.1f}ms')
    else:
        print(f'Error: {exec_result.error}')
    
    print()
    print('='*70)
    if exec_result.success:
        print('✅ PHASE 2 CODE GENERATION IS WORKING!')
    else:
        print('❌ Phase 2 has issues')
    print('='*70)
    
except Exception as e:
    import traceback
    print(f'Error: {e}')
    traceback.print_exc()

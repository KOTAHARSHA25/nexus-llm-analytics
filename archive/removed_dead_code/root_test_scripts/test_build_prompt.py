import sys, os
sys.path.insert(0, os.path.join(os.getcwd(),'src'))
from backend.utils.data_optimizer import DataOptimizer
filepath='data/samples/csv/sales_simple.csv'
optimizer=DataOptimizer(max_rows=5,max_chars=3000)
res=optimizer.optimize_for_llm(filepath)
print('PREVIEW FIRST 1000:\n')
print(res['preview'][:1000])

data_info=res['preview']
query='What is the total revenue?'
direct_prompt=f"""You are analyzing data from: sales_simple.csv

IMPORTANT INSTRUCTIONS:
1. The data below contains PRE-CALCULATED STATISTICS from the ENTIRE dataset
2. For aggregation questions (total, sum, average, count), USE ONLY THE PRE-CALCULATED VALUES
3. DO NOT attempt to calculate these yourself from the sample data
4. Answer in ONE clear sentence with the exact number shown

DATA INFORMATION:
{data_info}

QUESTION: {query}

Answer directly in 1 sentence. NO code, NO JSON, just the answer:"""
print('\n\nDIRECT PROMPT:\n\n')
print(direct_prompt)

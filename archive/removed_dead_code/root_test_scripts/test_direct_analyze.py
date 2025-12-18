"""Direct test of CrewManager analyze_structured_data to verify review disabled behavior"""
import sys, os, time
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from backend.core.crew_singleton import get_crew_manager

cm = get_crew_manager()
print('Calling handle_query for sales_simple.csv (enable_review default False)')
start=time.time()
res = cm.handle_query('What is the total revenue?', filename='sales_simple.csv')
elapsed=time.time()-start
print('Elapsed:', elapsed)
print('Result:')
print(res.get('result'))
print('--- full result ---')
print(res)

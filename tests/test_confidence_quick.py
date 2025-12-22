import sys, os
os.chdir(r'C:\Users\mitta\OneDrive\Documents\Major Project\Phase_2\nexus-llm-analytics-dist(Main)\nexus-llm-analytics-dist')
sys.path.insert(0, 'src')

from backend.plugins.data_analyst_agent import DataAnalystAgent
from backend.plugins.sql_agent import SQLAgent

# Initialize
da = DataAnalystAgent()
da.initialize()

sql = SQLAgent()
sql.initialize()

# Test confidence scoring
print("Testing confidence scores after fixes:")
print("=" * 60)

# Simple average query
query1 = "what is the average sales?"
print(f"\nQuery: '{query1}'")
print(f"  DataAnalyst (.csv): {da.can_handle(query1, '.csv'):.2f}")
print(f"  SQLAgent (.csv):    {sql.can_handle(query1, '.csv'):.2f}")

# SQL-specific query
query2 = "generate SQL query to find top customers"
print(f"\nQuery: '{query2}'")
print(f"  DataAnalyst (.csv): {da.can_handle(query2, '.csv'):.2f}")
print(f"  SQLAgent (.csv):    {sql.can_handle(query2, '.csv'):.2f}")

# Simple lookup
query3 = "what is the name?"
print(f"\nQuery: '{query3}'")
print(f"  DataAnalyst (.json): {da.can_handle(query3, '.json'):.2f}")
print(f"  SQLAgent (.json):    {sql.can_handle(query3, '.json'):.2f}")

print("\n" + "=" * 60)
print("✅ Test complete!")

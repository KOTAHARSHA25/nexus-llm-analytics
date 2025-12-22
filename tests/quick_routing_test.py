"""Quick Agent Routing Test"""
import sys
import os

# Setup paths
os.chdir('C:/Users/mitta/OneDrive/Documents/Major Project/Phase_2/nexus-llm-analytics-dist(Main)/nexus-llm-analytics-dist')
sys.path.insert(0, 'src')

from backend.core.plugin_system import get_agent_registry

print("=" * 80)
print("QUICK AGENT ROUTING TEST")
print("=" * 80)

# Get registry and discover agents
registry = get_agent_registry()
count = registry.discover_agents()

print(f"\n✅ Discovered {count} agents:")
for name in registry.registered_agents.keys():
    print(f"   - {name}")

# Test queries
test_queries = [
    ("Perform a t-test to compare sales between regions", "StatisticalAgent"),
    ("Forecast next quarter sales using ARIMA", "TimeSeriesAgent"),
    ("Calculate ROI and profitability ratios", "FinancialAgent"),
    ("Find customer segments using clustering", "MLInsightsAgent"),
    ("Generate SQL to find top customers", "EnhancedSQLAgent"),
    ("What is the average sales?", "DataAnalyst"),
]

print("\n" + "=" * 80)
print("ROUTING TESTS")
print("=" * 80)

correct = 0
total = len(test_queries)

for query, expected in test_queries:
    topic, conf, agent = registry.route_query(query, file_type="csv")
    actual = agent.__class__.__name__ if agent else "None"
    
    match = "✅" if actual == expected else "❌"
    print(f"\n{match} Query: {query[:60]}...")
    print(f"   Expected: {expected}")
    print(f"   Got: {actual} (confidence: {conf:.2f})")
    
    if actual == expected:
        correct += 1

print("\n" + "=" * 80)
print(f"RESULT: {correct}/{total} correct ({correct/total*100:.1f}%)")
print("=" * 80)

"""
PHASE 3: Plugin Agent System Testing
Tests all 5 specialized agents for initialization and execution
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 80)
print("PHASE 3: PLUGIN AGENT SYSTEM TESTING")
print("=" * 80)

# Import all agents
try:
    from backend.plugins.statistical_agent import StatisticalAgent
    from backend.plugins.financial_agent import FinancialAgent
    from backend.plugins.time_series_agent import TimeSeriesAgent
    from backend.plugins.ml_insights_agent import MLInsightsAgent
    from backend.plugins.sql_agent import SQLAgent
    print("✓ All 5 agents imported successfully\n")
except ImportError as e:
    print(f"✗ Failed to import agents: {e}")
    sys.exit(1)

total = 0
passed = 0

# Create test data
np.random.seed(42)
test_data = pd.DataFrame({
    'revenue': np.random.randint(1000, 5000, 100),
    'cost': np.random.randint(500, 3000, 100),
    'quantity': np.random.randint(10, 100, 100),
    'age': np.random.randint(25, 65, 100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})

# Add date column for time series
dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
test_data['date'] = dates

# TEST 1: Statistical Agent Initialization
print("[TEST 1] Statistical Agent - Initialization")
total += 1
stat_agent = StatisticalAgent()
if stat_agent.initialize():
    print("PASS: Statistical Agent initialized")
    print(f"  - Confidence level: {stat_agent.confidence_level}")
    passed += 1
else:
    print("FAIL: Statistical Agent initialization failed")

# TEST 2: Statistical Agent - Descriptive Statistics
print("\n[TEST 2] Statistical Agent - Descriptive Statistics")
total += 1
try:
    result = stat_agent.execute(
        query="Provide descriptive statistics for the revenue",
        data=test_data
    )
    if result.get('success') and 'statistics' in result:
        print("PASS: Descriptive statistics calculated")
        print(f"  - Mean revenue: {result['statistics'].get('mean', 'N/A')}")
        passed += 1
    else:
        print(f"FAIL: {result.get('error', 'Unknown error')}")
except Exception as e:
    print(f"FAIL: Exception - {str(e)[:100]}")

# TEST 3: Statistical Agent - Correlation
print("\n[TEST 3] Statistical Agent - Correlation Analysis")
total += 1
try:
    result = stat_agent.execute(
        query="Calculate correlation between revenue and quantity",
        data=test_data
    )
    if result.get('success'):
        print("PASS: Correlation analysis completed")
        passed += 1
    else:
        print(f"FAIL: {result.get('error', 'Unknown error')}")
except Exception as e:
    print(f"FAIL: Exception - {str(e)[:100]}")

# TEST 4: Financial Agent Initialization
print("\n[TEST 4] Financial Agent - Initialization")
total += 1
fin_agent = FinancialAgent()
if fin_agent.initialize():
    print("PASS: Financial Agent initialized")
    passed += 1
else:
    print("FAIL: Financial Agent initialization failed")

# TEST 5: Financial Agent - Profit Calculation
print("\n[TEST 5] Financial Agent - Profit Analysis")
total += 1
try:
    result = fin_agent.execute(
        query="Calculate total profit and profit margin",
        data=test_data
    )
    if result.get('success'):
        print("PASS: Financial analysis completed")
        print(f"  - Analysis type: {result.get('analysis_type', 'N/A')}")
        passed += 1
    else:
        print(f"FAIL: {result.get('error', 'Unknown error')}")
except Exception as e:
    print(f"FAIL: Exception - {str(e)[:100]}")

# TEST 6: Time Series Agent Initialization
print("\n[TEST 6] Time Series Agent - Initialization")
total += 1
ts_agent = TimeSeriesAgent()
if ts_agent.initialize():
    print("PASS: Time Series Agent initialized")
    passed += 1
else:
    print("FAIL: Time Series Agent initialization failed")

# TEST 7: Time Series Agent - Trend Detection
print("\n[TEST 7] Time Series Agent - Trend Detection")
total += 1
try:
    ts_data = test_data[['date', 'revenue']].copy()
    result = ts_agent.execute(
        query="Detect trend in revenue over time",
        data=ts_data
    )
    if result.get('success'):
        print("PASS: Trend detection completed")
        passed += 1
    else:
        print(f"FAIL: {result.get('error', 'Unknown error')}")
except Exception as e:
    print(f"FAIL: Exception - {str(e)[:100]}")

# TEST 8: ML Insights Agent Initialization
print("\n[TEST 8] ML Insights Agent - Initialization")
total += 1
ml_agent = MLInsightsAgent()
if ml_agent.initialize():
    print("PASS: ML Insights Agent initialized")
    print(f"  - Random state: {ml_agent.random_state}")
    passed += 1
else:
    print("FAIL: ML Insights Agent initialization failed")

# TEST 9: ML Insights Agent - Clustering
print("\n[TEST 9] ML Insights Agent - Clustering Analysis")
total += 1
try:
    numeric_data = test_data[['revenue', 'cost', 'quantity', 'age']]
    result = ml_agent.execute(
        query="Perform customer clustering",
        data=numeric_data
    )
    if result.get('success'):
        print("PASS: Clustering analysis completed")
        passed += 1
    else:
        print(f"FAIL: {result.get('error', 'Unknown error')}")
except Exception as e:
    print(f"FAIL: Exception - {str(e)[:100]}")

# TEST 10: SQL Agent Initialization
print("\n[TEST 10] SQL Agent - Initialization")
total += 1
sql_agent = SQLAgent()
if sql_agent.initialize():
    print("PASS: SQL Agent initialized")
    passed += 1
else:
    print("FAIL: SQL Agent initialization failed")

# TEST 11: Agent Can-Handle Scoring
print("\n[TEST 11] Agent Query Matching (can_handle scores)")
total += 1
queries_agents = [
    ("Calculate mean and standard deviation", stat_agent),
    ("Show me profit margins", fin_agent),
    ("Forecast next month's sales", ts_agent),
    ("Group customers into segments", ml_agent),
]

scores = []
for query, agent in queries_agents:
    score = agent.can_handle(query)
    scores.append(score)
    
if all(score > 0 for score in scores):
    print("PASS: All agents provide relevance scores")
    print(f"  - Avg score: {sum(scores)/len(scores):.2f}")
    passed += 1
else:
    print("FAIL: Some agents not providing scores")

# TEST 12: Agent Metadata
print("\n[TEST 12] Agent Metadata Availability")
total += 1
metadata_count = 0
for agent in [stat_agent, fin_agent, ts_agent, ml_agent, sql_agent]:
    try:
        metadata = agent.get_metadata()
        if metadata and metadata.name:
            metadata_count += 1
    except:
        pass

if metadata_count == 5:
    print(f"PASS: All {metadata_count}/5 agents have metadata")
    passed += 1
else:
    print(f"FAIL: Only {metadata_count}/5 agents have metadata")

# Summary
print("\n" + "=" * 80)
print("PHASE 3 RESULTS: Plugin Agent System")
print("=" * 80)
print(f"Total Tests: {total}")
print(f"Passed: {passed}")
print(f"Failed: {total - passed}")
print(f"Success Rate: {passed/total*100:.1f}%")

if passed == total:
    print("\n✅ ALL PLUGIN AGENT TESTS PASSED")
elif passed >= total * 0.75:
    print(f"\n⚠️ MOSTLY WORKING - {total - passed} tests need attention")
else:
    print(f"\n❌ SIGNIFICANT ISSUES - {total - passed} tests failed")

# Agent Summary
print("\n" + "=" * 80)
print("PLUGIN AGENT SUMMARY")
print("=" * 80)
agents_status = [
    ("Statistical Agent", stat_agent.initialized),
    ("Financial Agent", fin_agent.initialized),
    ("Time Series Agent", ts_agent.initialized),
    ("ML Insights Agent", ml_agent.initialized),
    ("SQL Agent", sql_agent.initialized)
]

for name, status in agents_status:
    print(f"{name:25} {'✓ Operational' if status else '✗ Not initialized'}")

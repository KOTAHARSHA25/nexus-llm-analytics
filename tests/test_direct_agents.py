"""
Direct Agent Test - Tests agents directly without registry
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backend.plugins.statistical_agent import StatisticalAgent
from src.backend.plugins.financial_agent import FinancialAgent
from src.backend.plugins.ml_insights_agent import MLInsightsAgent
from src.backend.plugins.time_series_agent import TimeSeriesAgent
from src.backend.plugins.data_analyst_agent import DataAnalystAgent

print("="*80)
print("DIRECT AGENT TESTING - Domain-Agnostic Validation")
print("="*80)

# Test 1: Initialize agents
print("\n📋 TEST 1: Agent Initialization")
print("-" * 80)

agents = {
    "StatisticalAgent": StatisticalAgent(),
    "FinancialAgent": FinancialAgent(),
    "MLInsightsAgent": MLInsightsAgent(),
    "TimeSeriesAgent": TimeSeriesAgent(),
    "DataAnalystAgent": DataAnalystAgent()
}

for name, agent in agents.items():
    try:
        success = agent.initialize()
        print(f"✅ {name}: {'Initialized' if success else 'Failed to initialize'}")
    except Exception as e:
        print(f"❌ {name}: Exception - {str(e)}")

# Test 2: Routing consistency across domains
print("\n\n📋 TEST 2: Domain-Agnostic Routing")
print("-" * 80)

test_queries = [
    # CORRELATION queries (should ALL go to StatisticalAgent)
    ("Show correlation between sales and revenue", "finance"),
    ("Show correlation between stress and sleep", "health"),
    ("Show correlation between attendance and grades", "education"),
    
    # PREDICTION queries (should ALL go to TimeSeriesAgent)
    ("Predict next quarter revenue", "finance"),
    ("Predict future stress trends", "health"),
    ("Predict final exam scores", "education"),
    
    # CLUSTERING queries (should ALL go to MLInsightsAgent)
    ("Group customers by purchasing behavior", "finance"),
    ("Group patients by symptoms", "health"),
    ("Group students by performance", "education"),
]

routing_results = {
    "correlation": [],
    "prediction": [],
    "clustering": []
}

print("\n  Testing Routing Consistency:")
for query, domain in test_queries:
    # Get confidence scores from all agents
    scores = {}
    for agent_name, agent in agents.items():
        try:
            confidence = agent.can_handle(query, file_type=".csv")
            scores[agent_name] = confidence
        except:
            scores[agent_name] = 0.0
    
    # Find best agent
    best_agent = max(scores, key=scores.get)
    best_confidence = scores[best_agent]
    
    # Categorize
    if "correlation" in query.lower():
        routing_results["correlation"].append((domain, best_agent, best_confidence))
    elif "predict" in query.lower():
        routing_results["prediction"].append((domain, best_agent, best_confidence))
    elif "group" in query.lower():
        routing_results["clustering"].append((domain, best_agent, best_confidence))
    
    print(f"    {domain:10} | {query[:45]:45} → {best_agent:20} ({best_confidence:.2f})")

# Analyze consistency
print("\n  Consistency Analysis:")
print("  " + "-" * 76)

total_tests = 0
passed_tests = 0

for operation, results in routing_results.items():
    agents_used = set([r[1] for r in results])
    total_tests += 1
    
    if len(agents_used) == 1:
        agent_name = list(agents_used)[0]
        print(f"  ✅ {operation.upper():12} - Consistent routing to {agent_name}")
        passed_tests += 1
    else:
        print(f"  ❌ {operation.upper():12} - Inconsistent: {agents_used}")

# Test 3: Specialized agent behavior
print("\n\n📋 TEST 3: Specialized Agent Behavior")
print("-" * 80)

financial_specific = [
    "Analyze financial portfolio with stock and bond allocation",
    "Calculate return on equity and debt-to-equity ratios",
    "Assess investment portfolio risk metrics"
]

generic_operations = [
    "Calculate profit margin from healthcare data",
    "Calculate survival rate from medical data",
    "Calculate pass rate from student data"
]

print("\n  Financial-Specific Queries (SHOULD go to FinancialAgent):")
for query in financial_specific:
    scores = {name: agent.can_handle(query, ".csv") for name, agent in agents.items()}
    best_agent = max(scores, key=scores.get)
    
    if best_agent == "FinancialAgent":
        print(f"    ✅ {query[:60]}")
        print(f"       → {best_agent} ({scores[best_agent]:.2f})")
        passed_tests += 1
    else:
        print(f"    ❌ {query[:60]}")
        print(f"       → {best_agent} ({scores[best_agent]:.2f}) - Expected FinancialAgent")
    total_tests += 1

print("\n  Generic Operations (should NOT go to FinancialAgent):")
for query in generic_operations:
    scores = {name: agent.can_handle(query, ".csv") for name, agent in agents.items()}
    best_agent = max(scores, key=scores.get)
    
    if best_agent != "FinancialAgent":
        print(f"    ✅ {query[:60]}")
        print(f"       → {best_agent} ({scores[best_agent]:.2f})")
        passed_tests += 1
    else:
        print(f"    ❌ {query[:60]}")
        print(f"       → {best_agent} ({scores[best_agent]:.2f}) - Should NOT be FinancialAgent")
    total_tests += 1

# Test 4: Execute with real data
print("\n\n📋 TEST 4: Execution with Real Data")
print("-" * 80)

execution_tests = [
    ("Calculate average revenue by region", "test_sales_monthly.csv", "DataAnalystAgent"),
]

for query, filename, expected_agent in execution_tests:
    print(f"\n  Query: {query}")
    print(f"  File: {filename}")
    
    # Route
    scores = {name: agent.can_handle(query, ".csv") for name, agent in agents.items()}
    best_agent_name = max(scores, key=scores.get)
    best_agent = agents[best_agent_name]
    
    print(f"  Routed to: {best_agent_name} ({scores[best_agent_name]:.2f})")
    
    # Execute
    try:
        result = best_agent.execute(query, filename=filename)
        
        if result.get("success"):
            print(f"  ✅ Execution successful")
            passed_tests += 1
        else:
            print(f"  ❌ Execution failed: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"  ❌ Exception: {str(e)}")
    
    total_tests += 1

# Final Summary
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"\nTotal Tests: {total_tests}")
print(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
print(f"Failed: {total_tests - passed_tests} ({(total_tests-passed_tests)/total_tests*100:.1f}%)")

if passed_tests == total_tests:
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ System is domain-agnostic")
    print("✅ Agent routing is accurate")
    sys.exit(0)
else:
    print(f"\n⚠️  {total_tests - passed_tests} TEST(S) FAILED")
    sys.exit(1)

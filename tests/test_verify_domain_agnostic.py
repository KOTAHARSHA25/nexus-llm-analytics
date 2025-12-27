"""
RIGOROUS Domain-Agnostic Routing Test
======================================
Tests actual routing behavior, not just enum names.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Disable async cache manager
os.environ['DISABLE_CACHE_BACKGROUND'] = '1'

from backend.core.plugin_system import get_agent_registry
from backend.core.intelligent_query_engine import QueryType, AgentCapability
import logging

logging.basicConfig(level=logging.WARNING)

print("="*80)
print("RIGOROUS DOMAIN-AGNOSTIC ROUTING TEST")
print("="*80)

# Test queries that should route to SAME agent across different domains
test_cases = [
    {
        'operation': 'Ratio Calculation',
        'queries': [
            ('Calculate profit margin from revenue and cost', 'finance'),
            ('Calculate survival rate from patients data', 'medical'),
            ('Calculate pass percentage from test scores', 'education'),
            ('Calculate conversion rate from clicks and purchases', 'marketing'),
        ],
        'expected_agent_types': ['FinancialAgent', 'DataAnalyst']  # Should route to same one
    },
    {
        'operation': 'Correlation Analysis', 
        'queries': [
            ('Find correlation between sales and advertising', 'business'),
            ('Find correlation between drug dosage and recovery', 'medical'),
            ('Find correlation between study hours and grades', 'education'),
        ],
        'expected_agent_types': ['StatisticalAgent', 'DataAnalyst']
    },
    {
        'operation': 'Time Series Forecasting',
        'queries': [
            ('Predict next quarter revenue', 'finance'),
            ('Predict patient admission trends', 'healthcare'),
            ('Forecast student enrollment', 'education'),
        ],
        'expected_agent_types': ['TimeSeriesAgent', 'DataAnalyst']
    },
    {
        'operation': 'Clustering/Grouping',
        'queries': [
            ('Group customers by purchasing behavior', 'business'),
            ('Group patients by symptom similarity', 'medical'),
            ('Group students by learning patterns', 'education'),
        ],
        'expected_agent_types': ['MLInsightsAgent', 'DataAnalyst']
    }
]

print("\n" + "="*80)
print("PART 1: ENUM STRUCTURE VALIDATION")
print("="*80)

# Quick enum check
query_types = [qt.value for qt in QueryType]
capabilities = [cap.value for cap in AgentCapability]

fail_count = 0

if 'financial_analysis' in query_types or 'business_intelligence' in query_types:
    print("[FAIL] Domain-specific QueryTypes still exist")
    fail_count += 1
else:
    print("[PASS] QueryType enum is domain-agnostic")

if 'financial_modeling' in capabilities or 'business_metrics' in capabilities:
    print("[FAIL] Domain-specific AgentCapabilities still exist")
    fail_count += 1
else:
    print("[PASS] AgentCapability enum is domain-agnostic")

if 'ratio_calculation' not in capabilities or 'metrics_computation' not in capabilities:
    print("[FAIL] Generic capabilities missing")
    fail_count += 1
else:
    print("[PASS] Generic capabilities present")

print("\n" + "="*80)
print("PART 2: ACTUAL ROUTING BEHAVIOR TEST")
print("="*80)

registry = get_agent_registry()

total_operations = 0
passed_operations = 0
failed_operations = 0

for test_case in test_cases:
    operation = test_case['operation']
    queries = test_case['queries']
    
    print(f"\nTesting: {operation}")
    print("-" * 80)
    
    routed_agents = []
    
    for query, domain in queries:
        total_operations += 1
        
        try:
            # Get routing decision - returns (topic, confidence, agent)
            topic, confidence, agent = registry.route_query(query, file_type=None)
            
            if agent is None:
                print(f"  [FAIL] {domain:12} | No agent selected for: {query[:40]}")
                failed_operations += 1
                routed_agents.append(None)
                continue
            
            agent_name = agent.__class__.__name__
            
            print(f"  [INFO] {domain:12} | Agent: {agent_name:20} | Confidence: {confidence:.2f} | Query: {query[:35]}")
            routed_agents.append(agent_name)
            
        except Exception as e:
            print(f"  [FAIL] {domain:12} | Error: {str(e)[:50]}")
            failed_operations += 1
            routed_agents.append(None)
    
    # Check consistency
    unique_agents = set([a for a in routed_agents if a is not None])
    
    if len(unique_agents) == 0:
        print(f"\n  [FAIL] {operation}: No agents selected")
        failed_operations += len(queries)
    elif len(unique_agents) == 1:
        print(f"\n  [PASS] {operation}: All queries routed to {unique_agents.pop()}")
        passed_operations += len(queries)
    else:
        # Check if all are in expected types
        if all(agent in test_case['expected_agent_types'] for agent in unique_agents):
            print(f"\n  [PASS] {operation}: Routed to {unique_agents} (all acceptable)")
            passed_operations += len(queries)
        else:
            print(f"\n  [FAIL] {operation}: Inconsistent routing - {unique_agents}")
            failed_operations += len(queries)

print("\n" + "="*80)
print("TEST RESULTS")
print("="*80)

pass_rate = (passed_operations / total_operations * 100) if total_operations > 0 else 0

print(f"\nTotal Operations Tested: {total_operations}")
print(f"Passed: {passed_operations} ({pass_rate:.1f}%)")
print(f"Failed: {failed_operations} ({100-pass_rate:.1f}%)")
print(f"Enum Structure Failures: {fail_count}")

if fail_count > 0 or failed_operations > 0:
    print("\n[OVERALL: FAIL] System is NOT domain-agnostic")
    print("\nFailures indicate:")
    if fail_count > 0:
        print("  - Core enum structure still has domain-specific types")
    if failed_operations > 0:
        print("  - Routing behavior is inconsistent across domains")
    sys.exit(1)
else:
    print("\n[OVERALL: PASS] System is domain-agnostic")
    print("\nAll identical operations route consistently regardless of domain vocabulary.")
    sys.exit(0)

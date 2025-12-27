"""
Comprehensive Integration Test
Tests the entire system end-to-end with real datasets to validate:
1. Domain-agnostic behavior across multiple domains
2. Specialized agent routing accuracy
3. Data loading from any location
4. Query execution and result quality
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backend.core.plugin_system import get_agent_registry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

class TestResults:
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.details = []
    
    def add_pass(self, test_name, details=""):
        self.total += 1
        self.passed += 1
        self.details.append(f"✅ PASS: {test_name}")
        if details:
            self.details.append(f"   {details}")
    
    def add_fail(self, test_name, reason):
        self.total += 1
        self.failed += 1
        self.details.append(f"❌ FAIL: {test_name}")
        self.details.append(f"   Reason: {reason}")
    
    def print_summary(self):
        print("\n" + "="*80)
        print("COMPREHENSIVE INTEGRATION TEST RESULTS")
        print("="*80)
        for detail in self.details:
            print(detail)
        print("\n" + "="*80)
        print(f"Total Tests: {self.total}")
        print(f"Passed: {self.passed} ({self.passed/self.total*100:.1f}%)")
        print(f"Failed: {self.failed} ({self.failed/self.total*100:.1f}%)")
        print("="*80)
        return self.failed == 0


def test_domain_agnostic_routing():
    """Test routing consistency across different domains"""
    print("\n📋 TEST 1: DOMAIN-AGNOSTIC ROUTING")
    print("-" * 80)
    
    results = TestResults()
    registry = get_agent_registry()
    
    # Initialize agents
    print("  Initializing agents...")
    count = registry.discover_agents()
    print(f"  Loaded {count} agents\n")
    
    # Define test cases across multiple domains
    test_cases = [
        # BUSINESS/FINANCE DOMAIN
        {
            "domain": "Business/Finance",
            "file": "test_sales_monthly.csv",
            "queries": [
                ("Show correlation between units_sold and revenue", "StatisticalAgent"),
                ("Calculate average profit margin by region", "DataAnalyst"),
                ("Predict next month revenue", "TimeSeriesAgent"),
                ("Group regions by sales performance", "MLInsightsAgent")
            ]
        },
        # HEALTH/PSYCHOLOGY DOMAIN
        {
            "domain": "Health/Psychology",
            "file": "StressLevelDataset.csv",
            "queries": [
                ("Show correlation between stress and sleep", "StatisticalAgent"),
                ("Calculate average stress level by category", "DataAnalyst"),
                ("Predict future stress trends", "TimeSeriesAgent"),
                ("Group individuals by stress patterns", "MLInsightsAgent")
            ]
        },
        # EDUCATION DOMAIN
        {
            "domain": "Education",
            "file": "test_student_grades.csv",
            "queries": [
                ("Show correlation between attendance and grades", "StatisticalAgent"),
                ("Calculate average grade by subject", "DataAnalyst"),
                ("Predict final exam scores", "TimeSeriesAgent"),
                ("Group students by performance level", "MLInsightsAgent")
            ]
        },
        # HR/EMPLOYEE DOMAIN
        {
            "domain": "HR/Employee",
            "file": "test_employee_data.csv",
            "queries": [
                ("Show correlation between experience and salary", "StatisticalAgent"),
                ("Calculate average salary by department", "DataAnalyst"),
                ("Predict employee retention", "TimeSeriesAgent"),
                ("Group employees by skill similarity", "MLInsightsAgent")
            ]
        }
    ]
    
    # Track routing consistency per operation type
    operation_routing = {
        "correlation": [],
        "average": [],
        "prediction": [],
        "grouping": []
    }
    
    for test_case in test_cases:
        domain = test_case["domain"]
        filename = test_case["file"]
        
        print(f"\n  Testing Domain: {domain} (File: {filename})")
        
        for query, expected_agent in test_case["queries"]:
            try:
                topic, confidence, agent = registry.route_query(query, file_type=".csv")
                agent_name = agent.metadata.name if agent else "None"
                
                # Categorize by operation type
                if "correlation" in query.lower():
                    operation_routing["correlation"].append((domain, agent_name))
                elif "average" in query.lower() or "calculate" in query.lower():
                    operation_routing["average"].append((domain, agent_name))
                elif "predict" in query.lower() or "forecast" in query.lower():
                    operation_routing["prediction"].append((domain, agent_name))
                elif "group" in query.lower() or "cluster" in query.lower():
                    operation_routing["grouping"].append((domain, agent_name))
                
                if agent_name == expected_agent:
                    results.add_pass(
                        f"{domain}: {query[:50]}...",
                        f"Routed to {agent_name} (confidence: {confidence:.2f})"
                    )
                else:
                    results.add_fail(
                        f"{domain}: {query[:50]}...",
                        f"Expected {expected_agent}, got {agent_name} (confidence: {confidence:.2f})"
                    )
                    
            except Exception as e:
                results.add_fail(
                    f"{domain}: {query[:50]}...",
                    f"Exception: {str(e)}"
                )
    
    # Check routing consistency per operation
    print("\n  Routing Consistency Analysis:")
    for operation, routings in operation_routing.items():
        agents = [r[1] for r in routings]
        unique_agents = set(agents)
        
        if len(unique_agents) == 1:
            print(f"    ✅ {operation.upper()}: Consistent → {agents[0]}")
            results.add_pass(
                f"Routing consistency for {operation}",
                f"All queries routed to {agents[0]}"
            )
        else:
            print(f"    ❌ {operation.upper()}: Inconsistent → {unique_agents}")
            results.add_fail(
                f"Routing consistency for {operation}",
                f"Multiple agents used: {unique_agents}"
            )
    
    return results


def test_agent_execution():
    """Test that agents can actually execute queries with real data"""
    print("\n📋 TEST 2: AGENT EXECUTION WITH REAL DATA")
    print("-" * 80)
    
    results = TestResults()
    registry = get_agent_registry()
    
    # Use absolute paths to test data loading from any location
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "samples"
    
    test_executions = [
        {
            "query": "Calculate average revenue by region",
            "filename": "test_sales_monthly.csv",
            "expected_agent": "DataAnalyst"
        },
        {
            "query": "Show correlation between units_sold and revenue",
            "filename": "test_sales_monthly.csv",
            "expected_agent": "StatisticalAgent"
        },
        {
            "query": "Calculate average stress level",
            "filename": "StressLevelDataset.csv",
            "expected_agent": "DataAnalyst"
        }
    ]
    
    for test in test_executions:
        query = test["query"]
        filename = test["filename"]
        expected_agent = test["expected_agent"]
        
        print(f"\n  Testing: {query}")
        print(f"  File: {filename}")
        
        try:
            # Route query
            topic, confidence, agent = registry.route_query(query, file_type=".csv")
            agent_name = agent.metadata.name if agent else "None"
            
            print(f"  Routed to: {agent_name} (confidence: {confidence:.2f})")
            
            if agent_name != expected_agent:
                results.add_fail(
                    f"Routing for: {query}",
                    f"Expected {expected_agent}, got {agent_name}"
                )
                continue
            
            # Execute query
            result = agent.execute(query, filename=filename)
            
            if result.get("success"):
                results.add_pass(
                    f"Execution: {query}",
                    f"Agent: {agent_name}, Success: {result.get('success')}"
                )
                
                # Check if result has meaningful output
                if result.get("result"):
                    print(f"  ✅ Result preview: {str(result.get('result'))[:100]}...")
                else:
                    print(f"  ⚠️  Warning: No result content returned")
            else:
                error = result.get("error", "Unknown error")
                results.add_fail(
                    f"Execution: {query}",
                    f"Agent: {agent_name}, Error: {error}"
                )
                print(f"  ❌ Error: {error}")
                
        except Exception as e:
            results.add_fail(
                f"Execution: {query}",
                f"Exception: {str(e)}"
            )
            print(f"  ❌ Exception: {str(e)}")
    
    return results


def test_specialized_agent_capabilities():
    """Test that specialized agents handle their domain correctly"""
    print("\n📋 TEST 3: SPECIALIZED AGENT CAPABILITIES")
    print("-" * 80)
    
    results = TestResults()
    registry = get_agent_registry()
    
    # Test specialized scenarios
    specialized_tests = [
        {
            "category": "Statistical Analysis",
            "queries": [
                "Show correlation between variables",
                "Perform t-test analysis",
                "Calculate standard deviation"
            ],
            "expected_agent": "StatisticalAgent"
        },
        {
            "category": "Time Series",
            "queries": [
                "Forecast next quarter sales",
                "Predict future trends",
                "Time series analysis"
            ],
            "expected_agent": "TimeSeriesAgent"
        },
        {
            "category": "Machine Learning",
            "queries": [
                "Cluster customers by behavior",
                "Classify data into groups",
                "Find patterns in data"
            ],
            "expected_agent": "MLInsightsAgent"
        },
        {
            "category": "Financial Domain (Strict)",
            "queries": [
                "Analyze financial portfolio performance",
                "Calculate investment returns and risk ratios",
                "Balance sheet analysis with liquidity ratios"
            ],
            "expected_agent": "FinancialAgent"
        },
        {
            "category": "Generic Operations (Should NOT go to FinancialAgent)",
            "queries": [
                "Calculate profit margin from healthcare data",
                "Calculate survival rate from medical data",
                "Calculate pass rate from education data"
            ],
            "expected_agent": "StatisticalAgent"  # NOT FinancialAgent
        }
    ]
    
    for test in specialized_tests:
        category = test["category"]
        expected_agent = test["expected_agent"]
        
        print(f"\n  Testing {category}:")
        
        for query in test["queries"]:
            try:
                topic, confidence, agent = registry.route_query(query, file_type=".csv")
                agent_name = agent.metadata.name if agent else "None"
                
                if agent_name == expected_agent:
                    results.add_pass(
                        f"{category}: {query[:50]}...",
                        f"Correctly routed to {agent_name} (conf: {confidence:.2f})"
                    )
                    print(f"    ✅ {query[:60]}... → {agent_name} ({confidence:.2f})")
                else:
                    results.add_fail(
                        f"{category}: {query[:50]}...",
                        f"Expected {expected_agent}, got {agent_name} (conf: {confidence:.2f})"
                    )
                    print(f"    ❌ {query[:60]}... → {agent_name} (expected {expected_agent})")
                    
            except Exception as e:
                results.add_fail(
                    f"{category}: {query[:50]}...",
                    f"Exception: {str(e)}"
                )
                print(f"    ❌ Exception: {str(e)}")
    
    return results


def main():
    """Run all comprehensive tests"""
    print("\n" + "="*80)
    print("COMPREHENSIVE INTEGRATION TEST SUITE")
    print("Testing domain-agnostic behavior and agent routing accuracy")
    print("="*80)
    
    all_results = []
    
    # Test 1: Domain-agnostic routing
    all_results.append(test_domain_agnostic_routing())
    
    # Test 2: Agent execution
    all_results.append(test_agent_execution())
    
    # Test 3: Specialized capabilities
    all_results.append(test_specialized_agent_capabilities())
    
    # Combined summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    total_tests = sum(r.total for r in all_results)
    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    
    print(f"\nTotal Tests Run: {total_tests}")
    print(f"Total Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)")
    print(f"Total Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)")
    
    if total_failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ System is domain-agnostic")
        print("✅ Agent routing is accurate")
        print("✅ All specialized agents working correctly")
        return 0
    else:
        print(f"\n⚠️  {total_failed} TEST(S) FAILED")
        print("Review the details above for specific failures")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

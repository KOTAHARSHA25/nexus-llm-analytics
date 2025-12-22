"""
Agent Routing Verification Test
Tests that queries are correctly routed to specialized agents based on content.
"""
import sys
import os
from pathlib import Path
import json

# Add src to path and set working directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
os.chdir(project_root)

from backend.core.plugin_system import get_agent_registry


def test_agent_routing():
    """Test that different query types route to the correct specialized agents"""
    
    print("\n" + "=" * 80)
    print("🔀 AGENT ROUTING VERIFICATION TEST")
    print("=" * 80)
    print("Testing if queries are routed to the correct specialized agents\n")
    
    # Get the agent registry and discover agents
    registry = get_agent_registry()
    registry.discover_agents()
    
    # Define test cases: (query, expected_agent_name, description)
    test_cases = [
        # Statistical Agent Tests
        (
            "Perform a t-test to compare sales between two regions",
            "StatisticalAgent",
            "Statistical hypothesis testing query"
        ),
        (
            "Calculate the correlation between marketing spend and revenue",
            "StatisticalAgent", 
            "Correlation analysis query"
        ),
        (
            "Run ANOVA to test if there's a significant difference in performance across groups",
            "StatisticalAgent",
            "ANOVA statistical test query"
        ),
        
        # Time Series Agent Tests
        (
            "Forecast next quarter's sales using ARIMA model",
            "TimeSeriesAgent",
            "ARIMA forecasting query"
        ),
        (
            "Analyze the trend and seasonality in monthly revenue data",
            "TimeSeriesAgent",
            "Seasonality decomposition query"
        ),
        (
            "Predict future demand based on historical patterns",
            "TimeSeriesAgent",
            "Time series prediction query"
        ),
        
        # Financial Agent Tests
        (
            "Calculate the ROI and profitability ratios for this quarter",
            "FinancialAgent",
            "Financial metrics query"
        ),
        (
            "Analyze our company's financial health and cash flow",
            "FinancialAgent",
            "Financial health assessment query"
        ),
        (
            "Compute break-even point and profit margins",
            "FinancialAgent",
            "Break-even analysis query"
        ),
        
        # ML Insights Agent Tests
        (
            "Find customer segments using clustering analysis",
            "MLInsightsAgent",
            "Clustering query"
        ),
        (
            "Detect anomalies in transaction data",
            "MLInsightsAgent",
            "Anomaly detection query"
        ),
        (
            "Perform PCA to reduce dimensionality and identify key features",
            "MLInsightsAgent",
            "PCA dimensionality reduction query"
        ),
        
        # SQL Agent Tests
        (
            "Query the database to find top 10 customers by revenue",
            "SQLAgent",
            "SQL query generation"
        ),
        (
            "Generate SQL to join orders and customers tables",
            "SQLAgent",
            "SQL join query"
        ),
        (
            "Analyze the database schema and optimize queries",
            "SQLAgent",
            "Database schema analysis"
        ),
        
        # General Data Analysis (should route to DataAnalyst)
        (
            "What is the average sales in the dataset?",
            "DataAnalystAgent",
            "Simple data analysis query"
        ),
        (
            "Show me the top 5 products by quantity sold",
            "DataAnalystAgent",
            "Basic aggregation query"
        ),
    ]
    
    results = {
        "correct": 0,
        "incorrect": 0,
        "errors": 0,
        "details": []
    }
    
    print(f"Running {len(test_cases)} routing tests...\n")
    
    for i, (query, expected_agent, description) in enumerate(test_cases, 1):
        print(f"Test {i}/{len(test_cases)}: {description}")
        print(f"Query: \"{query}\"")
        
        try:
            # Get routing decision
            # Use simpler context - just pass query
            topic, confidence, selected_agent = registry.route_query(query, file_type="csv")
            
            if selected_agent:
                agent_name = selected_agent.__class__.__name__
                
                if agent_name == expected_agent:
                    print(f"✅ CORRECT: Routed to {agent_name} (confidence: {confidence:.2f})")
                    results["correct"] += 1
                    results["details"].append({
                        "test": i,
                        "query": query,
                        "expected": expected_agent,
                        "actual": agent_name,
                        "status": "PASS"
                    })
                else:
                    print(f"❌ INCORRECT: Expected {expected_agent}, got {agent_name}")
                    results["incorrect"] += 1
                    results["details"].append({
                        "test": i,
                        "query": query,
                        "expected": expected_agent,
                        "actual": agent_name,
                        "status": "FAIL"
                    })
            else:
                print(f"⚠️  ERROR: No agent selected")
                results["errors"] += 1
                results["details"].append({
                    "test": i,
                    "query": query,
                    "expected": expected_agent,
                    "actual": "None",
                    "status": "ERROR"
                })
                
        except Exception as e:
            print(f"⚠️  ERROR: {str(e)}")
            results["errors"] += 1
            results["details"].append({
                "test": i,
                "query": query,
                "expected": expected_agent,
                "actual": f"Exception: {str(e)}",
                "status": "ERROR"
            })
        
        print()
    
    # Print summary
    print("=" * 80)
    print("📊 ROUTING TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {len(test_cases)}")
    print(f"✅ Correct Routing: {results['correct']} ({results['correct']/len(test_cases)*100:.1f}%)")
    print(f"❌ Incorrect Routing: {results['incorrect']} ({results['incorrect']/len(test_cases)*100:.1f}%)")
    print(f"⚠️  Errors: {results['errors']} ({results['errors']/len(test_cases)*100:.1f}%)")
    print("=" * 80)
    
    # Show incorrect routings in detail
    if results["incorrect"] > 0:
        print("\n❌ FAILED ROUTING DETAILS:")
        print("-" * 80)
        for detail in results["details"]:
            if detail["status"] == "FAIL":
                print(f"Test {detail['test']}: {detail['query'][:60]}...")
                print(f"  Expected: {detail['expected']}")
                print(f"  Got: {detail['actual']}")
                print()
    
    # Save detailed results
    results_file = Path(__file__).parent / "agent_routing_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Test passes if >80% correct routing
    success_rate = results['correct'] / len(test_cases)
    if success_rate >= 0.8:
        print(f"\n🎉 SUCCESS! Routing accuracy: {success_rate*100:.1f}%")
        return True
    else:
        print(f"\n⚠️  NEEDS IMPROVEMENT! Routing accuracy: {success_rate*100:.1f}%")
        print("   Target: 80% or higher")
        return False


def test_agent_capabilities():
    """Verify all expected agents are registered"""
    
    print("\n" + "=" * 80)
    print("🤖 AGENT REGISTRATION VERIFICATION")
    print("=" * 80)
    
    registry = get_agent_registry()
    
    # Discover agents
    print("\n🔍 Discovering agents...")
    discovered_count = registry.discover_agents()
    print(f"Found {discovered_count} plugin agents\n")
    
    expected_agents = [
        "DataAnalystAgent",
        "StatisticalAgent", 
        "TimeSeriesAgent",
        "FinancialAgent",
        "MLInsightsAgent",
        "SQLAgent",
        "RagAgent",
        "VisualizerAgent",
        "ReporterAgent",
        "ReviewerAgent"
    ]
    
    print(f"\nChecking for {len(expected_agents)} expected agents...\n")
    
    # Get all registered agents (use registered_agents dict)
    registered = {}
    for agent_name, instance in registry.registered_agents.items():
        # Agent name is already a string in registered_agents
        registered[agent_name] = instance
        
    found = []
    missing = []
    
    for expected in expected_agents:
        if expected in registered:
            agent = registered[expected]
            metadata = agent.get_metadata()
            print(f"✅ {expected}")
            print(f"   Priority: {metadata.priority}")
            print(f"   Capabilities: {', '.join([c.value for c in metadata.capabilities])}")
            found.append(expected)
        else:
            print(f"❌ {expected} - NOT FOUND")
            missing.append(expected)
    
    print("\n" + "=" * 80)
    print(f"📊 REGISTRATION SUMMARY")
    print("=" * 80)
    print(f"Expected: {len(expected_agents)}")
    print(f"Found: {len(found)}")
    print(f"Missing: {len(missing)}")
    
    if missing:
        print(f"\n⚠️  Missing agents: {', '.join(missing)}")
        return False
    else:
        print(f"\n🎉 All agents successfully registered!")
        return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 NEXUS LLM ANALYTICS - AGENT ROUTING TEST SUITE")
    print("=" * 80)
    
    # Test 1: Agent Registration
    registration_ok = test_agent_capabilities()
    
    # Test 2: Query Routing
    routing_ok = test_agent_routing()
    
    # Final verdict
    print("\n" + "=" * 80)
    print("🏁 FINAL VERDICT")
    print("=" * 80)
    
    if registration_ok and routing_ok:
        print("✅ ALL TESTS PASSED - Routing system working correctly!")
        sys.exit(0)
    else:
        print("⚠️  SOME TESTS FAILED - Review results above")
        sys.exit(1)

"""
Comprehensive Agent Routing Test with Real Data
Tests routing with actual uploaded files from the data folder.
"""
import sys
import os
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root / "src"))

from backend.core.plugin_system import get_agent_registry

def test_with_real_data():
    """Test routing with actual data files"""
    
    print("\n" + "=" * 80)
    print("🗂️  AGENT ROUTING TEST WITH REAL DATA FILES")
    print("=" * 80)
    
    # Get registry
    registry = get_agent_registry()
    registry.discover_agents()
    
    print(f"\n✅ Discovered {len(registry.registered_agents)} agents\n")
    
    # Test cases with real data files
    test_cases = [
        {
            "file": "data/samples/StressLevelDataset.csv",
            "query": "Perform statistical analysis and hypothesis testing on stress levels",
            "expected": "StatisticalAgent",
            "description": "Statistical analysis on stress level data"
        },
        {
            "file": "data/samples/sales_timeseries.json",
            "query": "Forecast future sales trends using time series analysis",
            "expected": "TimeSeriesAgent",
            "description": "Time series forecasting"
        },
        {
            "file": "data/samples/financial_quarterly.json",
            "query": "Calculate financial ratios and profitability metrics",
            "expected": "FinancialAgent",
            "description": "Financial analysis"
        },
        {
            "file": "data/samples/large_transactions.json",
            "query": "Detect anomalies and find patterns using machine learning",
            "expected": "MLInsightsAgent",
            "description": "ML-based anomaly detection"
        },
        {
            "file": "data/samples/sales_data.csv",
            "query": "What is the total revenue by product category?",
            "expected": "DataAnalyst",
            "description": "Simple data aggregation"
        },
        {
            "file": "data/samples/1.json",
            "query": "What is the name in this dataset?",
            "expected": "DataAnalyst",
            "description": "Simple JSON lookup (hallucination test case!)"
        },
    ]
    
    results = {"correct": 0, "total": len(test_cases)}
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}/{len(test_cases)}: {test['description']}")
        print(f"File: {test['file']}")
        print(f"Query: {test['query'][:70]}...")
        
        # Check if file exists
        file_path = Path(test['file'])
        if not file_path.exists():
            print(f"⚠️  File not found: {test['file']}\n")
            continue
        
        # Get file type
        file_type = file_path.suffix[1:]  # Remove the dot
        
        try:
            # Route query
            topic, confidence, agent = registry.route_query(
                test['query'],
                file_type=file_type
            )
            
            actual_agent = agent.__class__.__name__ if agent else "None"
            
            # Check if routing is correct
            if actual_agent == test['expected']:
                print(f"✅ CORRECT: Routed to {actual_agent} (confidence: {confidence:.2f})")
                results["correct"] += 1
            else:
                print(f"❌ INCORRECT: Expected {test['expected']}, got {actual_agent} (confidence: {confidence:.2f})")
            
        except Exception as e:
            print(f"⚠️  ERROR: {str(e)}")
        
        print()
    
    # Summary
    print("=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    accuracy = (results["correct"] / results["total"]) * 100
    print(f"Correct: {results['correct']}/{results['total']} ({accuracy:.1f}%)")
    
    if accuracy >= 80:
        print("\n🎉 SUCCESS! Routing system working well!")
    elif accuracy >= 60:
        print("\n✅ GOOD! Most queries routed correctly")
    else:
        print("\n⚠️  NEEDS IMPROVEMENT")
    
    print("=" * 80)
    
    return accuracy >= 60


def demo_all_agents():
    """Demonstrate routing to each specialized agent"""
    
    print("\n" + "=" * 80)
    print("🤖 DEMONSTRATION: ROUTING TO EACH SPECIALIZED AGENT")
    print("=" * 80)
    
    registry = get_agent_registry()
    registry.discover_agents()
    
    # Queries designed to trigger each agent
    demonstrations = [
        ("T-test comparison between two groups", "StatisticalAgent", "📊"),
        ("ARIMA forecast for next quarter", "TimeSeriesAgent", "📈"),
        ("Calculate ROI and break-even point", "FinancialAgent", "💰"),
        ("K-means clustering to segment customers", "MLInsightsAgent", "🤖"),
        ("Query database for top 10 records", "SQLAgent", "🗄️"),
        ("Show average and sum of sales column", "DataAnalyst", "📋"),
    ]
    
    print()
    for query, expected, emoji in demonstrations:
        topic, conf, agent = registry.route_query(query, file_type="csv")
        actual = agent.__class__.__name__ if agent else "None"
        
        status = "✅" if actual == expected else "❌"
        print(f"{status} {emoji} {expected:20} | Query: {query}")
        print(f"   → Routed to: {actual} (confidence: {conf:.2f})\n")
    
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 NEXUS LLM ANALYTICS - COMPREHENSIVE ROUTING VERIFICATION")
    print("=" * 80)
    
    # Run demonstrations
    demo_all_agents()
    
    # Run tests with real data
    success = test_with_real_data()
    
    sys.exit(0 if success else 1)

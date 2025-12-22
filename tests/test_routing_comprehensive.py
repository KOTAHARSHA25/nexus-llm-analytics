"""
COMPREHENSIVE Agent Routing Accuracy Test
Target: 80-90% accuracy across all test cases
"""
import sys
import os
from pathlib import Path

# Setup
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root / "src"))

from backend.core.plugin_system import get_agent_registry

def run_comprehensive_test():
    print("\n" + "=" * 80)
    print("🎯 COMPREHENSIVE AGENT ROUTING ACCURACY TEST")
    print("=" * 80)
    print("Target: 80-90% accuracy\n")
    
    registry = get_agent_registry()
    registry.discover_agents()
    
    # Comprehensive test cases covering all agents
    test_cases = [
        # Statistical Agent (4 tests)
        ("Perform a t-test to compare sales between two regions", "csv", "StatisticalAgent"),
        ("Calculate correlation between marketing spend and revenue", "csv", "StatisticalAgent"),
        ("Run ANOVA to test significance across groups", "csv", "StatisticalAgent"),
        ("Test the hypothesis that means are different using chi-square", "csv", "StatisticalAgent"),
        
        # Time Series Agent (4 tests)
        ("Forecast next quarter's sales using ARIMA model", "csv", "TimeSeriesAgent"),
        ("Analyze trend and seasonality in monthly revenue", "csv", "TimeSeriesAgent"),
        ("Predict future demand based on historical patterns", "csv", "TimeSeriesAgent"),
        ("Perform seasonal decomposition of the time series", "csv", "TimeSeriesAgent"),
        
        # Financial Agent (4 tests)
        ("Calculate ROI and profitability ratios for Q4", "csv", "FinancialAgent"),
        ("Analyze company's financial health and cash flow", "json", "FinancialAgent"),
        ("Compute break-even point and profit margins", "xlsx", "FinancialAgent"),
        ("Evaluate investment performance and returns", "csv", "FinancialAgent"),
        
        # ML Insights Agent (4 tests)
        ("Find customer segments using k-means clustering", "csv", "MLInsightsAgent"),
        ("Detect anomalies in transaction data", "csv", "MLInsightsAgent"),
        ("Perform PCA to reduce dimensionality", "csv", "MLInsightsAgent"),
        ("Identify patterns using machine learning", "json", "MLInsightsAgent"),
        
        # SQL Agent (4 tests)
        ("Generate SQL query to find top 10 customers by revenue", "csv", "SQLAgent"),
        ("Write SQL to join orders and customers tables", "csv", "SQLAgent"),
        ("Create SQL database query for product analysis", "csv", "SQLAgent"),
        ("Query the database using SQL to get monthly totals", "csv", "SQLAgent"),
        
        # Data Analyst Agent (5 tests - general queries)
        ("What is the average sales in the dataset?", "csv", "DataAnalystAgent"),
        ("Show me the top 5 products by quantity sold", "json", "DataAnalystAgent"),
        ("Calculate total revenue by category", "xlsx", "DataAnalystAgent"),
        ("What is the name in this file?", "json", "DataAnalystAgent"),
        ("Display summary statistics for all columns", "csv", "DataAnalystAgent"),
    ]
    
    results = {
        "correct": 0,
        "incorrect": 0,
        "by_agent": {}
    }
    
    print(f"Running {len(test_cases)} routing tests...\n")
    
    for i, (query, file_type, expected) in enumerate(test_cases, 1):
        topic, conf, agent = registry.route_query(query, file_type=file_type)
        actual = agent.__class__.__name__ if agent else "None"
        
        # Track by agent
        if expected not in results["by_agent"]:
            results["by_agent"][expected] = {"correct": 0, "total": 0}
        results["by_agent"][expected]["total"] += 1
        
        if actual == expected:
            status = "✅"
            results["correct"] += 1
            results["by_agent"][expected]["correct"] += 1
        else:
            status = "❌"
            results["incorrect"] += 1
        
        print(f"{status} Test {i:2d}/{len(test_cases)}: {query[:55]}...")
        print(f"         Expected: {expected:20} | Got: {actual:20} | Conf: {conf:.2f}")
    
    # Results summary
    total = len(test_cases)
    accuracy = (results["correct"] / total) * 100
    
    print("\n" + "=" * 80)
    print("📊 OVERALL RESULTS")
    print("=" * 80)
    print(f"Total Tests:  {total}")
    print(f"✅ Correct:   {results['correct']} ({accuracy:.1f}%)")
    print(f"❌ Incorrect: {results['incorrect']} ({100-accuracy:.1f}%)")
    
    # Per-agent accuracy
    print("\n" + "=" * 80)
    print("📈 PER-AGENT ACCURACY")
    print("=" * 80)
    
    for agent_name in sorted(results["by_agent"].keys()):
        stats = results["by_agent"][agent_name]
        agent_accuracy = (stats["correct"] / stats["total"]) * 100
        status = "✅" if agent_accuracy >= 75 else "⚠️" if agent_accuracy >= 50 else "❌"
        print(f"{status} {agent_name:20} {stats['correct']}/{stats['total']} ({agent_accuracy:.0f}%)")
    
    # Final verdict
    print("\n" + "=" * 80)
    if accuracy >= 90:
        print("🎉 EXCELLENT! 90%+ accuracy achieved!")
        verdict = "PASS"
    elif accuracy >= 80:
        print("✅ GOOD! 80%+ accuracy achieved (target met)")
        verdict = "PASS"
    elif accuracy >= 70:
        print("⚠️  ACCEPTABLE but below 80% target")
        verdict = "MARGINAL"
    else:
        print("❌ NEEDS IMPROVEMENT - Below 70% accuracy")
        verdict = "FAIL"
    print("=" * 80)
    
    return verdict, accuracy


if __name__ == "__main__":
    verdict, accuracy = run_comprehensive_test()
    sys.exit(0 if verdict == "PASS" else 1)

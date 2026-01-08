"""
Automated Test Runner for Nexus LLM Analytics
Executes comprehensive test suite and generates detailed reports
"""

import asyncio
import httpx
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import sys

# Configuration
API_BASE_URL = "http://localhost:8000"
API_ANALYZE_ENDPOINT = f"{API_BASE_URL}/api/analyze"
TIMEOUT = 300  # 5 minutes per query
RESULTS_DIR = Path("test_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Test Suite Definitions
BASIC_TESTS = [
    {
        "id": "Q1.1",
        "category": "Statistical",
        "complexity": "Low",
        "file": "sales_data.csv",
        "query": "What are the basic statistics for sales and revenue?",
        "expected_agent": "StatisticalAgent"
    },
    {
        "id": "Q2.1",
        "category": "Financial",
        "complexity": "Low",
        "file": "comprehensive_ecommerce.csv",
        "query": "Calculate total revenue, average order value, and revenue by product category.",
        "expected_agent": "FinancialAgent"
    },
    {
        "id": "Q3.1",
        "category": "Time Series",
        "complexity": "Low",
        "file": "time_series_stock.csv",
        "query": "Show the price trend for TECH stock over the time period.",
        "expected_agent": "TimeSeriesAgent"
    },
    {
        "id": "Q5.1",
        "category": "SQL",
        "complexity": "Low",
        "file": "comprehensive_ecommerce.csv",
        "query": "Load this data into SQL and show me the top 10 orders by total_amount.",
        "expected_agent": "SQLAgent"
    },
]

ADVANCED_TESTS = [
    {
        "id": "Q1.3",
        "category": "Statistical",
        "complexity": "High",
        "file": "sales_data.csv",
        "query": "Test if the average sales differs significantly between North and South regions using a t-test.",
        "expected_agent": "StatisticalAgent"
    },
    {
        "id": "Q2.3",
        "category": "Financial",
        "complexity": "High",
        "file": "comprehensive_ecommerce.csv",
        "query": "Calculate Customer Lifetime Value (CLV) for each customer segment. Which segment has highest CLV?",
        "expected_agent": "FinancialAgent"
    },
    {
        "id": "Q3.4",
        "category": "Time Series",
        "complexity": "High",
        "file": "time_series_stock.csv",
        "query": "Build an ARIMA model to forecast TECH stock prices for the next 5 days.",
        "expected_agent": "TimeSeriesAgent"
    },
    {
        "id": "Q4.4",
        "category": "ML",
        "complexity": "High",
        "file": "StressLevelDataset.csv",
        "query": "Build a classification model to predict stress_level based on other features. What's the accuracy?",
        "expected_agent": "MLInsightsAgent"
    },
]

CROSS_AGENT_TESTS = [
    {
        "id": "Q6.1",
        "category": "Cross-Agent",
        "complexity": "High",
        "file": "comprehensive_ecommerce.csv",
        "query": "Perform statistical analysis on revenue distribution and identify financial outliers. What's the business impact?",
        "expected_agents": ["StatisticalAgent", "FinancialAgent"]
    },
    {
        "id": "Q6.2",
        "category": "Cross-Agent",
        "complexity": "High",
        "file": "time_series_stock.csv",
        "query": "Analyze TECH stock trend, then use ML to predict if price will increase or decrease tomorrow.",
        "expected_agents": ["TimeSeriesAgent", "MLInsightsAgent"]
    },
]

EDGE_CASE_TESTS = [
    {
        "id": "Q8.1",
        "category": "Edge Case",
        "complexity": "Medium",
        "file": "edge_cases/null_values.json",
        "query": "Analyze this data and handle null values appropriately.",
        "expected_agent": "DataAnalyst"
    },
    {
        "id": "Q8.2",
        "category": "Edge Case",
        "complexity": "Medium",
        "file": "edge_cases/mixed_types.json",
        "query": "Calculate summary statistics despite mixed data types in value field.",
        "expected_agent": "StatisticalAgent"
    },
    {
        "id": "Q17.1",
        "category": "Error Handling",
        "complexity": "Error Test",
        "file": "nonexistent_file.csv",
        "query": "Analyze nonexistent_file.csv",
        "expect_error": True
    },
]

# Suite mapping
TEST_SUITES = {
    "basic": BASIC_TESTS,
    "advanced": ADVANCED_TESTS,
    "cross-agent": CROSS_AGENT_TESTS,
    "edge-cases": EDGE_CASE_TESTS,
    "all": BASIC_TESTS + ADVANCED_TESTS + CROSS_AGENT_TESTS + EDGE_CASE_TESTS
}


class TestRunner:
    def __init__(self):
        self.results = []
        self.client = httpx.AsyncClient(timeout=TIMEOUT)
        self.start_time = None
        self.end_time = None
        
    async def run_test(self, test: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single test"""
        print(f"\n🧪 Running {test['id']}: {test['category']} - {test['complexity']}")
        print(f"   Query: {test['query'][:80]}...")
        
        start = time.time()
        result = {
            "test_id": test["id"],
            "category": test["category"],
            "complexity": test["complexity"],
            "query": test["query"],
            "file": test.get("file"),
            "expected_agent": test.get("expected_agent"),
            "expected_agents": test.get("expected_agents"),
            "expect_error": test.get("expect_error", False),
            "timestamp": datetime.now().isoformat(),
        }
        
        try:
            # Make API request
            payload = {
                "query": test["query"],
                "filename": test.get("file")
            }
            
            response = await self.client.post(API_ANALYZE_ENDPOINT, json=payload)
            elapsed = time.time() - start
            
            result["response_time"] = round(elapsed, 2)
            result["status_code"] = response.status_code
            
            if response.status_code == 200:
                data = response.json()
                result["success"] = data.get("status") == "success"
                result["actual_agent"] = data.get("agent")
                result["has_result"] = bool(data.get("result"))
                result["has_interpretation"] = bool(data.get("interpretation"))
                result["execution_time"] = data.get("execution_time")
                
                # Validate agent selection
                if test.get("expected_agent"):
                    result["agent_match"] = data.get("agent") == test["expected_agent"]
                elif test.get("expected_agents"):
                    result["agent_match"] = data.get("agent") in test["expected_agents"]
                else:
                    result["agent_match"] = None
                    
                # Check for expected error
                if test.get("expect_error"):
                    result["error_handled_correctly"] = not data.get("success", True)
                else:
                    result["error_handled_correctly"] = True
                
                print(f"   ✅ Success: {data.get('status')} | Agent: {data.get('agent')} | Time: {elapsed:.2f}s")
                
            else:
                result["success"] = False
                result["error"] = response.text
                print(f"   ❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            result["response_time"] = time.time() - start
            print(f"   ❌ Exception: {str(e)}")
        
        self.results.append(result)
        return result
    
    async def run_suite(self, suite_name: str):
        """Run a test suite"""
        print(f"\n{'='*80}")
        print(f"🚀 Starting Test Suite: {suite_name.upper()}")
        print(f"{'='*80}")
        
        tests = TEST_SUITES.get(suite_name, [])
        if not tests:
            print(f"❌ Unknown suite: {suite_name}")
            print(f"Available suites: {', '.join(TEST_SUITES.keys())}")
            return
        
        self.start_time = datetime.now()
        
        for test in tests:
            await self.run_test(test)
            await asyncio.sleep(1)  # Brief pause between tests
        
        self.end_time = datetime.now()
        self.generate_report(suite_name)
    
    def generate_report(self, suite_name: str):
        """Generate test report"""
        print(f"\n{'='*80}")
        print(f"📊 TEST REPORT: {suite_name.upper()}")
        print(f"{'='*80}")
        
        total_tests = len(self.results)
        successful = sum(1 for r in self.results if r.get("success"))
        failed = total_tests - successful
        
        agent_matches = sum(1 for r in self.results if r.get("agent_match") == True)
        agent_total = sum(1 for r in self.results if r.get("agent_match") is not None)
        
        avg_response_time = sum(r.get("response_time", 0) for r in self.results) / total_tests if total_tests > 0 else 0
        
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0
        
        print(f"\nSummary:")
        print(f"  Total Tests: {total_tests}")
        print(f"  ✅ Successful: {successful} ({successful/total_tests*100:.1f}%)")
        print(f"  ❌ Failed: {failed} ({failed/total_tests*100:.1f}%)")
        print(f"  🎯 Agent Match: {agent_matches}/{agent_total} ({agent_matches/agent_total*100:.1f}% if agent_total > 0 else 0)")
        print(f"  ⏱️  Avg Response Time: {avg_response_time:.2f}s")
        print(f"  🕐 Total Duration: {duration:.1f}s")
        
        # Category breakdown
        print(f"\nBy Category:")
        categories = {}
        for r in self.results:
            cat = r.get("category", "Unknown")
            if cat not in categories:
                categories[cat] = {"total": 0, "success": 0}
            categories[cat]["total"] += 1
            if r.get("success"):
                categories[cat]["success"] += 1
        
        for cat, stats in sorted(categories.items()):
            success_rate = stats["success"] / stats["total"] * 100
            print(f"  {cat}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
        
        # Save detailed results
        report_file = RESULTS_DIR / f"report_{suite_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump({
                "suite": suite_name,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": duration,
                "summary": {
                    "total": total_tests,
                    "successful": successful,
                    "failed": failed,
                    "agent_matches": agent_matches,
                    "avg_response_time": avg_response_time
                },
                "results": self.results
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {report_file}")
        print(f"{'='*80}\n")
    
    async def close(self):
        """Cleanup"""
        await self.client.aclose()


async def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python automated_test_runner.py <suite>")
        print(f"Available suites: {', '.join(TEST_SUITES.keys())}")
        sys.exit(1)
    
    suite_name = sys.argv[1]
    
    # Check if backend is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/api/health", timeout=5)
            if response.status_code != 200:
                print("❌ Backend health check failed!")
                print("Please ensure the backend is running: python -m src.backend.main")
                sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to backend at {API_BASE_URL}")
        print(f"Error: {e}")
        print("Please start the backend first: python -m src.backend.main")
        sys.exit(1)
    
    print("✅ Backend is running")
    
    # Run tests
    runner = TestRunner()
    try:
        await runner.run_suite(suite_name)
    finally:
        await runner.close()


if __name__ == "__main__":
    asyncio.run(main())

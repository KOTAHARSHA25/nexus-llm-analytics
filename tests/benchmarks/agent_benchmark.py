
import requests
import time
import json
import pandas as pd

BASE_URL = "http://localhost:8000/api"
ANALYZE_URL = f"{BASE_URL}/analyze/"
PREFS_URL = f"{BASE_URL}/models/preferences"

# Ensure Intelligent Routing is ON
config = {
    "auto_selection": True,
    "enable_intelligent_routing": True,
    "primary_model": "phi3:mini", 
    "review_model": "phi3:mini"
}
requests.post(PREFS_URL, json=config)

TESTS = [
    {
        "name": "Statistical Agent Test",
        "query": "Perform a correlation analysis between marketing_spend and revenue. Is it significant?",
        "filename": "sales_data.csv",
        "expected_agent": "StatisticalAgent"
    },
    {
        "name": "Financial Agent Test",
        "query": "Calculate the total revenue, profit margins, and identify the most profitable category.",
        "filename": "comprehensive_ecommerce.csv",
        "expected_agent": "FinancialAgent"
    },
    {
        "name": "Time Series Agent Test",
        "query": "Forecast the closing price for TECH stock for the next 5 days using ARIMA.",
        "filename": "time_series_stock.csv",
        "expected_agent": "TimeSeriesAgent"
    },
    {
        "name": "ML Insights Agent Test",
        "query": "Cluster these customers based on their spending behavior.",
        "filename": "comprehensive_ecommerce.csv",
        "expected_agent": "MLInsightsAgent"
    },
    {
        "name": "Planner Multi-Step",
        "query": "First summarize the sales data, then perform a correlation analysis.",
        "filename": "sales_data.csv",
        "expected_agent": "StatisticalAgent" 
    },
    {
        "name": "Dynamic Planner (CoT Loop)",
        "query": "Write a Python script to reverse a string and print it.",
        "filename": None, # No file needed for pure code gen
        "force_review": "mandatory",
        "expected_agent": "SelfCorrectionEngine"
    }
]

def run_test():
    print("=== STARTING SPECIALIZED AGENT BENCHMARK ===")
    results = []
    
    for t in TESTS:
        print(f"\nTesting: {t['name']}...")
        print(f"  Query: {t['query']}")
        print(f"  File: {t['filename']}")
        
        payload = {
            "query": t["query"],
            "filename": t.get("filename"), # Use get() as filename is optional now
            "force_refresh": True
        }
        if "force_review" in t:
            payload["review_level"] = t["force_review"]
        
        start = time.time()
        try:
            res = requests.post(ANALYZE_URL, json=payload, timeout=120)
            duration = time.time() - start
            
            if res.status_code == 200:
                data = res.json()
                agent = data.get("agent")
                # Also check metadata for routed info
                routed_agent = data.get("metadata", {}).get("routed_agent")
                used_agent = agent or routed_agent or "Unknown"
                
                success = data.get("success", False) or data.get("status") == "success"
                result_text = str(data.get("result", ""))[:100].replace("\n", " ") + "..."
                
                match = (t["expected_agent"] in used_agent)
                print(f"  -> Success: {success}")
                if not success:
                    with open("debug_agent_output.txt", "a") as f:
                        f.write(f"\n--- {t['name']} ---\n")
                        f.write(json.dumps(data, indent=2, default=str))
                print(f"  -> Agent Used: {used_agent} (Expected: {t['expected_agent']})")
                
                results.append({
                    "Test": t["name"],
                    "Success": success,
                    "Agent Used": used_agent,
                    "Expected": t["expected_agent"],
                    "Match": "✅" if match else "❌",
                    "Time": f"{duration:.2f}s",
                    "Preview": result_text
                })
            else:
                print(f"  -> FAILED (HTTP {res.status_code})")
                results.append({
                    "Test": t["name"],
                    "Success": False,
                    "Error": f"HTTP {res.status_code}"
                })
        except Exception as e:
            print(f"  -> CRASH: {e}")
            results.append({"Test": t["name"], "Success": False, "Error": str(e)})

    # Report
    print("\n=== AGENT BENCHMARK RESULTS ===")
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    df.to_csv("agent_benchmark_results.csv", index=False)

if __name__ == "__main__":
    run_test()

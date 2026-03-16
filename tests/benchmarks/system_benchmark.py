
import requests
import time
import json
import statistics
import pandas as pd
from typing import Dict, List, Any

BASE_URL = "http://localhost:8000/api"
UPLOAD_URL = f"{BASE_URL}/upload/"
ANALYZE_URL = f"{BASE_URL}/analyze/"
PREFS_URL = f"{BASE_URL}/models/preferences"
AVAILABLE_URL = f"{BASE_URL}/models/available"

# Queries designed to trigger different routing behaviors
TEST_QUERIES = [
    {
        "name": "Simple Query (Routing Test)",
        "query": "What is 25 * 4?",
        "expected_complexity": "Low/Simple",
        "dataset": None
    },
    {
        "name": "Complex Analysis (Routing Test)",
        "query": "Calculate the average sales and forecast next month based on trends.",
        "expected_complexity": "High/Complex",
        "dataset": "sales_data.csv"
    }
]

SCENARIOS = {
    "Scenario A (Intelligent)": {
        "primary_model": "phi3:mini", # Default medium
        "review_model": "phi3:mini",
        "embedding_model": "nomic-embed-text:latest",
        "auto_selection": True,
        "enable_intelligent_routing": True, # KEY: ON
        "allow_swap": True
    },
    "Scenario B (High Perf/Forced)": {
        "primary_model": "llama3.1:8b", # Forced big model
        "review_model": "llama3.1:8b",
        "embedding_model": "nomic-embed-text:latest",
        "auto_selection": False,
        "enable_intelligent_routing": False, # KEY: OFF
        "allow_swap": True
    },
    "Scenario C (Low Mem/Forced)": {
        "primary_model": "qwen2.5:0.5b", # Forced tiny model (if available, else smallest)
        "review_model": "qwen2.5:0.5b",
        "embedding_model": "nomic-embed-text:latest",
        "auto_selection": False,
        "enable_intelligent_routing": False, # KEY: OFF
        "allow_swap": False
    }
}

def get_installed_models():
    try:
        res = requests.get(AVAILABLE_URL)
        models = res.json().get("models", [])
        return [m["name"] for m in models if "embed" not in m["name"]]
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

def set_config(config: Dict):
    print(f"  -> Applying config: Routing={config['enable_intelligent_routing']}, Primary={config['primary_model']}")
    try:
        res = requests.post(PREFS_URL, json=config)
        if res.status_code != 200:
            print(f"  -> FAILED to set config: {res.text}")
    except Exception as e:
        print(f"  -> Error setting config: {e}")

def run_query(query_text: str, filename: str = None) -> Dict:
    payload = {"query": query_text, "force_refresh": True}
    if filename:
        payload["filename"] = filename
    
    start = time.time()
    try:
        res = requests.post(ANALYZE_URL, json=payload, timeout=120) # Increased timeout for real generation
        duration = time.time() - start
        
        if res.status_code == 200:
            data = res.json()
            result_text = str(data.get("result", ""))
            return {
                "success": True,
                "duration": duration,
                "model_used": data.get("execution_plan", {}).get("model") or data.get("metadata", {}).get("routed_model", "unknown"),
                "method": data.get("execution_plan", {}).get("execution_method") or data.get("metadata", {}).get("execution_method", "unknown"),
                "review": data.get("execution_plan", {}).get("review_level", "unknown"),
                "response_len": len(result_text),
                "preview": result_text[:50].replace("\n", " ") + "..."
            }
        else:
            return {"success": False, "error": f"HTTP {res.status_code}", "duration": duration}
    except Exception as e:
        return {"success": False, "error": str(e), "duration": time.time() - start}

def main():
    print("=== STARTING SYSTEM BENCHMARK ===")
    
    # 0. Check available models to adjust Scenario C if needed
    installed = get_installed_models()
    print(f"Installed models: {installed}")
    if "qwen2.5:0.5b" not in installed:
        # Fallback to smallest available
        fallback = installed[0] if installed else "phi3:mini"
        print(f"  -> qwen2.5:0.5b not found, falling back to {fallback} for Scenario C")
        SCENARIOS["Scenario C (Low Mem/Forced)"]["primary_model"] = fallback
        SCENARIOS["Scenario C (Low Mem/Forced)"]["review_model"] = fallback

    # 1. Ensure test data exists (upload dummy if needed, but we assume sales_data.csv is in data/)
    # For this script we assume local execution where data/samples/sales_data.csv exists
    # If using API upload, we'd do that here.
    
    results = []
    
    for scenario_name, config in SCENARIOS.items():
        print(f"\nrunning {scenario_name}...")
        set_config(config)
        time.sleep(2) # Allow settle
        
        for q in TEST_QUERIES:
            print(f"  Query: {q['name']}")
            latencies = []
            final_meta = {}
            
            # Run 3 times
            for i in range(3):
                print(f"    Run {i+1}/3...", end="", flush=True)
                res = run_query(q["query"], q["dataset"])
                if res["success"]:
                    latencies.append(res["duration"])
                    final_meta = res # Keep last metadata
                    print(f" Done ({res['duration']:.2f}s)")
                else:
                    print(f" FAILED ({res.get('error')})")
            
            if latencies:
                avg_time = statistics.mean(latencies)
                results.append({
                    "Scenario": scenario_name,
                    "Query Type": q["name"],
                    "Avg Time (s)": round(avg_time, 2),
                    "Model Used": final_meta.get("model_used"),
                    "Method": final_meta.get("method"),
                    "Success Rate": f"{len(latencies)}/3",
                    "Response Preview": final_meta.get("preview")
                })
    
    # Report
    print("\n=== BENCHMARK RESULTS ===")
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    
    # Save to file
    df.to_csv("benchmark_results.csv", index=False)
    print("Results saved to benchmark_results.csv")

if __name__ == "__main__":
    main()

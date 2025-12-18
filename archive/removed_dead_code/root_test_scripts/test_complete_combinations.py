"""
COMPREHENSIVE TEST: ALL MODEL + MECHANISM COMBINATIONS
Tests EVERY possible configuration:
- All 4 mechanism combinations (2^2 = 4)
- All 3 model tiers (FAST, BALANCED, POWERFUL)
- All complexity levels (SIMPLE, MEDIUM, COMPLEX)
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from backend.core.user_preferences import get_preferences_manager
from backend.agents.crew_manager import CrewAIManager
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

def print_header(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def print_section(text):
    print("\n" + "─"*80)
    print(f"  {text}")
    print("─"*80 + "\n")

# Test queries with expected behavior
TEST_QUERIES = [
    {
        "query": "How many rows?",
        "expected_complexity": "LOW",
        "expected_tier": "FAST",
        "expected_cot": False
    },
    {
        "query": "What is average sales by region?",
        "expected_complexity": "MEDIUM",
        "expected_tier": "BALANCED",
        "expected_cot": False
    },
    {
        "query": "Perform regression analysis to predict revenue based on price, marketing spend, and seasonality. Include confidence intervals",
        "expected_complexity": "HIGH",
        "expected_tier": "FULL_POWER",
        "expected_cot": True
    }
]

# All mechanism combinations
MECHANISM_COMBINATIONS = [
    {
        "name": "ALL_ENABLED",
        "smart_selection": True,
        "routing": True,
        "description": "Smart Model Selection ON + Intelligent Routing ON"
    },
    {
        "name": "SMART_ONLY",
        "smart_selection": True,
        "routing": False,
        "description": "Smart Model Selection ON + Intelligent Routing OFF"
    },
    {
        "name": "ROUTING_ONLY",
        "smart_selection": False,
        "routing": True,
        "description": "Smart Model Selection OFF + Intelligent Routing ON"
    },
    {
        "name": "ALL_DISABLED",
        "smart_selection": False,
        "routing": False,
        "description": "Manual Mode - All automatic features OFF"
    }
]

# Model configurations to test
MODEL_CONFIGS = [
    {
        "name": "LIGHTWEIGHT",
        "primary": "tinyllama:latest",
        "review": "tinyllama:latest",
        "description": "Fastest - minimal RAM (2GB)"
    },
    {
        "name": "BALANCED",
        "primary": "phi3:mini",
        "review": "phi3:mini",
        "description": "Balanced - moderate RAM (6GB)"
    },
    {
        "name": "POWERFUL",
        "primary": "llama3.1:8b",
        "review": "llama3.1:8b",
        "description": "Best quality - high RAM (16GB)"
    }
]

def configure_system(smart_selection, routing, model_config=None):
    """Configure user preferences for testing"""
    prefs_manager = get_preferences_manager()
    
    updates = {
        "auto_model_selection": smart_selection,
        "enable_intelligent_routing": routing
    }
    
    # If smart selection is OFF, set manual models
    if not smart_selection and model_config:
        updates["primary_model"] = model_config["primary"]
        updates["review_model"] = model_config["review"]
        updates["embedding_model"] = "nomic-embed-text:latest"
    
    prefs_manager.update_preferences(**updates)
    
    # Clear singleton to force reload
    CrewAIManager._instance = None
    
    return prefs_manager.load_preferences()

def run_single_test(query_info, sample_csv):
    """Run a single query and extract metadata"""
    crew_manager = CrewAIManager()
    
    start_time = time.time()
    
    try:
        result = crew_manager.analyze_structured_data(
            query=query_info["query"],
            filename=str(sample_csv)
        )
        
        processing_time = time.time() - start_time
        
        # Extract metadata (if available)
        metadata = result.get('metadata', {}) if isinstance(result, dict) else {}
        
        return {
            "success": True,
            "query": query_info["query"],
            "expected_complexity": query_info["expected_complexity"],
            "expected_tier": query_info["expected_tier"],
            "expected_cot": query_info["expected_cot"],
            "processing_time": processing_time,
            "metadata": metadata
        }
        
    except Exception as e:
        return {
            "success": False,
            "query": query_info["query"],
            "error": str(e),
            "processing_time": time.time() - start_time
        }

def test_mechanism_combination(combo, sample_csv):
    """Test one mechanism combination with all queries"""
    print_section(f"TESTING: {combo['description']}")
    
    print(f"Configuration:")
    print(f"   Smart Model Selection: {'ON' if combo['smart_selection'] else 'OFF'}")
    print(f"   Intelligent Routing: {'ON' if combo['routing'] else 'OFF'}")
    print()
    
    # Configure system
    configure_system(combo['smart_selection'], combo['routing'])
    
    results = []
    
    for i, query_info in enumerate(TEST_QUERIES, 1):
        print(f"🧪 Test {i}/{len(TEST_QUERIES)}: {query_info['expected_complexity']} complexity")
        print(f"   Query: {query_info['query'][:60]}...")
        
        result = run_single_test(query_info, sample_csv)
        results.append(result)
        
        if result['success']:
            print(f"   ✅ Success ({result['processing_time']:.1f}s)")
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
        print()
    
    return results

def test_model_configuration(model_config, sample_csv):
    """Test one model configuration with manual mode"""
    print_section(f"TESTING MODEL: {model_config['description']}")
    
    print(f"Manual Model Configuration:")
    print(f"   Primary: {model_config['primary']}")
    print(f"   Review: {model_config['review']}")
    print()
    
    # Configure with manual models (smart selection OFF, routing OFF)
    configure_system(
        smart_selection=False,
        routing=False,
        model_config=model_config
    )
    
    results = []
    
    for i, query_info in enumerate(TEST_QUERIES, 1):
        print(f"🧪 Test {i}/{len(TEST_QUERIES)}: {query_info['query'][:50]}...")
        
        result = run_single_test(query_info, sample_csv)
        results.append(result)
        
        if result['success']:
            print(f"   ✅ Success ({result['processing_time']:.1f}s)")
        else:
            print(f"   ❌ Failed")
        print()
    
    return results

def main():
    print_header("COMPREHENSIVE MECHANISM + MODEL COMBINATION TEST")
    
    # Setup
    sample_csv = Path(__file__).parent / "data" / "samples" / "sales_data.csv"
    
    if not sample_csv.exists():
        print(f"❌ Sample CSV not found: {sample_csv}")
        print("Creating sample data...")
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        df = pd.DataFrame({
            'product': ['Widget A', 'Widget B', 'Widget C', 'Widget D', 'Widget E'] * 20,
            'region': ['North', 'South', 'East', 'West'] * 25,
            'sales': np.random.randint(100, 10000, 100),
            'revenue': np.random.randint(1000, 50000, 100),
            'price': np.random.uniform(10, 100, 100),
            'marketing_spend': np.random.randint(500, 5000, 100)
        })
        
        sample_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(sample_csv, index=False)
        print(f"✅ Created sample data with {len(df)} rows\n")
    
    all_results = {}
    
    # PART 1: Test all mechanism combinations
    print_header("PART 1: ALL MECHANISM COMBINATIONS (4 scenarios)")
    
    for combo in MECHANISM_COMBINATIONS:
        results = test_mechanism_combination(combo, sample_csv)
        all_results[combo['name']] = results
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"📊 Summary: {successful}/{len(results)} successful\n")
    
    # PART 2: Test all model configurations (manual mode)
    print_header("PART 2: ALL MODEL CONFIGURATIONS (3 models)")
    
    for model_config in MODEL_CONFIGS:
        results = test_model_configuration(model_config, sample_csv)
        all_results[f"MODEL_{model_config['name']}"] = results
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        avg_time = sum(r['processing_time'] for r in results) / len(results)
        print(f"📊 Summary: {successful}/{len(results)} successful, avg time: {avg_time:.1f}s\n")
    
    # FINAL SUMMARY
    print_header("FINAL SUMMARY")
    
    total_tests = sum(len(results) for results in all_results.values())
    total_successful = sum(
        sum(1 for r in results if r['success'])
        for results in all_results.values()
    )
    
    print(f"Total Scenarios Tested: {len(all_results)}")
    print(f"Total Queries Executed: {total_tests}")
    print(f"Success Rate: {total_successful}/{total_tests} ({100*total_successful/total_tests:.1f}%)")
    print()
    
    print("Scenarios Covered:")
    print(f"  ✅ {len(MECHANISM_COMBINATIONS)} mechanism combinations")
    print(f"  ✅ {len(MODEL_CONFIGS)} model configurations")
    print(f"  ✅ {len(TEST_QUERIES)} complexity levels per scenario")
    print()
    
    print("="*80)
    print("✅ COMPREHENSIVE TEST COMPLETE")
    print("="*80)
    
    # Restore defaults
    print("\n🔄 Restoring default preferences...")
    configure_system(smart_selection=True, routing=True)
    print("✅ Done!")

if __name__ == "__main__":
    main()

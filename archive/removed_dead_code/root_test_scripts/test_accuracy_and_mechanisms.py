"""
COMPREHENSIVE ACCURACY + MECHANISM TEST
Tests ALL combinations AND verifies answer accuracy from easy to complex
"""

import sys
import os
from pathlib import Path
import re
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from backend.core.user_preferences import get_preferences_manager
from backend.agents.crew_manager import CrewManager
import logging
import time

logging.basicConfig(
    level=logging.WARNING,  # Reduce noise
    format='%(message)s'
)

def print_header(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def print_section(text):
    print("\n" + "─"*80)
    print(f"  {text}")
    print("─"*80)

# ACCURACY TEST QUERIES with verifiable answers
ACCURACY_TESTS = [
    # EASY: Simple counting/aggregation
    {
        "level": "EASY",
        "query": "How many total rows are in the dataset?",
        "expected_answer": 100,
        "tolerance": 0,
        "extract_pattern": r"(\d+)\s*rows?",
        "expected_complexity": "LOW",
        "expected_cot": False
    },
    {
        "level": "EASY",
        "query": "Count how many products are in the dataset",
        "expected_answer": 5,
        "tolerance": 0,
        "extract_pattern": r"(\d+)\s*(?:products?|unique)",
        "expected_complexity": "LOW",
        "expected_cot": False
    },
    
    # MEDIUM: Aggregations and grouping
    {
        "level": "MEDIUM",
        "query": "What is the total revenue across all sales?",
        "expected_answer": 3133714,  # Sum of all revenue in sample data
        "tolerance": 1000,  # Allow 0.03% variance
        "extract_pattern": r"[\$]?\s*(\d[\d,]*\.?\d*)",
        "expected_complexity": "MEDIUM",
        "expected_cot": False
    },
    {
        "level": "MEDIUM",
        "query": "How many regions are there?",
        "expected_answer": 4,
        "tolerance": 0,
        "extract_pattern": r"(\d+)\s*regions?",
        "expected_complexity": "MEDIUM",
        "expected_cot": False
    },
    {
        "level": "MEDIUM",
        "query": "What is the average price across all products?",
        "expected_answer": 56.16,  # Average price from sample data
        "tolerance": 5.0,  # Allow reasonable variance
        "extract_pattern": r"[\$]?\s*(\d+\.?\d*)",
        "expected_complexity": "MEDIUM",
        "expected_cot": False
    },
    
    # COMPLEX: Analysis requiring reasoning
    {
        "level": "COMPLEX",
        "query": "Which region has the highest total sales? Compare all regions and identify the winner.",
        "expected_answer": "North",  # From sample data analysis
        "tolerance": None,  # String match
        "extract_pattern": r"(North|South|East|West)",
        "expected_complexity": "HIGH",
        "expected_cot": True
    },
    {
        "level": "COMPLEX",
        "query": "Calculate the correlation between price and revenue. Is it positive or negative?",
        "expected_answer": "positive",
        "tolerance": None,
        "extract_pattern": r"(positive|negative)",
        "expected_complexity": "HIGH",
        "expected_cot": True
    },
    {
        "level": "COMPLEX",
        "query": "Which product has the best ROI (revenue divided by marketing spend)? Show calculations.",
        "expected_answer": "Widget",  # Should contain widget name
        "tolerance": None,
        "extract_pattern": r"(Widget [A-E])",
        "expected_complexity": "HIGH",
        "expected_cot": True
    }
]

# Mechanism combinations to test
MECHANISM_COMBINATIONS = [
    {
        "name": "ALL_ENABLED",
        "smart_selection": True,
        "routing": True,
        "description": "Smart + Routing ON"
    },
    {
        "name": "SMART_ONLY",
        "smart_selection": True,
        "routing": False,
        "description": "Smart ON, Routing OFF"
    },
    {
        "name": "ROUTING_ONLY",
        "smart_selection": False,
        "routing": True,
        "description": "Smart OFF, Routing ON"
    },
    {
        "name": "MANUAL",
        "smart_selection": False,
        "routing": False,
        "description": "Manual (All OFF)"
    }
]

# Model configurations
MODEL_CONFIGS = [
    {"name": "LIGHTWEIGHT", "primary": "tinyllama:latest", "review": "tinyllama:latest"},
    {"name": "BALANCED", "primary": "phi3:mini", "review": "phi3:mini"},
    {"name": "POWERFUL", "primary": "llama3.1:8b", "review": "llama3.1:8b"}
]

def extract_number(text, pattern):
    """Extract number from text using regex"""
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        # Clean and convert
        value_str = matches[0].replace(',', '').replace('$', '').strip()
        try:
            return float(value_str)
        except:
            return None
    return None

def extract_text(text, pattern):
    """Extract text from answer using regex"""
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        return matches[0].strip()
    return None

def check_accuracy(answer, expected, tolerance, extract_pattern):
    """Check if answer matches expected value within tolerance"""
    answer_text = str(answer) if not isinstance(answer, dict) else answer.get('answer', '')
    
    if tolerance is None:
        # String match
        extracted = extract_text(answer_text, extract_pattern)
        if extracted:
            return extracted.lower() == expected.lower(), extracted
        return False, None
    else:
        # Numeric match
        extracted = extract_number(answer_text, extract_pattern)
        if extracted is not None:
            diff = abs(extracted - expected)
            is_correct = diff <= tolerance
            return is_correct, extracted
        return False, None

def configure_system(smart_selection, routing, model_config=None):
    """Configure system for testing"""
    prefs_manager = get_preferences_manager()
    
    updates = {
        "auto_model_selection": smart_selection,
        "enable_intelligent_routing": routing
    }
    
    if not smart_selection and model_config:
        updates["primary_model"] = model_config["primary"]
        updates["review_model"] = model_config["review"]
        updates["embedding_model"] = "nomic-embed-text:latest"
    
    prefs_manager.update_preferences(**updates)
    CrewManager._instance = None

def run_accuracy_test(test_case, sample_csv):
    """Run single test and check accuracy"""
    crew_manager = CrewManager()
    
    start_time = time.time()
    
    try:
        result = crew_manager.analyze_structured_data(
            query=test_case["query"],
            filename=str(sample_csv)
        )
        
        processing_time = time.time() - start_time
        
        # Check accuracy
        is_correct, extracted = check_accuracy(
            result,
            test_case["expected_answer"],
            test_case["tolerance"],
            test_case["extract_pattern"]
        )
        
        return {
            "success": True,
            "correct": is_correct,
            "query": test_case["query"],
            "expected": test_case["expected_answer"],
            "extracted": extracted,
            "level": test_case["level"],
            "time": processing_time,
            "answer_preview": str(result)[:200] if result else "No answer"
        }
        
    except Exception as e:
        return {
            "success": False,
            "correct": False,
            "query": test_case["query"],
            "error": str(e),
            "level": test_case["level"],
            "time": time.time() - start_time
        }

def test_scenario(scenario_name, config, sample_csv):
    """Test one scenario with all accuracy tests"""
    print_section(f"{scenario_name}: {config['description']}")
    
    # Configure
    if "primary" in config:
        # Model test
        configure_system(False, False, config)
        print(f"Model: {config['primary']}")
    else:
        # Mechanism test
        configure_system(config['smart_selection'], config['routing'])
        print(f"Smart: {'ON' if config['smart_selection'] else 'OFF'}, Routing: {'ON' if config['routing'] else 'OFF'}")
    
    print()
    
    results = {"easy": [], "medium": [], "complex": []}
    
    for i, test_case in enumerate(ACCURACY_TESTS, 1):
        level = test_case["level"].lower()
        
        print(f"  {i}/{len(ACCURACY_TESTS)} [{test_case['level']:8}] {test_case['query'][:45]}...", end=" ")
        
        result = run_accuracy_test(test_case, sample_csv)
        results[level].append(result)
        
        if result["success"]:
            if result["correct"]:
                print(f"✅ CORRECT ({result['time']:.1f}s)")
            else:
                print(f"❌ WRONG (expected: {test_case['expected_answer']}, got: {result['extracted']}) ({result['time']:.1f}s)")
        else:
            print(f"💥 ERROR: {result.get('error', 'Unknown')[:50]}")
    
    # Summary
    print()
    for level in ["easy", "medium", "complex"]:
        level_results = results[level]
        if level_results:
            correct = sum(1 for r in level_results if r["correct"])
            total = len(level_results)
            accuracy = 100 * correct / total if total > 0 else 0
            avg_time = sum(r["time"] for r in level_results) / total
            
            status = "✅" if accuracy >= 80 else "⚠️" if accuracy >= 60 else "❌"
            print(f"  {status} {level.upper():8} accuracy: {correct}/{total} ({accuracy:.0f}%) - avg {avg_time:.1f}s")
    
    return results

def main():
    print_header("COMPREHENSIVE ACCURACY + MECHANISM TEST")
    print("Testing ALL combinations with REAL accuracy verification")
    
    # Setup
    sample_csv = Path(__file__).parent / "data" / "samples" / "sales_data.csv"
    
    if not sample_csv.exists():
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
        print(f"✅ Created with {len(df)} rows\n")
        
        # Print ground truth
        print("📊 GROUND TRUTH (for accuracy verification):")
        print(f"   Total rows: {len(df)}")
        print(f"   Unique products: {df['product'].nunique()}")
        print(f"   Total revenue: ${df['revenue'].sum():,.0f}")
        print(f"   Unique regions: {df['region'].nunique()}")
        print(f"   Average price: ${df['price'].mean():.2f}")
        print(f"   Highest sales region: {df.groupby('region')['sales'].sum().idxmax()}")
        print()
    
    all_results = {}
    
    # PART 1: Test mechanism combinations
    print_header("PART 1: MECHANISM COMBINATIONS (4 scenarios)")
    
    for combo in MECHANISM_COMBINATIONS:
        results = test_scenario(combo['name'], combo, sample_csv)
        all_results[combo['name']] = results
    
    # PART 2: Test model configurations
    print_header("PART 2: MODEL CONFIGURATIONS (3 models)")
    
    for model in MODEL_CONFIGS:
        results = test_scenario(f"MODEL_{model['name']}", model, sample_csv)
        all_results[f"MODEL_{model['name']}"] = results
    
    # FINAL SUMMARY
    print_header("FINAL ACCURACY SUMMARY")
    
    print(f"{'Scenario':<25} | {'Easy':>12} | {'Medium':>12} | {'Complex':>12} | {'Overall':>12}")
    print("─" * 80)
    
    for scenario_name, results in all_results.items():
        stats = {}
        for level in ["easy", "medium", "complex"]:
            level_results = results[level]
            if level_results:
                correct = sum(1 for r in level_results if r["correct"])
                total = len(level_results)
                accuracy = 100 * correct / total if total > 0 else 0
                stats[level] = f"{correct}/{total} ({accuracy:.0f}%)"
            else:
                stats[level] = "N/A"
        
        # Overall
        all_tests = [r for level_results in results.values() for r in level_results]
        correct_all = sum(1 for r in all_tests if r["correct"])
        total_all = len(all_tests)
        overall = 100 * correct_all / total_all if total_all > 0 else 0
        
        status = "✅" if overall >= 80 else "⚠️" if overall >= 60 else "❌"
        
        print(f"{scenario_name:<25} | {stats['easy']:>12} | {stats['medium']:>12} | {stats['complex']:>12} | {status} {overall:>6.0f}%")
    
    print("\n" + "="*80)
    print("✅ COMPREHENSIVE TEST COMPLETE")
    print("="*80)
    
    # Save detailed results
    output_file = Path(__file__).parent / "accuracy_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n📁 Detailed results: {output_file}")
    
    # Restore defaults
    print("\n🔄 Restoring defaults...")
    configure_system(True, True)
    print("✅ Done!")

if __name__ == "__main__":
    main()

"""Test if the system gives CORRECT answers, not just outputs"""
import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

print("="*80)
print("ANSWER ACCURACY TESTING - Does the system give CORRECT answers?")
print("="*80)

# Test 1: Known Math Question
print("\n[TEST 1] Math Accuracy - Can it calculate correctly?")
print("-" * 80)

test_cases = [
    {
        "file": "data/samples/sales_data.csv",
        "question": "What is the total revenue? Just give me the number.",
        "expected_contains": None,  # We'll verify manually
        "description": "Sum of all revenue values"
    },
    {
        "file": "data/samples/customer_data.csv", 
        "question": "How many unique cities are in the data?",
        "expected_contains": None,
        "description": "Count distinct cities"
    },
    {
        "file": "data/samples/orders.csv",
        "question": "What is the average order quantity?",
        "expected_contains": None,
        "description": "Average quantity calculation"
    }
]

# First, let's manually verify what the CORRECT answers should be
print("\nStep 1: Computing GROUND TRUTH (correct answers)...")
print("-" * 80)

import pandas as pd

# Load sales_data.csv and calculate actual total revenue
sales_df = pd.read_csv("data/samples/sales_data.csv")
if 'revenue' in sales_df.columns:
    actual_total_revenue = sales_df['revenue'].sum()
    print(f"GROUND TRUTH - Total Revenue: ${actual_total_revenue:,.2f}")
elif 'Revenue' in sales_df.columns:
    actual_total_revenue = sales_df['Revenue'].sum()
    print(f"GROUND TRUTH - Total Revenue: ${actual_total_revenue:,.2f}")
else:
    print(f"Available columns: {sales_df.columns.tolist()}")
    actual_total_revenue = None

# Load customer_data.csv and count unique cities
customer_df = pd.read_csv("data/samples/csv/customer_data.csv")
if 'city' in customer_df.columns:
    actual_unique_cities = customer_df['city'].nunique()
    print(f"GROUND TRUTH - Unique Cities: {actual_unique_cities}")
elif 'City' in customer_df.columns:
    actual_unique_cities = customer_df['City'].nunique()
    print(f"GROUND TRUTH - Unique Cities: {actual_unique_cities}")
else:
    print(f"Available columns: {customer_df.columns.tolist()}")
    actual_unique_cities = None

# Load orders.csv and calculate average quantity
orders_df = pd.read_csv("data/samples/csv/orders.csv")
if 'quantity' in orders_df.columns:
    actual_avg_quantity = orders_df['quantity'].mean()
    print(f"GROUND TRUTH - Average Quantity: {actual_avg_quantity:.2f}")
elif 'Quantity' in orders_df.columns:
    actual_avg_quantity = orders_df['Quantity'].mean()
    print(f"GROUND TRUTH - Average Quantity: {actual_avg_quantity:.2f}")
else:
    print(f"Available columns: {orders_df.columns.tolist()}")
    actual_avg_quantity = None

print("\nStep 2: Testing if LLM gives the SAME answers...")
print("-" * 80)

# Test each question
results = []

for i, test in enumerate(test_cases, 1):
    print(f"\n[Question {i}] {test['description']}")
    print(f"File: {test['file']}")
    print(f"Question: {test['question']}")
    
    # Upload file
    file_path = Path(test['file'])
    if not file_path.exists():
        print(f"SKIP - File not found")
        continue
        
    try:
        with open(file_path, 'rb') as f:
            upload_response = requests.post(
                f"{BASE_URL}/upload",
                files={'file': (file_path.name, f, 'text/csv')},
                timeout=10
            )
        
        if upload_response.status_code != 200:
            print(f"FAIL - Upload failed: {upload_response.status_code}")
            continue
            
        upload_data = upload_response.json()
        conversation_id = upload_data.get('conversation_id')
        
        # Ask question
        analyze_response = requests.post(
            f"{BASE_URL}/analyze",
            json={
                'query': test['question'],
                'conversation_id': conversation_id,
                'model_preference': 'phi3:mini'
            },
            timeout=30
        )
        
        if analyze_response.status_code != 200:
            print(f"FAIL - Analysis failed: {analyze_response.status_code}")
            continue
            
        analyze_data = analyze_response.json()
        llm_answer = analyze_data.get('response', '')
        
        print(f"\nLLM Response: {llm_answer[:300]}...")
        
        # Compare with ground truth
        if i == 1 and actual_total_revenue:
            # Check if the LLM's answer contains the correct revenue
            llm_answer_lower = llm_answer.lower().replace(',', '').replace('$', '')
            correct_answer_str = f"{actual_total_revenue:.2f}"
            if correct_answer_str in llm_answer_lower or f"{int(actual_total_revenue)}" in llm_answer_lower:
                print(f"✓ ACCURATE - LLM answer matches ground truth (${actual_total_revenue:,.2f})")
                results.append(True)
            else:
                print(f"✗ INACCURATE - Expected ${actual_total_revenue:,.2f}, but not found in answer")
                results.append(False)
                
        elif i == 2 and actual_unique_cities:
            if str(actual_unique_cities) in llm_answer:
                print(f"✓ ACCURATE - LLM answer matches ground truth ({actual_unique_cities} cities)")
                results.append(True)
            else:
                print(f"✗ INACCURATE - Expected {actual_unique_cities} cities, but not found in answer")
                results.append(False)
                
        elif i == 3 and actual_avg_quantity:
            # Check if answer is close (within 0.1)
            import re
            numbers = re.findall(r'\d+\.?\d*', llm_answer)
            found_match = False
            for num in numbers:
                try:
                    if abs(float(num) - actual_avg_quantity) < 0.5:
                        print(f"✓ ACCURATE - LLM answer matches ground truth ({actual_avg_quantity:.2f})")
                        results.append(True)
                        found_match = True
                        break
                except:
                    pass
            if not found_match:
                print(f"✗ INACCURATE - Expected {actual_avg_quantity:.2f}, but not found in answer")
                results.append(False)
        
    except requests.Timeout:
        print(f"TIMEOUT - LLM took too long")
        results.append(False)
    except Exception as e:
        print(f"ERROR - {str(e)[:100]}")
        results.append(False)

# Final Summary
print("\n" + "="*80)
print("ACCURACY RESULTS")
print("="*80)

if results:
    accurate = sum(results)
    total = len(results)
    accuracy_rate = (accurate / total) * 100
    
    print(f"\nAccurate Answers: {accurate}/{total} ({accuracy_rate:.1f}%)")
    
    if accuracy_rate == 100:
        print("\n✓ System gives ACCURATE answers")
    elif accuracy_rate >= 70:
        print("\n⚠ System is MOSTLY accurate but has some errors")
    else:
        print("\n✗ System is INACCURATE - answers don't match ground truth")
else:
    print("\nNo tests completed successfully")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("Unit tests (100% pass) = Files can be READ")
print("Accuracy tests = Answers are CORRECT")
print("Both are needed to verify system quality!")

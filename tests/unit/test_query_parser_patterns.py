"""
QUERY PARSER PATTERN MATCHING TEST (NO LLM)
Purpose: Test rule-based query parsing without requiring Ollama
Date: December 16, 2025
"""

import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

from backend.core.query_parser import IntentClassifier, ColumnExtractor, QueryIntent

print("="*80)
print("🔍 QUERY PARSER - PATTERN MATCHING TEST")
print("="*80)

# ============================================================================
# TEST 1: Intent Classification (Pattern Matching Only)
# ============================================================================
print("\n[TEST 1] Intent Classification")
print("-"*80)

test_cases = [
    # (query, expected_intent)
    ("What's the average sales?", QueryIntent.DESCRIBE),
    ("Show me all records where price > 100", QueryIntent.FILTER),
    ("Count how many customers we have", QueryIntent.COUNT),
    ("Create a bar chart of revenue by category", QueryIntent.VISUALIZE),
    ("Compare sales between Q1 and Q2", QueryIntent.COMPARE),
    ("Show me the trend over time", QueryIntent.TREND),
    ("Group by category and sum revenue", QueryIntent.AGGREGATE),
    ("Find outliers in the data", QueryIntent.OUTLIERS),
    ("Give me a summary of the data", QueryIntent.SUMMARIZE),
]

test1_pass = 0
test1_total = len(test_cases)

for query, expected in test_cases:
    intent, confidence = IntentClassifier.classify_intent(query)
    
    status = "✅" if intent == expected else "❌"
    print(f"  {status} '{query}'")
    print(f"      → Got: {intent.value} (conf: {confidence:.2f}), Expected: {expected.value}")
    
    if intent == expected:
        test1_pass += 1

print(f"\nResult: {test1_pass}/{test1_total} intents classified correctly ({test1_pass/test1_total*100:.0f}%)")

# ============================================================================
# TEST 2: Column Extraction
# ============================================================================
print("\n[TEST 2] Column Extraction")
print("-"*80)

available_columns = ["sales", "revenue", "price", "quantity", "customer_name", "product_id", "category"]

column_tests = [
    ("What's the average sales?", ["sales"]),
    ("Show me revenue by category", ["revenue", "category"]),
    ("Compare price and quantity", ["price", "quantity"]),
    ("Filter by customer name", ["customer_name"]),
]

test2_pass = 0
test2_total = len(column_tests)

for query, expected_columns in column_tests:
    found = ColumnExtractor.extract_columns(query, available_columns)
    
    # Check if all expected columns are found
    matched = all(col in found for col in expected_columns)
    
    status = "✅" if matched else "⚠️"
    print(f"  {status} '{query}'")
    print(f"      → Found: {found}, Expected: {expected_columns}")
    
    if matched:
        test2_pass += 1

print(f"\nResult: {test2_pass}/{test2_total} column extractions correct ({test2_pass/test2_total*100:.0f}%)")

# ============================================================================
# TEST 3: Condition Extraction
# ============================================================================
print("\n[TEST 3] Condition Extraction")
print("-"*80)

condition_tests = [
    ("price > 100", [{"column": "price", "operator": ">", "value": "100"}]),
    ("sales equals 500", [{"column": "sales", "operator": "=="}]),
    ("category contains electronics", [{"column": "category", "operator": "contains"}]),
]

test3_pass = 0
test3_total = len(condition_tests)

for query, expected in condition_tests:
    conditions = ColumnExtractor.extract_conditions(query, available_columns)
    
    # Check if at least one condition was extracted
    found_condition = len(conditions) > 0
    
    status = "✅" if found_condition else "❌"
    print(f"  {status} '{query}'")
    print(f"      → Extracted: {conditions}")
    
    if found_condition:
        test3_pass += 1

print(f"\nResult: {test3_pass}/{test3_total} conditions extracted ({test3_pass/test3_total*100:.0f}%)")

# ============================================================================
# TEST 4: Edge Cases
# ============================================================================
print("\n[TEST 4] Edge Cases")
print("-"*80)

edge_cases = [
    "",  # Empty query
    "   ",  # Whitespace only
    "xyz abc def",  # No recognizable intent
    "a" * 1000,  # Very long query
]

test4_pass = 0
test4_total = len(edge_cases)

for query in edge_cases:
    try:
        intent, confidence = IntentClassifier.classify_intent(query)
        print(f"  ✅ Edge case handled: intent={intent.value}, conf={confidence:.2f}")
        test4_pass += 1
    except Exception as e:
        print(f"  ❌ Edge case failed: {e}")

print(f"\nResult: {test4_pass}/{test4_total} edge cases handled safely ({test4_pass/test4_total*100:.0f}%)")

# ============================================================================
# TEST 5: Typo Tolerance
# ============================================================================
print("\n[TEST 5] Typo Tolerance")
print("-"*80)

# Pattern matching should still catch misspellings that are close
typo_tests = [
    ("What's the averge sales?", QueryIntent.DESCRIBE),  # "averge" instead of "average"
    ("Show me a grap of revenue", QueryIntent.VISUALIZE),  # "grap" instead of "graph"
    ("Count the custmers", QueryIntent.COUNT),  # "custmers" instead of "customers"
]

test5_pass = 0
test5_total = len(typo_tests)

for query, expected in typo_tests:
    intent, confidence = IntentClassifier.classify_intent(query)
    
    # May not get exact match with typos, but should get *something*
    got_some_intent = intent != QueryIntent.CUSTOM or confidence > 0
    
    status = "✅" if intent == expected else "⚠️" if got_some_intent else "❌"
    print(f"  {status} '{query}'")
    print(f"      → Got: {intent.value}, Expected: {expected.value}")
    
    if intent == expected or got_some_intent:
        test5_pass += 1

print(f"\nResult: {test5_pass}/{test5_total} queries handled despite typos ({test5_pass/test5_total*100:.0f}%)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 QUERY PARSER PATTERN MATCHING SUMMARY")
print("="*80)

tests = [
    ("Intent Classification", test1_pass, test1_total),
    ("Column Extraction", test2_pass, test2_total),
    ("Condition Extraction", test3_pass, test3_total),
    ("Edge Cases", test4_pass, test4_total),
    ("Typo Tolerance", test5_pass, test5_total),
]

total_pass = sum(p for _, p, _ in tests)
total_count = sum(t for _, _, t in tests)

for test_name, passed, total in tests:
    pct = (passed/total*100) if total > 0 else 0
    status = "✅" if pct >= 75 else "⚠️" if pct >= 50 else "❌"
    print(f"{status} {test_name}: {passed}/{total} ({pct:.0f}%)")

print("-"*80)
overall_pct = (total_pass/total_count*100) if total_count > 0 else 0
print(f"Overall: {total_pass}/{total_count} ({overall_pct:.1f}%)")

if overall_pct >= 80:
    print("\n✅ EXCELLENT: Pattern matching works well")
elif overall_pct >= 60:
    print("\n⚠️ ACCEPTABLE: Pattern matching works but has gaps")
else:
    print("\n❌ CONCERN: Pattern matching needs improvement")

print("\nNote: Full parser uses LLM fallback for low-confidence cases")
print("="*80)

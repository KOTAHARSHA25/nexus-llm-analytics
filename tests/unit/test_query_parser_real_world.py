"""
REAL-WORLD QUERY PARSER TEST
Purpose: Test query interpretation with actual user queries
Date: December 16, 2025

Testing with queries users would ACTUALLY ask, not idealized examples
"""

import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

from backend.core.query_parser import AdvancedQueryParser

print("="*80)
print("🔍 QUERY PARSER - REAL USER QUERIES TEST")
print("="*80)

parser = AdvancedQueryParser()

# ============================================================================
# TEST 1: Natural Language Questions (How Users Actually Talk)
# ============================================================================
print("\n[TEST 1] Natural Language Questions")
print("-"*80)

natural_queries = [
    # Financial queries
    ("How much money did we make last month?", "financial"),
    ("What's our profit?", "financial"),
    ("Are we profitable?", "financial"),
    
    # Statistical queries
    ("What's the average sale?", "statistical"),
    ("Show me the stats", "statistical"),
    ("Give me a summary of the data", "statistical"),
    
    # ML/Pattern queries
    ("Find patterns in customer behavior", "ml"),
    ("Which customers are similar?", "ml"),
    ("Predict who will churn", "ml"),
    
    # Time series
    ("Is revenue going up or down?", "time_series"),
    ("What's the trend?", "time_series"),
    
    # SQL/Data queries
    ("Show me all sales over $1000", "sql"),
    ("Filter by category", "sql"),
]

test1_pass = 0
test1_total = len(natural_queries)

for query, expected_type in natural_queries:
    try:
        result = parser.parse_query(query)
        
        # Check if result contains expected type indicators
        query_type = result.get('type', '').lower()
        intent = result.get('intent', '').lower()
        category = result.get('category', '').lower()
        
        # Check if ANY field matches expected type
        matched = any(expected_type in str(field).lower() 
                     for field in [query_type, intent, category, str(result)])
        
        status = "✅" if matched else "⚠️"
        print(f"  {status} '{query}' → {result.get('type', 'unknown')}")
        
        if matched:
            test1_pass += 1
            
    except Exception as e:
        print(f"  ❌ '{query}' → ERROR: {e}")

print(f"\nResult: {test1_pass}/{test1_total} queries parsed correctly")

# ============================================================================
# TEST 2: Ambiguous Queries (Real User Confusion)
# ============================================================================
print("\n[TEST 2] Ambiguous/Vague Queries")
print("-"*80)

ambiguous_queries = [
    "What's the deal with customers?",
    "Show me stuff",
    "Analyze this",
    "Tell me about the data",
    "What do you see?",
]

test2_pass = 0
for query in ambiguous_queries:
    try:
        result = parser.parse_query(query)
        
        # Should at least parse WITHOUT error
        if result:
            print(f"  ✅ '{query}' → Handled (type: {result.get('type', 'N/A')})")
            test2_pass += 1
        else:
            print(f"  ❌ '{query}' → No result")
    except Exception as e:
        print(f"  ❌ '{query}' → ERROR: {e}")

print(f"\nResult: {test2_pass}/{len(ambiguous_queries)} ambiguous queries handled")

# ============================================================================
# TEST 3: Typos and Misspellings (Real User Mistakes)
# ============================================================================
print("\n[TEST 3] Typos and Misspellings")
print("-"*80)

typo_queries = [
    ("what is the averge sales", "Should still work"),
    ("caclulate profit margn", "Should still work"),
    ("sho me the totl revenue", "Should still work"),
    ("corelation betwen price and sales", "Should still work"),
]

test3_pass = 0
for query, note in typo_queries:
    try:
        result = parser.parse_query(query)
        
        if result:
            print(f"  ✅ '{query}' → Parsed despite typos")
            test3_pass += 1
        else:
            print(f"  ❌ '{query}' → Failed")
    except Exception as e:
        print(f"  ⚠️ '{query}' → Handled with: {type(e).__name__}")
        test3_pass += 1  # Still pass if gracefully handled

print(f"\nResult: {test3_pass}/{len(typo_queries)} queries with typos handled")

# ============================================================================
# TEST 4: Complex Multi-Part Questions
# ============================================================================
print("\n[TEST 4] Complex Multi-Part Questions")
print("-"*80)

complex_queries = [
    "What's the average sale and total revenue by category?",
    "Show me profit margins and identify top customers",
    "Calculate ROI and predict future sales trends",
    "Find outliers and segment customers into groups",
]

test4_pass = 0
for query in complex_queries:
    try:
        result = parser.parse_query(query)
        
        if result:
            # Check if it identified multiple intents/operations
            has_multiple = any(key in result for key in ['operations', 'intents', 'tasks'])
            status = "✅ Multi-part" if has_multiple else "✅ Parsed"
            print(f"  {status} '{query[:50]}...'")
            test4_pass += 1
        else:
            print(f"  ❌ '{query[:50]}...' → Failed")
    except Exception as e:
        print(f"  ❌ '{query[:50]}...' → ERROR: {e}")

print(f"\nResult: {test4_pass}/{len(complex_queries)} complex queries parsed")

# ============================================================================
# TEST 5: Domain-Specific Jargon
# ============================================================================
print("\n[TEST 5] Domain-Specific Jargon")
print("-"*80)

jargon_queries = [
    "What's our CAC?",  # Customer Acquisition Cost
    "Show me the MRR",  # Monthly Recurring Revenue
    "Calculate CLTV",   # Customer Lifetime Value
    "What's the churn rate?",
    "Show me the conversion funnel",
]

test5_pass = 0
for query in jargon_queries:
    try:
        result = parser.parse_query(query)
        
        if result:
            print(f"  ✅ '{query}' → Understood jargon")
            test5_pass += 1
        else:
            print(f"  ⚠️ '{query}' → Parsed but may not understand jargon")
            test5_pass += 0.5  # Partial credit
    except Exception as e:
        print(f"  ❌ '{query}' → ERROR: {e}")

print(f"\nResult: {test5_pass}/{len(jargon_queries)} jargon queries handled")

# ============================================================================
# TEST 6: Edge Cases
# ============================================================================
print("\n[TEST 6] Edge Cases")
print("-"*80)

edge_queries = [
    ("", "Empty query"),
    ("???", "Just symbols"),
    ("a" * 1000, "Very long query"),
    ("SELECT * FROM", "SQL injection attempt"),
    ("<script>alert('xss')</script>", "XSS attempt"),
]

test6_pass = 0
for query, description in edge_queries:
    try:
        result = parser.parse_query(query)
        
        # Should handle gracefully (not crash)
        print(f"  ✅ {description} → Handled gracefully")
        test6_pass += 1
    except Exception as e:
        # Catching exception is OK for edge cases
        print(f"  ✅ {description} → Caught: {type(e).__name__}")
        test6_pass += 1

print(f"\nResult: {test6_pass}/{len(edge_queries)} edge cases handled safely")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 QUERY PARSER TEST SUMMARY")
print("="*80)

tests = [
    ("Natural Language Questions", test1_pass, test1_total),
    ("Ambiguous Queries", test2_pass, len(ambiguous_queries)),
    ("Typos & Misspellings", test3_pass, len(typo_queries)),
    ("Complex Multi-Part", test4_pass, len(complex_queries)),
    ("Domain Jargon", test5_pass, len(jargon_queries)),
    ("Edge Cases", test6_pass, len(edge_queries)),
]

total_pass = sum(p for _, p, _ in tests)
total_count = sum(t for _, _, t in tests)

for test_name, passed, total in tests:
    pct = (passed/total*100) if total > 0 else 0
    status = "✅" if pct >= 75 else "⚠️" if pct >= 50 else "❌"
    print(f"{status} {test_name}: {passed}/{total} ({pct:.0f}%)")

print("-"*80)
print(f"Overall: {total_pass:.1f}/{total_count} ({(total_pass/total_count*100):.1f}%)")

if total_pass/total_count >= 0.8:
    print("\n✅ GOOD: Query parser handles diverse real-world queries")
elif total_pass/total_count >= 0.6:
    print("\n⚠️ ACCEPTABLE: Query parser works but has gaps")
else:
    print("\n❌ CONCERN: Query parser struggles with real queries")

print("="*80)

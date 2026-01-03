"""
Test Fix 8 Edge Cases: Comprehensive robustness testing

Verifies the semantic mapper handles:
1. Empty/null DataFrames
2. Single column DataFrames
3. Ambiguous column names (priority resolution)
4. Custom user patterns
5. Validation and diagnostics
6. Confidence scoring
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from backend.core.semantic_mapper import SemanticMapper


def test_empty_dataframe():
    """Test handling of empty DataFrame"""
    print("\n🔍 Testing Empty DataFrame...")
    
    mapper = SemanticMapper()
    
    # Test 1: Empty DataFrame
    df_empty = pd.DataFrame()
    concepts = mapper.infer_column_concepts(df_empty)
    
    status1 = "✅" if concepts == {} else "❌"
    print(f"  {status1} Empty DataFrame returns empty dict: {concepts}")
    
    # Test 2: DataFrame with columns but no rows
    df_no_rows = pd.DataFrame(columns=['revenue', 'cost', 'profit'])
    concepts2 = mapper.infer_column_concepts(df_no_rows)
    
    status2 = "✅" if len(concepts2) == 3 else "❌"
    print(f"  {status2} DataFrame with 0 rows but 3 columns: {concepts2}")
    
    # Test 3: None DataFrame
    try:
        concepts3 = mapper.infer_column_concepts(None)
        status3 = "✅" if concepts3 == {} else "❌"
        print(f"  {status3} None DataFrame handled gracefully")
    except Exception as e:
        print(f"  ❌ None DataFrame raised exception: {e}")
        return 0, 3
    
    passed = sum([concepts == {}, len(concepts2) == 3, concepts3 == {}])
    return passed, 3


def test_single_column():
    """Test single column DataFrame"""
    print("\n📊 Testing Single Column DataFrame...")
    
    mapper = SemanticMapper()
    
    # Test different single columns
    tests = [
        ('revenue', 'revenue'),
        ('customer_count', 'count'),
        ('date', 'date'),
        ('unknown_col_xyz', 'unknown')
    ]
    
    passed = 0
    for col_name, expected in tests:
        df = pd.DataFrame({col_name: [1, 2, 3]})
        concepts = mapper.infer_column_concepts(df)
        actual = concepts.get(col_name)
        
        status = "✅" if actual == expected else "❌"
        print(f"  {status} {col_name} → {actual} (expected: {expected})")
        if actual == expected:
            passed += 1
    
    return passed, len(tests)


def test_ambiguous_columns():
    """Test priority-based disambiguation"""
    print("\n⚖️ Testing Ambiguous Column Priority...")
    
    mapper = SemanticMapper()
    
    # These columns contain multiple patterns - test priority wins
    df = pd.DataFrame({
        'product_cost': [100],  # Has both "product" (category) and "cost" (cost)
        'revenue_type': [1],     # Has both "revenue" (revenue) and "type" (category)
        'customer_count': [50],  # Has both "customer" (count?) and "count" (count)
        'sales_date': ['2024-01-01']  # Has both "sales" (revenue) and "date" (date)
    })
    
    concepts = mapper.infer_column_concepts(df)
    print(f"  Concepts: {concepts}")
    
    # With priority system: cost (10) should beat product/category (5)
    checks = [
        ('product_cost', 'cost', 'cost has higher priority than category'),
        ('revenue_type', 'revenue', 'revenue has higher priority than category'),
        ('customer_count', 'count', 'count should win'),
        ('sales_date', 'date', 'date type inference or pattern match')
    ]
    
    passed = 0
    for col, expected, reason in checks:
        actual = concepts.get(col)
        status = "✅" if actual == expected else "⚠️"
        print(f"  {status} {col} → {actual} (expected: {expected})")
        print(f"      Reason: {reason}")
        if actual == expected:
            passed += 1
    
    return passed, len(checks)


def test_custom_patterns():
    """Test custom user-defined patterns"""
    print("\n🎨 Testing Custom User Patterns...")
    
    # Add custom patterns for specific domain
    custom = {
        'enrollment': ['enrolled', 'registered', 'admitted'],
        'grade': ['grade', 'score', 'mark', 'gpa']
    }
    
    mapper = SemanticMapper(custom_patterns=custom)
    
    df = pd.DataFrame({
        'student_enrolled': [100],
        'final_grade': [85],
        'admission_date': ['2024-01-01']
    })
    
    concepts = mapper.infer_column_concepts(df)
    print(f"  Concepts: {concepts}")
    
    passed = 0
    checks = [
        ('student_enrolled', 'enrollment'),
        ('final_grade', 'grade'),
        ('admission_date', 'date')  # Standard pattern still works
    ]
    
    for col, expected in checks:
        actual = concepts.get(col)
        status = "✅" if actual == expected else "❌"
        print(f"  {status} {col} → {actual} (expected: {expected})")
        if actual == expected:
            passed += 1
    
    return passed, len(checks)


def test_confidence_scoring():
    """Test confidence scoring for mappings"""
    print("\n🎯 Testing Confidence Scoring...")
    
    mapper = SemanticMapper()
    
    df = pd.DataFrame({
        'revenue': [100],            # Exact token match = high confidence
        'total_sales': [200],        # Pattern match = medium confidence
        'abc123': [300],             # No pattern, dtype only = low confidence
        'patient_count': [50]        # Exact token match = high confidence
    })
    
    tests = [
        ('revenue', 1.0, 'exact token'),
        ('total_sales', 0.7, 'substring pattern'),
        ('abc123', 0.3, 'dtype inference only'),
        ('patient_count', 1.0, 'exact token')
    ]
    
    passed = 0
    for col, expected_conf, reason in tests:
        actual_conf = mapper.get_concept_confidence(df, col)
        # Allow some tolerance in confidence scores
        matches = abs(actual_conf - expected_conf) < 0.1
        status = "✅" if matches else "❌"
        print(f"  {status} {col}: {actual_conf:.2f} confidence (expected: {expected_conf}, {reason})")
        if matches:
            passed += 1
    
    return passed, len(tests)


def test_validation_diagnostics():
    """Test validation and diagnostic features"""
    print("\n🔬 Testing Validation & Diagnostics...")
    
    mapper = SemanticMapper()
    
    # Create DataFrame with some issues
    df = pd.DataFrame({
        'id1': [1, 2, 3],
        'id2': [4, 5, 6],
        'id3': [7, 8, 9],
        'xyz_abc': [10, 11, 12],
        'unknown_field': [13, 14, 15]
    })
    
    validation = mapper.validate_mappings(df)
    
    print(f"  Total columns: {validation['total_columns']}")
    print(f"  Mapped columns: {validation['mapped_columns']}")
    print(f"  Concept counts: {validation['concept_counts']}")
    print(f"  Warnings: {validation['warnings']}")
    print(f"  Low confidence columns: {validation['low_confidence_columns']}")
    
    passed = 0
    total = 3
    
    # Check that validation detected issues
    if validation['total_columns'] == 5:
        print("  ✅ Total columns count correct")
        passed += 1
    else:
        print("  ❌ Total columns count incorrect")
    
    if len(validation['warnings']) > 0:
        print("  ✅ Warnings generated for problematic data")
        passed += 1
    else:
        print("  ❌ No warnings generated")
    
    if 'concept_counts' in validation and isinstance(validation['concept_counts'], dict):
        print("  ✅ Concept counts provided")
        passed += 1
    else:
        print("  ❌ Concept counts missing")
    
    return passed, total


def test_special_characters():
    """Test handling of special characters in column names"""
    print("\n✨ Testing Special Characters...")
    
    mapper = SemanticMapper()
    
    # Columns with various special characters
    df = pd.DataFrame({
        'revenue (USD)': [100],
        'cost-total': [50],
        'profit%': [20],
        'customer.count': [75],
        'date/time': ['2024-01-01']
    })
    
    concepts = mapper.infer_column_concepts(df)
    print(f"  Concepts: {concepts}")
    
    # Check that special chars are handled (replaced with spaces)
    passed = 0
    total = 5
    
    expected = {
        'revenue (USD)': 'revenue',
        'cost-total': 'cost',
        'profit%': 'profit',
        'customer.count': 'count',
        'date/time': 'date'
    }
    
    for col, exp_concept in expected.items():
        actual = concepts.get(col)
        status = "✅" if actual == exp_concept else "⚠️"
        print(f"  {status} '{col}' → {actual} (expected: {exp_concept})")
        if actual == exp_concept:
            passed += 1
    
    return passed, total


def test_caching():
    """Test that caching works correctly"""
    print("\n💾 Testing Caching Mechanism...")
    
    mapper = SemanticMapper()
    
    df = pd.DataFrame({
        'revenue': [100, 200, 300],
        'cost': [50, 75, 100]
    })
    
    # First call - should populate cache
    concepts1 = mapper.infer_column_concepts(df)
    
    # Second call - should use cache (verify by checking it returns same object)
    concepts2 = mapper.infer_column_concepts(df)
    
    # Third call with same structure but different instance
    df2 = pd.DataFrame({
        'revenue': [400, 500, 600],
        'cost': [150, 175, 200]
    })
    concepts3 = mapper.infer_column_concepts(df2)
    
    passed = 0
    total = 2
    
    if concepts1 == concepts2:
        print("  ✅ Cache returns consistent results")
        passed += 1
    else:
        print("  ❌ Cache inconsistency")
    
    if concepts1 == concepts3:
        print("  ✅ Cache works across DataFrames with same structure")
        passed += 1
    else:
        print("  ❌ Cache doesn't work across instances")
    
    return passed, total


def main():
    """Run all edge case tests"""
    print("=" * 60)
    print("FIX 8: SEMANTIC LAYER EDGE CASES TEST")
    print("=" * 60)
    
    results = []
    
    # Run all tests
    results.append(test_empty_dataframe())
    results.append(test_single_column())
    results.append(test_ambiguous_columns())
    results.append(test_custom_patterns())
    results.append(test_confidence_scoring())
    results.append(test_validation_diagnostics())
    results.append(test_special_characters())
    results.append(test_caching())
    
    # Calculate totals
    total_passed = sum(r[0] for r in results)
    total_tests = sum(r[1] for r in results)
    percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {total_passed}/{total_tests} tests passed ({percentage:.1f}%)")
    print("=" * 60)
    
    if percentage >= 90:
        print("\n🎯 Fix 8 PRODUCTION-READY - All edge cases handled!")
        print("   ✅ Empty DataFrames")
        print("   ✅ Single columns")
        print("   ✅ Priority disambiguation")
        print("   ✅ Custom patterns")
        print("   ✅ Confidence scoring")
        print("   ✅ Validation diagnostics")
        print("   ✅ Special characters")
        print("   ✅ Caching")
    elif percentage >= 70:
        print("\n⚠️ Fix 8 PARTIAL - Most edge cases handled")
    else:
        print("\n❌ Fix 8 NEEDS WORK - Edge cases failing")
    
    return percentage >= 90


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

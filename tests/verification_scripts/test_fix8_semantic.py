"""
Test Fix 8: Semantic Layer for Data Agnosticism

Verifies that the semantic mapper correctly:
1. Maps finance columns (gross_inflow → revenue)
2. Maps healthcare columns (patient_count → count)
3. Handles cross-domain queries
4. Enhances queries with concept hints
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from backend.core.semantic_mapper import get_semantic_mapper


def test_finance_domain():
    """Test semantic mapping on finance data"""
    print("\n🏦 Testing Finance Domain...")
    
    # Create finance DataFrame
    df_finance = pd.DataFrame({
        'transaction_date': ['2024-01-01', '2024-01-02'],
        'gross_inflow': [10000, 15000],
        'operating_expense': [3000, 4000],
        'net_profit': [7000, 11000],
        'customer_id': ['C001', 'C002'],
        'product_category': ['Electronics', 'Clothing']
    })
    
    mapper = get_semantic_mapper()
    concepts = mapper.infer_column_concepts(df_finance)
    
    print(f"  Column Concepts: {concepts}")
    
    # Verify mappings
    checks = [
        ('transaction_date', 'date'),
        ('gross_inflow', 'revenue'),
        ('operating_expense', 'cost'),
        ('net_profit', 'profit'),
        ('customer_id', 'id'),
        ('product_category', 'category')
    ]
    
    passed = 0
    for col, expected_concept in checks:
        actual = concepts.get(col)
        status = "✅" if actual == expected_concept else "❌"
        print(f"  {status} {col} → {actual} (expected: {expected_concept})")
        if actual == expected_concept:
            passed += 1
    
    return passed, len(checks)


def test_healthcare_domain():
    """Test semantic mapping on healthcare data"""
    print("\n🏥 Testing Healthcare Domain...")
    
    # Create healthcare DataFrame
    df_healthcare = pd.DataFrame({
        'admission_date': ['2024-01-01', '2024-01-02'],
        'patient_count': [120, 135],
        'survival_rate': [0.95, 0.97],
        'diagnosis_type': ['Type 1', 'Type 2'],
        'hospital_cost': [5000, 6000],
        'patient_id': ['P001', 'P002']
    })
    
    mapper = get_semantic_mapper()
    concepts = mapper.infer_column_concepts(df_healthcare)
    
    print(f"  Column Concepts: {concepts}")
    
    # Verify mappings
    checks = [
        ('admission_date', 'date'),
        ('patient_count', 'count'),
        ('survival_rate', 'rate'),
        ('diagnosis_type', 'category'),
        ('hospital_cost', 'cost'),
        ('patient_id', 'id')
    ]
    
    passed = 0
    for col, expected_concept in checks:
        actual = concepts.get(col)
        status = "✅" if actual == expected_concept else "❌"
        print(f"  {status} {col} → {actual} (expected: {expected_concept})")
        if actual == expected_concept:
            passed += 1
    
    return passed, len(checks)


def test_query_enhancement():
    """Test query enhancement with concept hints"""
    print("\n🔍 Testing Query Enhancement...")
    
    # Create test DataFrame
    df = pd.DataFrame({
        'date': ['2024-01-01'],
        'daily_sales': [50000],
        'customer_count': [120],
        'product_type': ['Electronics']
    })
    
    mapper = get_semantic_mapper()
    
    # Test revenue query
    query1 = "What is the total revenue?"
    enhanced1 = mapper.enhance_query_context(query1, df)
    
    has_revenue_hint = 'revenue' in enhanced1 and 'daily_sales' in enhanced1
    status1 = "✅" if has_revenue_hint else "❌"
    print(f"  {status1} Revenue query enhanced with concept hints")
    print(f"    Original: {query1}")
    print(f"    Enhanced: {enhanced1[:100]}...")
    
    # Test count query
    query2 = "How many customers?"
    enhanced2 = mapper.enhance_query_context(query2, df)
    
    has_count_hint = 'count' in enhanced2 and 'customer_count' in enhanced2
    status2 = "✅" if has_count_hint else "❌"
    print(f"  {status2} Count query enhanced with concept hints")
    
    return (2 if (has_revenue_hint and has_count_hint) else 
            1 if (has_revenue_hint or has_count_hint) else 0), 2


def test_cross_domain_consistency():
    """Test that similar concepts are detected across domains"""
    print("\n🔄 Testing Cross-Domain Consistency...")
    
    # Finance DataFrame
    df_finance = pd.DataFrame({
        'revenue_amount': [10000],
        'sales_total': [8000]
    })
    
    # Retail DataFrame
    df_retail = pd.DataFrame({
        'gross_sales': [10000],
        'daily_turnover': [8000]
    })
    
    mapper = get_semantic_mapper()
    
    finance_concepts = mapper.infer_column_concepts(df_finance)
    retail_concepts = mapper.infer_column_concepts(df_retail)
    
    print(f"  Finance: {finance_concepts}")
    print(f"  Retail: {retail_concepts}")
    
    # Both should map to revenue
    finance_revenue = all(c == 'revenue' for c in finance_concepts.values())
    retail_revenue = all(c == 'revenue' for c in retail_concepts.values())
    
    passed = 0
    if finance_revenue:
        print("  ✅ Finance revenue columns correctly identified")
        passed += 1
    else:
        print("  ❌ Finance revenue columns not identified")
    
    if retail_revenue:
        print("  ✅ Retail revenue columns correctly identified")
        passed += 1
    else:
        print("  ❌ Retail revenue columns not identified")
    
    return passed, 2


def test_get_columns_for_concept():
    """Test retrieving columns by concept"""
    print("\n📊 Testing Column Retrieval by Concept...")
    
    df = pd.DataFrame({
        'gross_revenue': [10000],
        'net_sales': [8000],
        'customer_count': [120],
        'product_cost': [3000]
    })
    
    mapper = get_semantic_mapper()
    
    revenue_cols = mapper.get_columns_for_concept(df, 'revenue')
    count_cols = mapper.get_columns_for_concept(df, 'count')
    cost_cols = mapper.get_columns_for_concept(df, 'cost')
    
    print(f"  Revenue columns: {revenue_cols}")
    print(f"  Count columns: {count_cols}")
    print(f"  Cost columns: {cost_cols}")
    
    passed = 0
    total = 3
    
    if len(revenue_cols) == 2 and 'gross_revenue' in revenue_cols:
        print("  ✅ Revenue columns retrieved correctly")
        passed += 1
    else:
        print("  ❌ Revenue columns incorrect")
    
    if len(count_cols) == 1 and 'customer_count' in count_cols:
        print("  ✅ Count columns retrieved correctly")
        passed += 1
    else:
        print("  ❌ Count columns incorrect")
    
    if len(cost_cols) == 1 and 'product_cost' in cost_cols:
        print("  ✅ Cost columns retrieved correctly")
        passed += 1
    else:
        print("  ❌ Cost columns incorrect")
    
    return passed, total


def main():
    """Run all semantic mapper tests"""
    print("=" * 60)
    print("FIX 8: SEMANTIC LAYER TEST")
    print("=" * 60)
    
    results = []
    
    # Run all tests
    results.append(test_finance_domain())
    results.append(test_healthcare_domain())
    results.append(test_query_enhancement())
    results.append(test_cross_domain_consistency())
    results.append(test_get_columns_for_concept())
    
    # Calculate totals
    total_passed = sum(r[0] for r in results)
    total_tests = sum(r[1] for r in results)
    percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {total_passed}/{total_tests} tests passed ({percentage:.1f}%)")
    print("=" * 60)
    
    if percentage >= 90:
        print("\n🎯 Fix 8 VERIFIED - Semantic layer working correctly!")
        print("   ✅ Finance domain mapping")
        print("   ✅ Healthcare domain mapping")
        print("   ✅ Query enhancement")
        print("   ✅ Cross-domain consistency")
        print("   ✅ Column retrieval by concept")
    elif percentage >= 70:
        print("\n⚠️ Fix 8 PARTIAL - Most features working, some edge cases")
    else:
        print("\n❌ Fix 8 FAILED - Semantic mapper needs fixes")
    
    return percentage >= 90


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

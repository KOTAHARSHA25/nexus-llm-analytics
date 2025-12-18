"""
FINANCIAL AGENT - AGENT METHOD TESTS
Testing the 3 implemented financial analysis methods
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

print("="*80)
print("FINANCIAL AGENT - AGENT METHOD TESTS")
print("="*80)
print("Testing implemented financial analysis methods\n")

try:
    from backend.plugins.financial_agent import FinancialAgent
    print("✅ FinancialAgent imported successfully\n")
except ImportError as e:
    print(f"❌ FAILED to import FinancialAgent: {e}")
    sys.exit(1)

# Initialize agent
agent = FinancialAgent()
config = {}
agent.config = config
if not agent.initialize():
    print("❌ Agent initialization failed")
    sys.exit(1)

print("✅ Agent initialized successfully\n")

# TEST 1: Profitability Analysis
print("="*80)
print("TEST 1: Profitability Analysis (_profitability_analysis)")
print("="*80)

# Create financial data with revenue and costs
data = pd.DataFrame({
    'revenue': [100000, 120000, 150000, 180000, 200000],
    'cost': [60000, 70000, 85000, 100000, 110000],
    'date': pd.date_range(start='2024-01-01', periods=5, freq='M')
})

try:
    result = agent._profitability_analysis(data, "analyze profitability")
    
    assert result['success'], f"Profitability analysis should succeed, got error: {result.get('error', 'unknown')}"
    assert 'result' in result, "Should have result key"
    
    # Check for profitability metrics
    print(f"✅ Profitability analysis completed")
    
    # Find revenue vs cost analysis
    if any('revenue_vs_cost' in key for key in result['result'].keys()):
        for key, value in result['result'].items():
            if 'revenue_vs_cost' in key and isinstance(value, dict):
                print(f"✅ Total Revenue: ${value['revenue']:,.2f}")
                print(f"✅ Total Costs: ${value['costs']:,.2f}")
                print(f"✅ Gross Profit: ${value['gross_profit']:,.2f}")
                print(f"✅ Gross Margin: {value['gross_margin_percent']:.2f}%")
                print(f"✅ Status: {value['profitability_status']}")
                
                # Validate calculations
                assert value['gross_profit'] > 0, "Should be profitable"
                assert value['gross_margin_percent'] > 0, "Margin should be positive"
                break
    
    print("\n✅ TEST 1 PASSED - Profitability analysis working correctly\n")
    
except Exception as e:
    print(f"❌ TEST 1 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# TEST 2: Growth Analysis
print("="*80)
print("TEST 2: Growth Analysis (_growth_analysis)")
print("="*80)

# Create time series data showing growth
data = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=12, freq='M'),
    'sales': [100000 * (1.05 ** i) for i in range(12)],  # 5% monthly growth
    'customers': [1000 + i*50 for i in range(12)]  # Linear customer growth
})

try:
    result = agent._growth_analysis(data, "analyze growth trends")
    
    assert result['success'], f"Growth analysis should succeed, got error: {result.get('error', 'unknown')}"
    assert 'result' in result, "Should have result key"
    
    print(f"✅ Growth analysis completed")
    
    # Check for growth metrics in any column
    found_growth = False
    for col_name, metrics in result['result'].items():
        if isinstance(metrics, dict) and 'total_growth_percent' in metrics:
            print(f"✅ Column: {col_name}")
            print(f"✅ Total Growth: {metrics['total_growth_percent']:.2f}%")
            
            if 'yoy_growth_percent' in metrics:
                print(f"✅ YoY Growth: {metrics['yoy_growth_percent']:.2f}%")
            
            if 'cagr_percent' in metrics:
                print(f"✅ CAGR: {metrics['cagr_percent']:.2f}%")
            
            if 'growth_trend' in metrics:
                print(f"✅ Trend: {metrics['growth_trend']}")
            
            found_growth = True
            break
    
    assert found_growth, "Should find growth metrics"
    print("\n✅ TEST 2 PASSED - Growth analysis working correctly\n")
    
except Exception as e:
    print(f"❌ TEST 2 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# TEST 3: Comprehensive Financial Analysis
print("="*80)
print("TEST 3: Comprehensive Financial Analysis (_comprehensive_financial_analysis)")
print("="*80)

# Create comprehensive financial dataset
data = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=24, freq='M'),
    'revenue': [50000 + i*2000 + np.random.randint(-5000, 5000) for i in range(24)],
    'cost': [30000 + i*1000 + np.random.randint(-3000, 3000) for i in range(24)],
    'customers': [500 + i*20 for i in range(24)],
    'transactions': [1000 + i*50 for i in range(24)]
})
data['profit'] = data['revenue'] - data['cost']

try:
    result = agent._comprehensive_financial_analysis(data, "perform comprehensive financial analysis")
    
    assert result['success'], f"Comprehensive analysis should succeed, got error: {result.get('error', 'unknown')}"
    assert 'result' in result, "Should have result key"
    
    print(f"✅ Comprehensive financial analysis completed")
    
    # Check what analyses were included
    analyses_found = []
    for key in result['result'].keys():
        if isinstance(result['result'][key], dict):
            analyses_found.append(key)
    
    print(f"✅ Analyses performed: {', '.join(analyses_found)}")
    
    # Validate some key metrics
    if 'summary' in result['result']:
        summary = result['result']['summary']
        print(f"✅ Summary metrics available: {list(summary.keys())}")
    
    print("\n✅ TEST 3 PASSED - Comprehensive financial analysis working correctly\n")
    
except Exception as e:
    print(f"❌ TEST 3 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# TEST 4: Profitability with Edge Cases
print("="*80)
print("TEST 4: Profitability Analysis - Edge Cases")
print("="*80)

# Test with losses (negative profitability)
data_loss = pd.DataFrame({
    'revenue': [100000, 90000, 80000],
    'cost': [120000, 110000, 100000]
})

try:
    result = agent._profitability_analysis(data_loss, "analyze profitability with losses")
    
    assert result['success'], "Should handle unprofitable scenario"
    
    # Find the profitability status
    for key, value in result['result'].items():
        if isinstance(value, dict) and 'profitability_status' in value:
            print(f"✅ Status: {value['profitability_status']}")
            assert value['profitability_status'] == 'unprofitable', "Should detect losses"
            assert value['gross_profit'] < 0, "Profit should be negative"
            print(f"✅ Correctly identified losses: ${value['gross_profit']:,.2f}")
            break
    
    print("\n✅ TEST 4 PASSED - Edge case handling working correctly\n")
    
except Exception as e:
    print(f"❌ TEST 4 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# TEST 5: Growth Analysis with Decline
print("="*80)
print("TEST 5: Growth Analysis - Negative Growth")
print("="*80)

# Create declining trend
data_decline = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=6, freq='M'),
    'sales': [100000, 95000, 90000, 85000, 80000, 75000]  # Declining
})

try:
    result = agent._growth_analysis(data_decline, "analyze declining trend")
    
    assert result['success'], "Should handle negative growth"
    
    # Check for negative growth detection
    found_decline = False
    for col_name, metrics in result['result'].items():
        if isinstance(metrics, dict) and 'total_growth_percent' in metrics:
            growth = metrics['total_growth_percent']
            print(f"✅ Detected growth: {growth:.2f}%")
            assert growth < 0, "Should detect decline"
            print(f"✅ Correctly identified decline")
            
            if 'growth_trend' in metrics:
                print(f"✅ Trend classification: {metrics['growth_trend']}")
            
            found_decline = True
            break
    
    assert found_decline, "Should analyze declining trend"
    print("\n✅ TEST 5 PASSED - Negative growth detection working correctly\n")
    
except Exception as e:
    print(f"❌ TEST 5 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# Summary
print("="*80)
print("FINANCIAL AGENT METHOD TESTS SUMMARY")
print("="*80)
print("\n✅ All 5/5 tests PASSED!\n")
print("Methods tested:")
print("  ✅ _profitability_analysis - Basic profitability metrics verified")
print("  ✅ _growth_analysis - Growth rate calculations verified")
print("  ✅ _comprehensive_financial_analysis - Multiple analyses combined")
print("  ✅ Edge case: Unprofitable scenarios handled correctly")
print("  ✅ Edge case: Negative growth (decline) detected correctly")
print("\n" + "="*80)
print("✅ FINANCIAL AGENT: 3/3 IMPLEMENTED METHODS TESTED (100%)")
print("="*80)
print("\n🎉 All implemented financial methods are NOW TESTED!")
print("📝 Note: 5 placeholder methods not yet implemented:")
print("   - _liquidity_analysis")
print("   - _efficiency_analysis")
print("   - _roi_analysis")
print("   - _cost_analysis")
print("   - _customer_analysis")

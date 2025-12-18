"""
Comprehensive Test Suite for Statistical Analysis Agent
Tests all statistical methods: t-tests, ANOVA, regression, chi-square, correlation, outliers, normality
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from src.backend.plugins.statistical_agent import StatisticalAgent


def create_test_data():
    """Create test datasets for statistical analysis"""
    np.random.seed(42)
    
    # Dataset 1: For t-tests and ANOVA
    test_data_groups = pd.DataFrame({
        'group': ['A']*30 + ['B']*30 + ['C']*30,
        'value': np.concatenate([
            np.random.normal(100, 15, 30),  # Group A
            np.random.normal(110, 15, 30),  # Group B  
            np.random.normal(95, 15, 30)    # Group C
        ]),
        'gender': np.random.choice(['Male', 'Female'], 90),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 90)
    })
    
    # Dataset 2: For regression
    n = 100
    x1 = np.random.normal(50, 10, n)
    x2 = np.random.normal(30, 5, n)
    y = 10 + 2*x1 + 1.5*x2 + np.random.normal(0, 5, n)
    
    test_data_regression = pd.DataFrame({
        'predictor1': x1,
        'predictor2': x2,
        'outcome': y
    })
    
    # Dataset 3: For correlation
    test_data_correlation = pd.DataFrame({
        'height': np.random.normal(170, 10, 50),
        'weight': np.random.normal(70, 15, 50),
        'age': np.random.randint(20, 60, 50),
        'score': np.random.normal(75, 10, 50)
    })
    
    # Dataset 4: For chi-square
    test_data_categorical = pd.DataFrame({
        'smoking': np.random.choice(['Yes', 'No'], 100),
        'disease': np.random.choice(['Present', 'Absent'], 100),
        'gender': np.random.choice(['Male', 'Female'], 100),
        'outcome': np.random.choice(['Success', 'Failure'], 100)
    })
    
    return test_data_groups, test_data_regression, test_data_correlation, test_data_categorical


def test_descriptive_statistics(agent, data):
    """Test descriptive statistics"""
    print("\n" + "="*80)
    print("TEST 1: DESCRIPTIVE STATISTICS")
    print("="*80)
    
    query = "Provide descriptive statistics for the dataset"
    result = agent.execute(query, data=data)
    
    if result["success"]:
        print("✅ Descriptive statistics: SUCCESS")
        print(f"   - Dataset shape: {result['result']['data_info']['shape']}")
        print(f"   - Numeric columns: {len(result['result']['numeric_summary'])}")
        print(f"   - Categorical columns: {len(result['result']['categorical_summary'])}")
        print(f"   - Interpretation: {result['interpretation'][:150]}...")
        return True
    else:
        print(f"❌ Descriptive statistics FAILED: {result['error']}")
        return False


def test_t_test(agent, data):
    """Test t-test analysis"""
    print("\n" + "="*80)
    print("TEST 2: T-TEST ANALYSIS")
    print("="*80)
    
    query = "Perform t-test comparing group A and B"
    result = agent.execute(query, data=data, group_column='group', value_column='value', test_type='independent')
    
    if result["success"]:
        print("✅ T-test: SUCCESS")
        res = result['result']
        print(f"   - Test type: {res['test_type']}")
        print(f"   - T-statistic: {res['t_statistic']:.4f}")
        print(f"   - P-value: {res['p_value']:.4f}")
        print(f"   - Significant: {res['significant']}")
        print(f"   - Cohen's d: {res['cohens_d']:.4f} ({res['effect_size_interpretation']})")
        print(f"   - Interpretation: {result['interpretation']}")
        return True
    else:
        print(f"❌ T-test FAILED: {result['error']}")
        return False


def test_anova(agent, data):
    """Test ANOVA analysis"""
    print("\n" + "="*80)
    print("TEST 3: ANOVA ANALYSIS")
    print("="*80)
    
    query = "Perform ANOVA comparing all three groups"
    result = agent.execute(query, data=data, group_column='group', value_column='value')
    
    if result["success"]:
        print("✅ ANOVA: SUCCESS")
        res = result['result']
        print(f"   - Test type: {res['test_type']}")
        print(f"   - F-statistic: {res['f_statistic']:.4f}")
        print(f"   - P-value: {res['p_value']:.4f}")
        print(f"   - Significant: {res['significant']}")
        print(f"   - Eta-squared: {res['eta_squared']:.4f} ({res['effect_size_interpretation']})")
        print(f"   - Number of groups: {res['n_groups']}")
        
        if res['posthoc_comparisons']:
            sig_comparisons = sum(1 for v in res['posthoc_comparisons'].values() if v['significant'])
            print(f"   - Significant post-hoc comparisons: {sig_comparisons}/{len(res['posthoc_comparisons'])}")
        
        print(f"   - Interpretation: {result['interpretation']}")
        return True
    else:
        print(f"❌ ANOVA FAILED: {result['error']}")
        return False


def test_regression(agent, data):
    """Test regression analysis"""
    print("\n" + "="*80)
    print("TEST 4: REGRESSION ANALYSIS")
    print("="*80)
    
    query = "Perform regression analysis to predict outcome"
    result = agent.execute(query, data=data, dependent_variable='outcome', 
                          independent_variables=['predictor1', 'predictor2'])
    
    if result["success"]:
        print("✅ Regression: SUCCESS")
        res = result['result']
        print(f"   - Dependent variable: {res['dependent_variable']}")
        print(f"   - Independent variables: {res['independent_variables']}")
        print(f"   - R-squared: {res['model_statistics']['r_squared']:.4f}")
        print(f"   - Adjusted R-squared: {res['model_statistics']['adjusted_r_squared']:.4f}")
        print(f"   - RMSE: {res['model_statistics']['rmse']:.4f}")
        print(f"   - Model significant: {res['model_statistics']['model_significant']}")
        
        # Check significant predictors
        sig_predictors = [k for k, v in res['coefficients'].items() 
                         if k != 'intercept' and v['significant']]
        print(f"   - Significant predictors: {sig_predictors}")
        
        print(f"   - Interpretation: {result['interpretation']}")
        return True
    else:
        print(f"❌ Regression FAILED: {result['error']}")
        return False


def test_correlation(agent, data):
    """Test correlation analysis"""
    print("\n" + "="*80)
    print("TEST 5: CORRELATION ANALYSIS")
    print("="*80)
    
    query = "Analyze correlations between all variables"
    result = agent.execute(query, data=data)
    
    if result["success"]:
        print("✅ Correlation: SUCCESS")
        res = result['result']
        print(f"   - Correlation method: Pearson & Spearman")
        print(f"   - Number of variable pairs tested: {len(res['significance_tests'])}")
        
        # Count significant correlations
        sig_corr = sum(1 for test in res['significance_tests'].values() 
                      if test['pearson']['p_value'] < 0.05)
        print(f"   - Significant correlations (p<0.05): {sig_corr}")
        
        if res['strong_correlations']:
            print(f"   - Strong correlations (|r|>=0.7): {len(res['strong_correlations'])}")
            for corr in res['strong_correlations'][:3]:  # Show first 3
                print(f"     * {corr['variable1']} vs {corr['variable2']}: r={corr['correlation']:.3f}")
        
        print(f"   - Interpretation: {result['interpretation'][:150]}...")
        return True
    else:
        print(f"❌ Correlation FAILED: {result['error']}")
        return False


def test_chi_square(agent, data):
    """Test chi-square analysis"""
    print("\n" + "="*80)
    print("TEST 6: CHI-SQUARE TEST")
    print("="*80)
    
    query = "Test independence between smoking and disease"
    result = agent.execute(query, data=data, variable1='smoking', variable2='disease')
    
    if result["success"]:
        print("✅ Chi-square: SUCCESS")
        res = result['result']
        print(f"   - Test type: {res['test_type']}")
        print(f"   - Variables: {res['variables']}")
        print(f"   - Table shape: {res['table_shape']}")
        print(f"   - Chi-square statistic: {res['chi2_statistic']:.4f}")
        print(f"   - P-value: {res['p_value']:.4f}")
        print(f"   - Significant: {res['significant']}")
        print(f"   - Cramér's V: {res['cramers_v']:.4f} ({res['effect_size_interpretation']})")
        
        if res['significant_cells']:
            print(f"   - Cells with notable deviations: {len(res['significant_cells'])}")
        
        print(f"   - Interpretation: {result['interpretation']}")
        return True
    else:
        print(f"❌ Chi-square FAILED: {result['error']}")
        return False


def test_outlier_detection(agent, data):
    """Test outlier detection"""
    print("\n" + "="*80)
    print("TEST 7: OUTLIER DETECTION")
    print("="*80)
    
    query = "Detect outliers in the data"
    result = agent.execute(query, data=data)
    
    if result["success"]:
        print("✅ Outlier detection: SUCCESS")
        res = result['result']
        
        for col, methods in res.items():
            iqr_count = methods['iqr_method']['count']
            iqr_pct = methods['iqr_method']['percentage']
            print(f"   - {col}: {iqr_count} outliers ({iqr_pct:.1f}%) using IQR method")
        
        print(f"   - Interpretation: {result['interpretation'][:150]}...")
        return True
    else:
        print(f"❌ Outlier detection FAILED: {result['error']}")
        return False


def test_normality(agent, data):
    """Test normality testing"""
    print("\n" + "="*80)
    print("TEST 8: NORMALITY TESTS")
    print("="*80)
    
    query = "Test if data follows normal distribution"
    result = agent.execute(query, data=data)
    
    if result["success"]:
        print("✅ Normality testing: SUCCESS")
        res = result['result']
        
        for col, tests in res.items():
            if 'shapiro_wilk' in tests:
                sw = tests['shapiro_wilk']
                print(f"   - {col}: {'Normal' if sw['is_normal'] else 'Not normal'} (Shapiro-Wilk p={sw['p_value']:.4f})")
        
        print(f"   - Interpretation: {result['interpretation'][:150]}...")
        return True
    else:
        print(f"❌ Normality testing FAILED: {result['error']}")
        return False


def main():
    """Run all statistical agent tests"""
    print("="*80)
    print("STATISTICAL ANALYSIS AGENT - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Testing all statistical methods...")
    
    # Initialize agent
    print("\nInitializing Statistical Agent...")
    agent = StatisticalAgent()
    
    if not agent.initialize():
        print("❌ Failed to initialize Statistical Agent")
        return
    
    print("✅ Statistical Agent initialized successfully")
    
    # Create test data
    print("\nCreating test datasets...")
    data_groups, data_regression, data_correlation, data_categorical = create_test_data()
    print(f"✅ Test datasets created:")
    print(f"   - Groups data: {data_groups.shape}")
    print(f"   - Regression data: {data_regression.shape}")
    print(f"   - Correlation data: {data_correlation.shape}")
    print(f"   - Categorical data: {data_categorical.shape}")
    
    # Run tests
    results = []
    
    results.append(("Descriptive Statistics", test_descriptive_statistics(agent, data_groups)))
    results.append(("T-Test", test_t_test(agent, data_groups)))
    results.append(("ANOVA", test_anova(agent, data_groups)))
    results.append(("Regression", test_regression(agent, data_regression)))
    results.append(("Correlation", test_correlation(agent, data_correlation)))
    results.append(("Chi-Square", test_chi_square(agent, data_categorical)))
    results.append(("Outlier Detection", test_outlier_detection(agent, data_groups)))
    results.append(("Normality Tests", test_normality(agent, data_groups)))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "="*80)
    print(f"FINAL RESULT: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*80)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Statistical Agent is fully functional!")
        return 0
    else:
        print(f"\n⚠️ {total-passed} test(s) failed. Review output above for details.")
        return 1


if __name__ == "__main__":
    exit(main())

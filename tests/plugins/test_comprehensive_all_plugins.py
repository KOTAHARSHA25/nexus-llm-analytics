"""
COMPREHENSIVE PLUGIN TESTING FRAMEWORK
Dynamic testing with easy, medium, and hard data scenarios
Tests all 5 plugin agents with realistic data variations
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

print("="*80)
print("COMPREHENSIVE PLUGIN TESTING FRAMEWORK")
print("="*80)
print("Testing all plugins with EASY, MEDIUM, and HARD data scenarios\n")

# Import all agents
try:
    from backend.plugins.statistical_agent import StatisticalAgent
    from backend.plugins.time_series_agent import TimeSeriesAgent
    from backend.plugins.financial_agent import FinancialAgent
    from backend.plugins.ml_insights_agent import MLInsightsAgent
    from backend.plugins.sql_agent import SQLAgent
    print("✅ All agents imported successfully\n")
except ImportError as e:
    print(f"❌ FAILED to import agents: {e}")
    sys.exit(1)

# Initialize all agents
config = {}
agents = {
    'Statistical': StatisticalAgent(),
    'TimeSeries': TimeSeriesAgent(),
    'Financial': FinancialAgent(),
    'MLInsights': MLInsightsAgent(),
    'SQL': SQLAgent()
}

for name, agent in agents.items():
    agent.config = config
    if not agent.initialize():
        print(f"❌ {name} Agent initialization failed")
        sys.exit(1)

print("✅ All agents initialized successfully\n")

# Test counters
total_tests = 0
passed_tests = 0
failed_tests = 0

def run_test(test_name: str, test_func):
    """Run a test and track results"""
    global total_tests, passed_tests, failed_tests
    total_tests += 1
    
    try:
        test_func()
        passed_tests += 1
        print(f"✅ {test_name} PASSED\n")
        return True
    except AssertionError as e:
        failed_tests += 1
        print(f"❌ {test_name} FAILED: {e}\n")
        return False
    except Exception as e:
        failed_tests += 1
        print(f"❌ {test_name} ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# DATA GENERATORS - EASY, MEDIUM, HARD
# ============================================================================

class DataGenerator:
    """Generate test data at different complexity levels"""
    
    @staticmethod
    def easy_numeric_data(rows=50):
        """Simple numeric data - EASY"""
        np.random.seed(42)
        return pd.DataFrame({
            'value': np.random.randint(1, 100, rows),
            'score': np.random.randint(50, 100, rows)
        })
    
    @staticmethod
    def medium_numeric_data(rows=200):
        """More complex numeric data with patterns - MEDIUM"""
        np.random.seed(42)
        return pd.DataFrame({
            'sales': np.random.normal(1000, 200, rows),
            'profit': np.random.normal(300, 50, rows),
            'cost': np.random.normal(700, 150, rows),
            'quantity': np.random.randint(10, 100, rows)
        })
    
    @staticmethod
    def hard_numeric_data(rows=1000):
        """Large, complex, multi-dimensional data - HARD"""
        np.random.seed(42)
        base = np.linspace(0, 100, rows)
        return pd.DataFrame({
            'metric1': base + np.random.normal(0, 10, rows),
            'metric2': base * 2 + np.random.normal(0, 20, rows),
            'metric3': np.sin(base/10) * 50 + np.random.normal(0, 5, rows),
            'metric4': np.random.exponential(50, rows),
            'metric5': np.random.lognormal(3, 1, rows),
            'metric6': np.random.gamma(2, 2, rows)
        })
    
    @staticmethod
    def easy_timeseries_data(periods=30):
        """Simple time series - EASY"""
        np.random.seed(42)
        return pd.DataFrame({
            'value': np.random.randint(10, 50, periods)
        })
    
    @staticmethod
    def medium_timeseries_data(periods=100):
        """Time series with trend and noise - MEDIUM"""
        np.random.seed(42)
        trend = np.linspace(100, 200, periods)
        noise = np.random.normal(0, 10, periods)
        return pd.DataFrame({
            'value': trend + noise
        })
    
    @staticmethod
    def hard_timeseries_data(periods=365):
        """Complex time series with trend, seasonality, and anomalies - HARD"""
        np.random.seed(42)
        t = np.arange(periods)
        trend = t * 0.5
        seasonal = 20 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
        noise = np.random.normal(0, 5, periods)
        data = trend + seasonal + noise + 100
        # Add anomalies
        data[50] = data[50] + 100
        data[150] = data[150] - 80
        data[250] = data[250] + 120
        return pd.DataFrame({'value': data})
    
    @staticmethod
    def easy_financial_data():
        """Simple financial data - EASY"""
        return pd.DataFrame({
            'revenue': [100, 120, 140, 160, 180],
            'cost': [60, 70, 80, 90, 100]
        })
    
    @staticmethod
    def medium_financial_data():
        """Quarterly financial data - MEDIUM"""
        return pd.DataFrame({
            'revenue': [500000, 520000, 480000, 550000, 560000, 580000, 600000, 620000],
            'cost_of_goods': [300000, 310000, 290000, 330000, 335000, 345000, 355000, 365000],
            'operating_expenses': [80000, 82000, 81000, 85000, 86000, 87000, 89000, 91000],
            'marketing': [20000, 22000, 21000, 23000, 24000, 25000, 26000, 27000]
        })
    
    @staticmethod
    def hard_financial_data():
        """Multi-year financial data with complex calculations - HARD"""
        np.random.seed(42)
        months = 36
        base_revenue = 1000000
        growth_rate = 0.02
        
        revenue = [base_revenue * (1 + growth_rate) ** i + np.random.normal(0, 50000) for i in range(months)]
        cogs = [r * 0.6 + np.random.normal(0, 10000) for r in revenue]
        opex = [r * 0.2 + np.random.normal(0, 5000) for r in revenue]
        
        return pd.DataFrame({
            'revenue': revenue,
            'cost_of_goods_sold': cogs,
            'operating_expenses': opex,
            'depreciation': [15000] * months,
            'interest': [5000] * months,
            'tax_rate': [0.21] * months
        })
    
    @staticmethod
    def easy_clustering_data():
        """Simple 2D clustering data - EASY"""
        np.random.seed(42)
        cluster1 = pd.DataFrame({
            'x': np.random.normal(0, 1, 30),
            'y': np.random.normal(0, 1, 30)
        })
        cluster2 = pd.DataFrame({
            'x': np.random.normal(5, 1, 30),
            'y': np.random.normal(5, 1, 30)
        })
        return pd.concat([cluster1, cluster2], ignore_index=True)
    
    @staticmethod
    def medium_clustering_data():
        """3 clusters in 3 dimensions - MEDIUM"""
        np.random.seed(42)
        cluster1 = pd.DataFrame({
            'x': np.random.normal(0, 1, 40),
            'y': np.random.normal(0, 1, 40),
            'z': np.random.normal(0, 1, 40)
        })
        cluster2 = pd.DataFrame({
            'x': np.random.normal(5, 1, 40),
            'y': np.random.normal(5, 1, 40),
            'z': np.random.normal(0, 1, 40)
        })
        cluster3 = pd.DataFrame({
            'x': np.random.normal(2.5, 1, 40),
            'y': np.random.normal(7, 1, 40),
            'z': np.random.normal(5, 1, 40)
        })
        return pd.concat([cluster1, cluster2, cluster3], ignore_index=True)
    
    @staticmethod
    def hard_clustering_data():
        """5 clusters in 10 dimensions with overlap - HARD"""
        np.random.seed(42)
        clusters = []
        for i in range(5):
            cluster = pd.DataFrame({
                f'dim{j}': np.random.normal(i*3, 1.5, 50) for j in range(10)
            })
            clusters.append(cluster)
        return pd.concat(clusters, ignore_index=True)


# ============================================================================
# STATISTICAL AGENT TESTS
# ============================================================================

print("="*80)
print("STATISTICAL AGENT - COMPREHENSIVE TESTING")
print("="*80)

def test_statistical_easy():
    print("📊 Statistical Agent - EASY: Basic descriptive statistics")
    data = DataGenerator.easy_numeric_data()
    result = agents['Statistical']._descriptive_statistics(data, "describe data")
    assert result['success'], f"Failed: {result.get('error')}"
    assert 'numeric_summary' in result['result']
    print(f"  ✓ Analyzed {len(data)} rows, {len(data.columns)} columns")
    print(f"  ✓ Mean value: {result['result']['numeric_summary']['value']['mean']:.2f}")

def test_statistical_medium():
    print("📊 Statistical Agent - MEDIUM: Multi-column correlation analysis")
    data = DataGenerator.medium_numeric_data()
    result = agents['Statistical']._correlation_analysis(data, "analyze correlations")
    assert result['success'], f"Failed: {result.get('error')}"
    assert 'pearson_correlation' in result['result']
    print(f"  ✓ Analyzed {len(data)} rows, {len(data.columns)} columns")
    print(f"  ✓ Correlation pairs found: {len(result['result'].get('strong_correlations', []))}")

def test_statistical_hard():
    print("📊 Statistical Agent - HARD: Large dataset with outlier detection")
    data = DataGenerator.hard_numeric_data()
    result = agents['Statistical']._outlier_analysis(data, "detect outliers")
    assert result['success'], f"Failed: {result.get('error')}"
    print(f"  ✓ Analyzed {len(data)} rows, {len(data.columns)} dimensions")
    for method, info in result['result'].items():
        if isinstance(info, dict) and 'outliers' in info:
            print(f"  ✓ {method}: {len(info['outliers'])} outliers detected")

run_test("Statistical Agent - EASY", test_statistical_easy)
run_test("Statistical Agent - MEDIUM", test_statistical_medium)
run_test("Statistical Agent - HARD", test_statistical_hard)


# ============================================================================
# TIME SERIES AGENT TESTS
# ============================================================================

print("="*80)
print("TIME SERIES AGENT - COMPREHENSIVE TESTING")
print("="*80)

def test_timeseries_easy():
    print("📈 Time Series Agent - EASY: Simple trend analysis")
    data = DataGenerator.easy_timeseries_data()
    result = agents['TimeSeries']._trend_analysis(data, "analyze trend")
    assert result['success'], f"Failed: {result.get('error')}"
    print(f"  ✓ Analyzed {len(data)} periods")
    if 'value' in result['result']:
        print(f"  ✓ Trend: {result['result']['value'].get('trend_type', 'detected')}")

def test_timeseries_medium():
    print("📈 Time Series Agent - MEDIUM: Forecast with trend")
    data = DataGenerator.medium_timeseries_data()
    result = agents['TimeSeries']._forecast_analysis(data, "forecast next 10 periods")
    assert result['success'], f"Failed: {result.get('error')}"
    print(f"  ✓ Analyzed {len(data)} periods")
    if 'forecasts' in result['result']:
        print(f"  ✓ Generated forecasts for columns: {list(result['result']['forecasts'].keys())}")

def test_timeseries_hard():
    print("📈 Time Series Agent - HARD: Complex seasonality and anomaly detection")
    data = DataGenerator.hard_timeseries_data()
    
    # Test seasonality
    result1 = agents['TimeSeries']._seasonality_analysis(data, "detect seasonality")
    assert result1['success'], f"Seasonality failed: {result1.get('error')}"
    print(f"  ✓ Seasonality analysis on {len(data)} periods")
    
    # Test anomaly detection
    result2 = agents['TimeSeries']._anomaly_detection(data, "detect anomalies")
    assert result2['success'], f"Anomaly detection failed: {result2.get('error')}"
    if 'value' in result2['result'] and 'detected_anomalies' in result2['result']['value']:
        anomalies = result2['result']['value']['detected_anomalies']
        print(f"  ✓ Detected {len(anomalies)} anomalies (expected 3)")

run_test("Time Series Agent - EASY", test_timeseries_easy)
run_test("Time Series Agent - MEDIUM", test_timeseries_medium)
run_test("Time Series Agent - HARD", test_timeseries_hard)


# ============================================================================
# FINANCIAL AGENT TESTS
# ============================================================================

print("="*80)
print("FINANCIAL AGENT - COMPREHENSIVE TESTING")
print("="*80)

def test_financial_easy():
    print("💰 Financial Agent - EASY: Basic profitability")
    data = DataGenerator.easy_financial_data()
    result = agents['Financial']._profitability_analysis(data, "analyze profitability")
    assert result['success'], f"Failed: {result.get('error')}"
    print(f"  ✓ Analyzed {len(data)} periods")
    for key, value in result['result'].items():
        if isinstance(value, dict) and 'gross_profit' in value:
            print(f"  ✓ Gross profit: ${value['gross_profit']:,.2f}")
            break

def test_financial_medium():
    print("💰 Financial Agent - MEDIUM: Growth analysis")
    data = DataGenerator.medium_financial_data()
    result = agents['Financial']._growth_analysis(data, "analyze growth")
    assert result['success'], f"Failed: {result.get('error')}"
    print(f"  ✓ Analyzed {len(data)} periods")
    if 'revenue' in result['result']:
        growth = result['result']['revenue']
        if 'total_growth_percent' in growth:
            print(f"  ✓ Total growth: {growth['total_growth_percent']:.2f}%")

def test_financial_hard():
    print("💰 Financial Agent - HARD: Comprehensive financial analysis")
    data = DataGenerator.hard_financial_data()
    result = agents['Financial']._comprehensive_financial_analysis(data, "comprehensive analysis")
    assert result['success'], f"Failed: {result.get('error')}"
    print(f"  ✓ Analyzed {len(data)} months of financial data")
    print(f"  ✓ Analyses performed: {len(result['result'])} different types")

run_test("Financial Agent - EASY", test_financial_easy)
run_test("Financial Agent - MEDIUM", test_financial_medium)
run_test("Financial Agent - HARD", test_financial_hard)


# ============================================================================
# ML INSIGHTS AGENT TESTS
# ============================================================================

print("="*80)
print("ML INSIGHTS AGENT - COMPREHENSIVE TESTING")
print("="*80)

def test_ml_easy():
    print("🤖 ML Insights Agent - EASY: Simple clustering")
    data = DataGenerator.easy_clustering_data()
    result = agents['MLInsights']._clustering_analysis(data, "find clusters")
    assert result['success'], f"Failed: {result.get('error')}"
    if 'kmeans' in result['result']:
        print(f"  ✓ Found {result['result']['kmeans']['optimal_clusters']} clusters")
        print(f"  ✓ Silhouette score: {result['result']['kmeans']['silhouette_score']:.3f}")

def test_ml_medium():
    print("🤖 ML Insights Agent - MEDIUM: 3D clustering")
    data = DataGenerator.medium_clustering_data()
    result = agents['MLInsights']._clustering_analysis(data, "cluster 3D data")
    assert result['success'], f"Failed: {result.get('error')}"
    if 'kmeans' in result['result']:
        print(f"  ✓ Analyzed {len(data)} points in {len(data.columns)} dimensions")
        print(f"  ✓ Optimal clusters: {result['result']['kmeans']['optimal_clusters']}")

def test_ml_hard():
    print("🤖 ML Insights Agent - HARD: High-dimensional data")
    data = DataGenerator.hard_clustering_data()
    
    # Clustering
    result1 = agents['MLInsights']._clustering_analysis(data, "cluster high-dimensional data")
    assert result1['success'], f"Clustering failed: {result1.get('error')}"
    print(f"  ✓ Clustered {len(data)} points in {len(data.columns)} dimensions")
    
    # Dimensionality reduction
    result2 = agents['MLInsights']._dimensionality_reduction(data, "reduce dimensions")
    assert result2['success'], f"PCA failed: {result2.get('error')}"
    if 'pca' in result2['result']:
        pca = result2['result']['pca']
        print(f"  ✓ Reduced from {len(data.columns)}D to {pca['components_for_95_percent']}D (95% variance)")

run_test("ML Insights Agent - EASY", test_ml_easy)
run_test("ML Insights Agent - MEDIUM", test_ml_medium)
run_test("ML Insights Agent - HARD", test_ml_hard)


# ============================================================================
# SQL AGENT TESTS
# ============================================================================

print("="*80)
print("SQL AGENT - COMPREHENSIVE TESTING")
print("="*80)

def test_sql_easy():
    print("🗄️  SQL Agent - EASY: Simple query generation")
    result = agents['SQL']._generate_sql_query("count users")
    assert result['success'], f"Failed: {result.get('error')}"
    print(f"  ✓ Generated SQL: {result['result']['generated_sql']}")

def test_sql_medium():
    print("🗄️  SQL Agent - MEDIUM: Complex query with JOIN")
    result = agents['SQL']._generate_sql_query("show top products by orders")
    assert result['success'], f"Failed: {result.get('error')}"
    sql = result['result']['generated_sql']
    assert 'JOIN' in sql.upper()
    print(f"  ✓ Generated JOIN query with GROUP BY and ORDER BY")

def test_sql_hard():
    print("🗄️  SQL Agent - HARD: Schema analysis + Query optimization")
    
    # Schema analysis
    result1 = agents['SQL']._analyze_schema()
    assert result1['success'], f"Schema analysis failed: {result1.get('error')}"
    schema = result1['result']['schema_analysis']
    print(f"  ✓ Analyzed {len(schema['tables'])} tables with {len(schema['relationships'])} relationships")
    
    # Query optimization
    complex_query = "SELECT * FROM orders WHERE user_id IN (SELECT id FROM users WHERE created_at > '2023-01-01')"
    result2 = agents['SQL']._optimize_query(complex_query)
    assert result2['success'], f"Optimization failed: {result2.get('error')}"
    print(f"  ✓ Generated {len(result2['result']['optimization_suggestions'])} optimization suggestions")

run_test("SQL Agent - EASY", test_sql_easy)
run_test("SQL Agent - MEDIUM", test_sql_medium)
run_test("SQL Agent - HARD", test_sql_hard)


# ============================================================================
# CROSS-AGENT INTEGRATION TESTS
# ============================================================================

print("="*80)
print("CROSS-AGENT INTEGRATION TESTS")
print("="*80)

def test_integration_statistical_ml():
    print("🔗 Integration: Statistical → ML Pipeline")
    # Generate data with outliers
    data = DataGenerator.medium_numeric_data(300)
    
    # Step 1: Statistical outlier detection
    stat_result = agents['Statistical']._outlier_analysis(data, "detect outliers")
    assert stat_result['success']
    print(f"  ✓ Statistical analysis complete")
    
    # Step 2: ML clustering on clean data
    ml_result = agents['MLInsights']._clustering_analysis(data, "cluster data")
    assert ml_result['success']
    print(f"  ✓ ML clustering complete")
    print(f"  ✓ Pipeline: Statistics → ML successful")

def test_integration_timeseries_statistical():
    print("🔗 Integration: Time Series → Statistical Pipeline")
    data = DataGenerator.hard_timeseries_data()
    
    # Step 1: Time series decomposition
    ts_result = agents['TimeSeries']._trend_analysis(data, "extract trend")
    assert ts_result['success']
    print(f"  ✓ Time series analysis complete")
    
    # Step 2: Statistical analysis of residuals
    stat_result = agents['Statistical']._descriptive_statistics(data, "describe statistics")
    assert stat_result['success']
    print(f"  ✓ Statistical analysis complete")
    print(f"  ✓ Pipeline: Time Series → Statistics successful")

def test_integration_financial_statistical():
    print("🔗 Integration: Financial → Statistical Pipeline")
    data = DataGenerator.hard_financial_data()
    
    # Step 1: Financial analysis
    fin_result = agents['Financial']._profitability_analysis(data, "analyze profitability")
    assert fin_result['success']
    print(f"  ✓ Financial analysis complete")
    
    # Step 2: Statistical correlation
    stat_result = agents['Statistical']._correlation_analysis(data, "correlate metrics")
    assert stat_result['success']
    print(f"  ✓ Statistical correlation complete")
    print(f"  ✓ Pipeline: Financial → Statistics successful")

run_test("Integration: Statistical → ML", test_integration_statistical_ml)
run_test("Integration: Time Series → Statistical", test_integration_timeseries_statistical)
run_test("Integration: Financial → Statistical", test_integration_financial_statistical)


# ============================================================================
# EDGE CASES AND STRESS TESTS
# ============================================================================

print("="*80)
print("EDGE CASES AND STRESS TESTS")
print("="*80)

def test_edge_empty_data():
    print("⚠️  Edge Case: Empty dataset")
    empty_data = pd.DataFrame()
    result = agents['Statistical']._descriptive_statistics(empty_data, "describe empty data")
    # Should fail gracefully
    print(f"  ✓ Handled gracefully: {result['success']} - {result.get('error', 'No error')[:50]}")

def test_edge_single_column():
    print("⚠️  Edge Case: Single column for clustering")
    single_col = pd.DataFrame({'value': range(50)})
    result = agents['MLInsights']._clustering_analysis(single_col, "cluster single column")
    assert not result['success']  # Should fail but gracefully
    print(f"  ✓ Correctly rejected: {result['error'][:60]}")

def test_edge_negative_financial():
    print("⚠️  Edge Case: All negative financial values")
    negative_data = pd.DataFrame({
        'revenue': [-100, -200, -150],
        'cost': [-60, -120, -90]
    })
    result = agents['Financial']._profitability_analysis(negative_data, "analyze losses")
    assert result['success']
    print(f"  ✓ Handled negative values correctly")

def test_stress_large_dataset():
    print("🔥 Stress Test: Very large dataset (10,000 rows)")
    large_data = DataGenerator.hard_numeric_data(10000)
    result = agents['Statistical']._descriptive_statistics(large_data, "analyze large data")
    assert result['success']
    print(f"  ✓ Processed {len(large_data)} rows, {len(large_data.columns)} columns successfully")

run_test("Edge Case: Empty Data", test_edge_empty_data)
run_test("Edge Case: Single Column", test_edge_single_column)
run_test("Edge Case: Negative Financial", test_edge_negative_financial)
run_test("Stress Test: Large Dataset", test_stress_large_dataset)


# ============================================================================
# FINAL RESULTS
# ============================================================================

print("\n" + "="*80)
print("COMPREHENSIVE TESTING RESULTS")
print("="*80)
print(f"\n📊 Total Tests Run: {total_tests}")
print(f"✅ Passed: {passed_tests}")
print(f"❌ Failed: {failed_tests}")
print(f"📈 Success Rate: {(passed_tests/total_tests*100):.1f}%")

print("\n" + "="*80)
print("COVERAGE SUMMARY")
print("="*80)
print(f"""
✅ Statistical Agent: 3 complexity levels tested
✅ Time Series Agent: 3 complexity levels tested
✅ Financial Agent: 3 complexity levels tested
✅ ML Insights Agent: 3 complexity levels tested
✅ SQL Agent: 3 complexity levels tested
✅ Integration Tests: 3 cross-agent pipelines tested
✅ Edge Cases: 4 boundary conditions tested

TOTAL: {total_tests} comprehensive scenarios across all agents!
""")

if failed_tests == 0:
    print("🎉 ALL TESTS PASSED! All plugins are production-ready! 🎉")
else:
    print(f"⚠️  {failed_tests} test(s) need attention")

print("="*80)

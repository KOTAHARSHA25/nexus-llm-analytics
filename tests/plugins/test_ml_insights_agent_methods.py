"""
ML INSIGHTS AGENT - AGENT METHOD TESTS
Testing the 3 implemented ML analysis methods
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

print("="*80)
print("ML INSIGHTS AGENT - AGENT METHOD TESTS")
print("="*80)
print("Testing implemented machine learning methods\n")

try:
    from backend.plugins.ml_insights_agent import MLInsightsAgent
    print("✅ MLInsightsAgent imported successfully\n")
except ImportError as e:
    print(f"❌ FAILED to import MLInsightsAgent: {e}")
    sys.exit(1)

# Initialize agent
agent = MLInsightsAgent()
config = {}
agent.config = config
if not agent.initialize():
    print("❌ Agent initialization failed")
    sys.exit(1)

print("✅ Agent initialized successfully\n")

# TEST 1: Clustering Analysis
print("="*80)
print("TEST 1: Clustering Analysis (_clustering_analysis)")
print("="*80)

# Create data with natural clusters (3 groups)
np.random.seed(42)
cluster1 = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 50),
    'feature2': np.random.normal(0, 1, 50)
})
cluster2 = pd.DataFrame({
    'feature1': np.random.normal(5, 1, 50),
    'feature2': np.random.normal(5, 1, 50)
})
cluster3 = pd.DataFrame({
    'feature1': np.random.normal(10, 1, 50),
    'feature2': np.random.normal(0, 1, 50)
})
data = pd.concat([cluster1, cluster2, cluster3], ignore_index=True)

try:
    result = agent._clustering_analysis(data, "perform clustering analysis")
    
    assert result['success'], f"Clustering analysis should succeed, got error: {result.get('error', 'unknown')}"
    assert 'result' in result, "Should have result key"
    assert 'kmeans' in result['result'], "Should have K-means results"
    
    kmeans_results = result['result']['kmeans']
    print(f"✅ Clustering analysis completed")
    print(f"✅ Optimal clusters: {kmeans_results['optimal_clusters']}")
    print(f"✅ Silhouette score: {kmeans_results['silhouette_score']:.4f}")
    print(f"✅ Inertia: {kmeans_results['inertia']:.2f}")
    
    # Validate cluster detection
    optimal_k = kmeans_results['optimal_clusters']
    assert 2 <= optimal_k <= 5, f"Should find 2-5 clusters, found {optimal_k}"
    assert kmeans_results['silhouette_score'] > 0.3, "Should have decent separation"
    
    # Check cluster analysis
    cluster_analysis = kmeans_results['cluster_analysis']
    print(f"✅ Number of clusters analyzed: {len(cluster_analysis)}")
    
    for cluster_name, cluster_info in cluster_analysis.items():
        print(f"  - {cluster_name}: {cluster_info['size']} points ({cluster_info['percentage']:.1f}%)")
    
    print("\n✅ TEST 1 PASSED - Clustering analysis working correctly\n")
    
except Exception as e:
    print(f"❌ TEST 1 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# TEST 2: Anomaly Detection
print("="*80)
print("TEST 2: Anomaly Detection (_anomaly_detection)")
print("="*80)

# Create data with obvious outliers
np.random.seed(42)
normal_data = pd.DataFrame({
    'feature1': np.random.normal(50, 10, 100),
    'feature2': np.random.normal(100, 20, 100)
})
# Add 5 outliers
outliers = pd.DataFrame({
    'feature1': [150, 160, 10, 5, 170],
    'feature2': [300, 280, 10, 5, 320]
})
data = pd.concat([normal_data, outliers], ignore_index=True)

try:
    result = agent._anomaly_detection(data, "detect anomalies")
    
    assert result['success'], f"Anomaly detection should succeed, got error: {result.get('error', 'unknown')}"
    assert 'result' in result, "Should have result key"
    assert 'isolation_forest' in result['result'], "Should have Isolation Forest results"
    
    iso_forest = result['result']['isolation_forest']
    print(f"✅ Anomaly detection completed")
    print(f"✅ Anomalies detected: {iso_forest['n_anomalies']}")
    print(f"✅ Anomaly percentage: {iso_forest['anomaly_percentage']:.2f}%")
    print(f"✅ Anomaly indices: {len(iso_forest['anomaly_indices'])} detected")
    
    # Validate anomaly detection
    assert iso_forest['n_anomalies'] > 0, "Should detect some anomalies"
    assert iso_forest['n_anomalies'] <= 15, "Should not over-detect anomalies"
    
    if 'anomaly_scores' in iso_forest:
        scores = iso_forest['anomaly_scores']
        print(f"✅ Anomaly scores - mean: {scores['mean']:.4f}, std: {scores['std']:.4f}")
    
    print("\n✅ TEST 2 PASSED - Anomaly detection working correctly\n")
    
except Exception as e:
    print(f"❌ TEST 2 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# TEST 3: Dimensionality Reduction (PCA)
print("="*80)
print("TEST 3: Dimensionality Reduction (_dimensionality_reduction)")
print("="*80)

# Create high-dimensional data
np.random.seed(42)
n_samples = 100
n_features = 10

# Create correlated features
base_features = np.random.randn(n_samples, 3)
data = pd.DataFrame({
    f'feature_{i}': base_features[:, i % 3] + np.random.randn(n_samples) * 0.1
    for i in range(n_features)
})

try:
    result = agent._dimensionality_reduction(data, "reduce dimensionality")
    
    assert result['success'], f"Dimensionality reduction should succeed, got error: {result.get('error', 'unknown')}"
    assert 'result' in result, "Should have result key"
    assert 'pca' in result['result'], "Should have PCA results"
    
    pca_results = result['result']['pca']
    print(f"✅ Dimensionality reduction completed")
    print(f"✅ Original dimensions: {n_features}")
    print(f"✅ Reduced dimensions: {pca_results['n_components']}")
    print(f"✅ Explained variance: {pca_results['total_variance_explained']*100:.2f}%")
    print(f"✅ Components for 95% variance: {pca_results['components_for_95_percent']}")
    
    # Validate PCA
    assert pca_results['n_components'] <= n_features, "Components should not exceed features"
    assert pca_results['total_variance_explained'] > 0.5, "Should retain significant variance"
    
    # Check variance per component
    if 'explained_variance_ratio' in pca_results:
        variance_ratios = pca_results['explained_variance_ratio']
        print(f"✅ Variance by component: {[f'{v*100:.2f}%' for v in variance_ratios[:3]]}")
    
    print("\n✅ TEST 3 PASSED - Dimensionality reduction working correctly\n")
    
except Exception as e:
    print(f"❌ TEST 3 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# TEST 4: Clustering with Edge Cases
print("="*80)
print("TEST 4: Clustering Analysis - Insufficient Columns")
print("="*80)

# Create data with only 1 numeric column
data_single = pd.DataFrame({
    'feature1': np.random.randn(50)
})

try:
    result = agent._clustering_analysis(data_single, "cluster single feature")
    
    # Should fail gracefully
    assert not result['success'], "Should fail with insufficient columns"
    assert 'error' in result, "Should have error message"
    print(f"✅ Error message: {result['error']}")
    print(f"✅ Gracefully handled insufficient data")
    
    print("\n✅ TEST 4 PASSED - Edge case handling working correctly\n")
    
except Exception as e:
    print(f"❌ TEST 4 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# TEST 5: Anomaly Detection with Clean Data
print("="*80)
print("TEST 5: Anomaly Detection - Clean Data (Few Anomalies)")
print("="*80)

# Create very clean data
np.random.seed(42)
clean_data = pd.DataFrame({
    'feature1': np.random.normal(50, 5, 100),
    'feature2': np.random.normal(100, 10, 100)
})

try:
    result = agent._anomaly_detection(clean_data, "detect anomalies in clean data")
    
    assert result['success'], "Should succeed with clean data"
    
    iso_forest = result['result']['isolation_forest']
    print(f"✅ Anomalies detected in clean data: {iso_forest['n_anomalies']}")
    print(f"✅ Anomaly percentage: {iso_forest['anomaly_percentage']:.2f}%")
    
    # Should detect very few or no anomalies
    assert iso_forest['anomaly_percentage'] < 15, "Should detect few anomalies in clean data"
    
    print("\n✅ TEST 5 PASSED - Clean data handling working correctly\n")
    
except Exception as e:
    print(f"❌ TEST 5 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# TEST 6: PCA with Highly Correlated Features
print("="*80)
print("TEST 6: PCA - Highly Correlated Features")
print("="*80)

# Create highly correlated features (should reduce well)
np.random.seed(42)
base = np.random.randn(100)
data_corr = pd.DataFrame({
    'feature1': base,
    'feature2': base + np.random.randn(100) * 0.01,  # Almost identical
    'feature3': base + np.random.randn(100) * 0.01,
    'feature4': base + np.random.randn(100) * 0.01,
    'feature5': base + np.random.randn(100) * 0.01
})

try:
    result = agent._dimensionality_reduction(data_corr, "reduce highly correlated features")
    
    assert result['success'], "Should succeed with correlated features"
    
    pca_results = result['result']['pca']
    print(f"✅ Original: 5 features")
    print(f"✅ Reduced to: {pca_results['n_components']} components")
    print(f"✅ Variance retained: {pca_results['total_variance_explained']*100:.2f}%")
    print(f"✅ Components for 95% variance: {pca_results['components_for_95_percent']}")
    
    # Highly correlated features should compress well
    assert pca_results['components_for_95_percent'] <= 2, "Should compress to 1-2 components for 95% variance"
    assert pca_results['total_variance_explained'] > 0.95, "Should retain >95% variance"
    
    print("\n✅ TEST 6 PASSED - High correlation handling working correctly\n")
    
except Exception as e:
    print(f"❌ TEST 6 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# Summary
print("="*80)
print("ML INSIGHTS AGENT METHOD TESTS SUMMARY")
print("="*80)
print("\n✅ All 6/6 tests PASSED!\n")
print("Methods tested:")
print("  ✅ _clustering_analysis - K-means with elbow method verified")
print("  ✅ _anomaly_detection - Isolation Forest working correctly")
print("  ✅ _dimensionality_reduction - PCA with variance analysis")
print("  ✅ Edge case: Insufficient columns handled gracefully")
print("  ✅ Edge case: Clean data produces few anomalies")
print("  ✅ Edge case: Highly correlated features compress well")
print("\n" + "="*80)
print("✅ ML INSIGHTS AGENT: 3/3 IMPLEMENTED METHODS TESTED (100%)")
print("="*80)
print("\n🎉 All implemented ML methods are NOW TESTED!")
print("📝 Note: 4 placeholder methods not yet implemented:")
print("   - _classification_analysis")
print("   - _regression_analysis")
print("   - _association_analysis")
print("   - _feature_importance_analysis")

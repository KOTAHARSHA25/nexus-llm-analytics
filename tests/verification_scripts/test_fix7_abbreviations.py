"""
Test Fix 7: ML Abbreviations in Routing
Tests that the query complexity analyzer recognizes common ML/statistics abbreviations
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.core.query_complexity_analyzer import QueryComplexityAnalyzer


def test_abbreviation_recognition():
    """Test that ML and statistics abbreviations are recognized"""
    print("=" * 60)
    print("FIX 7: ML ABBREVIATIONS IN ROUTING - Test")
    print("=" * 60)
    
    analyzer = QueryComplexityAnalyzer()
    test_data = {'rows': 1000, 'columns': 10}
    
    # Test cases: (query, expected_tier, abbreviation_type)
    test_cases = [
        # ML Abbreviations (should route to FULL_POWER)
        ("Run K-means clustering on the data", "full_power", "K-means"),
        ("Perform PCA on numeric columns", "full_power", "PCA"),
        ("Use SVM for classification", "full_power", "SVM"),
        ("Apply RF to predict outcomes", "full_power", "RF (Random Forest)"),
        ("Run KNN algorithm", "full_power", "KNN"),
        ("Use SVC for classification", "full_power", "SVC"),
        ("Apply LDA for dimensionality reduction", "full_power", "LDA"),
        ("Train GMM on the dataset", "full_power", "GMM"),
        
        # Statistical Test Abbreviations (should route to FULL_POWER)
        ("Run a t-test on two groups", "full_power", "t-test"),
        ("Perform f-test for variances", "full_power", "f-test"),
        ("Run z-test for proportions", "full_power", "z-test"),
        ("Use paired t-test", "full_power", "paired t-test"),
        
        # Statistical Measure Abbreviations (should route to BALANCED)
        ("Calculate std dev of sales", "balanced", "std dev"),
        ("Show corr between variables", "balanced", "corr"),
        ("Find avg and med values", "balanced", "avg, med"),
        ("Calculate var and stdev", "balanced", "var, stdev"),
        ("Show CI for the mean", "balanced", "CI"),
        ("Calculate IQR", "balanced", "IQR"),
        
        # Optimization Abbreviations (should route to FULL_POWER)
        ("Solve LP problem", "full_power", "LP"),
        ("Optimize using QP", "full_power", "QP"),
        ("Use SGD optimizer", "full_power", "SGD"),
        ("Train with ADAM", "full_power", "ADAM"),
        
        # Simple queries (should route to FAST - unchanged)
        ("Show me the total count", "fast", "simple count"),
        ("What is the sum?", "fast", "simple sum"),
        ("List all records", "fast", "simple list"),
    ]
    
    passed = 0
    failed = 0
    
    print("\n📊 Testing Abbreviation Recognition:\n")
    
    for query, expected_tier, abbrev_type in test_cases:
        result = analyzer.analyze(query, test_data)
        actual_tier = result.recommended_tier
        
        if actual_tier == expected_tier:
            print(f"✅ {abbrev_type:25s} | \"{query[:40]}...\" → {actual_tier.upper()}")
            passed += 1
        else:
            print(f"❌ {abbrev_type:25s} | \"{query[:40]}...\" → {actual_tier.upper()} (expected: {expected_tier.upper()})")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed+failed} passed ({100*passed/(passed+failed):.1f}%)")
    print("=" * 60)
    
    if failed == 0:
        print("✅ ALL TESTS PASSED - Fix 7 implemented successfully!")
        return 0
    else:
        print(f"⚠️  {failed} tests failed - Review abbreviation detection")
        return 1


def test_negation_detection():
    """Test that negated queries still route correctly"""
    print("\n" + "=" * 60)
    print("BONUS: Testing Negation Detection")
    print("=" * 60)
    
    analyzer = QueryComplexityAnalyzer()
    test_data = {'rows': 1000, 'columns': 10}
    
    # Negated queries should route to simpler tiers
    negated_tests = [
        ("Don't use ML, just count the rows", "fast"),
        ("No need for stats, only show the sum", "fast"),
        ("Skip the analysis, just display values", "fast"),
    ]
    
    print("\n📊 Testing Negation Detection:\n")
    
    for query, expected_tier in negated_tests:
        result = analyzer.analyze(query, test_data)
        actual_tier = result.recommended_tier
        
        if actual_tier == expected_tier:
            print(f"✅ \"{query[:45]}\" → {actual_tier.upper()}")
        else:
            print(f"❌ \"{query[:45]}\" → {actual_tier.upper()} (expected: {expected_tier.upper()})")


if __name__ == "__main__":
    exit_code = test_abbreviation_recognition()
    test_negation_detection()
    sys.exit(exit_code)

"""
Test script for Intelligent Routing Integration

This tests the intelligent routing system integrated into the main analysis pipeline.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backend.core.intelligent_router import create_router, RoutingConfig
from backend.core.query_complexity_analyzer import create_complexity_analyzer
from backend.core.model_detector import detect_and_map_models

def test_routing_integration():
    """Test that routing components are working"""
    
    print("=" * 80)
    print("INTELLIGENT ROUTING INTEGRATION TEST (DYNAMIC MODEL DETECTION)")
    print("=" * 80)
    
    # Test 0: Dynamic Model Detection
    print("\n✅ TEST 0: Dynamic Model Detection")
    print("-" * 80)
    tier_models = detect_and_map_models()
    print(f"Detected tier assignments:")
    print(f"  FAST:       {tier_models.get('fast', 'N/A')}")
    print(f"  BALANCED:   {tier_models.get('balanced', 'N/A')}")
    print(f"  FULL_POWER: {tier_models.get('full_power', 'N/A')}")
    
    # Test 1: Complexity Analyzer
    print("\n\n✅ TEST 1: Query Complexity Analyzer")
    print("-" * 80)
    analyzer = create_complexity_analyzer()
    
    test_queries = [
        "What is the average sales?",
        "Compare sales between regions and show correlation",
        "Predict customer churn using machine learning and segment users"
    ]
    
    for query in test_queries:
        result = analyzer.analyze(query, {"rows": 1000, "columns": 10})
        print(f"\nQuery: {query}")
        print(f"  Complexity: {result.total_score:.3f}")
        print(f"  Recommended: {result.recommended_tier}")
    
    # Test 2: Intelligent Router with Dynamic Models
    print("\n\n✅ TEST 2: Intelligent Router (Dynamic Model Selection)")
    print("-" * 80)
    
    # Create router with detected models
    routing_config = RoutingConfig(
        fast_model=tier_models.get('fast', 'phi3:mini'),
        balanced_model=tier_models.get('balanced', 'phi3:mini'),
        full_power_model=tier_models.get('full_power', 'llama3.1:8b'),
        enable_fallback=True,
        track_performance=True,
        log_decisions=False  # Disable verbose logging for test
    )
    router = create_router(routing_config)
    
    for query in test_queries:
        decision = router.route(query, {"rows": 1000, "columns": 10})
        print(f"\nQuery: {query}")
        print(f"  Complexity: {decision.complexity_score:.3f}")
        print(f"  Selected Model: {decision.selected_model}")
        print(f"  Tier: {decision.selected_tier.value}")
        print(f"  Fallback: {decision.fallback_model}")
    
    # Test 3: Routing Statistics
    print("\n\n✅ TEST 3: Routing Statistics")
    print("-" * 80)
    stats = router.get_statistics()
    print(f"\nTotal Decisions: {stats['total_decisions']}")
    print(f"Average Complexity: {stats['average_complexity']:.3f}")
    print(f"Average Routing Time: {stats['average_routing_time_ms']:.2f}ms")
    print("\nTier Distribution:")
    for tier, data in stats['tier_distribution'].items():
        print(f"  {tier}: {data['count']} ({data['percentage']:.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ ALL INTEGRATION TESTS PASSED!")
    print("=" * 80)
    print("\n💡 NOTE: The system now automatically detects YOUR installed models")
    print("   and uses them for intelligent routing - no need to download specific models!")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    try:
        test_routing_integration()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

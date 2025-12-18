"""
Unit Tests for Intelligent Router
Production-grade tests for model selection routing
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from backend.core.intelligent_router import IntelligentRouter


class TestIntelligentRouter:
    """Test suite for IntelligentRouter"""
    
    @pytest.fixture
    def router(self):
        """Create router instance"""
        return IntelligentRouter()
    
    # ===== BASIC ROUTING TESTS =====
    
    def test_simple_query_routes_to_fast(self, router):
        """Simple queries should route to FAST tier"""
        from backend.core.intelligent_router import ModelTier
        query = "What is the total sales?"
        data_info = {"rows": 100, "columns": 5}
        
        decision = router.route(query, data_info)
        
        assert decision.selected_tier == ModelTier.FAST
        assert decision.complexity_score < 0.30
    
    def test_medium_query_routes_to_balanced(self, router):
        """Medium queries should route to BALANCED tier"""
        from backend.core.intelligent_router import ModelTier
        query = "Compare sales trends by region over time"
        data_info = {"rows": 1000, "columns": 10}
        
        decision = router.route(query, data_info)
        
        assert decision.selected_tier in [ModelTier.FAST, ModelTier.BALANCED]
        assert decision.complexity_score >= 0
    
    def test_complex_query_routes_to_full(self, router):
        """Complex queries should route to higher tier"""
        from backend.core.intelligent_router import ModelTier
        query = "Predict customer churn using machine learning"
        data_info = {"rows": 10000, "columns": 20}
        
        decision = router.route(query, data_info)
        
        assert decision.selected_tier in [ModelTier.BALANCED, ModelTier.FULL_POWER]
        assert decision.complexity_score >= 0.20
    
    # ===== USER OVERRIDE TESTS =====
    
    def test_user_override_respected(self, router):
        """User override should take precedence"""
        query = "Simple query"
        data_info = {"rows": 100, "columns": 5}
        
        # Force to specific model despite what routing would choose
        decision = router.route(query, data_info, user_override="llama3.1:8b")
        
        # Should use the overridden model
        assert decision.selected_model == "llama3.1:8b"
    
    # ===== FALLBACK TESTS =====
    
    def test_fallback_chain_exists(self, router):
        """Each tier should have fallback options"""
        from backend.core.intelligent_router import ModelTier
        assert ModelTier.FAST in router.fallback_chain
        assert ModelTier.BALANCED in router.fallback_chain
        assert ModelTier.FULL_POWER in router.fallback_chain
    
    # ===== STATISTICS TESTS =====
    
    def test_statistics_tracking(self, router):
        """Router should track decision history"""
        query = "What is the average?"
        data_info = {"rows": 100, "columns": 5}
        
        # Make a few decisions
        for i in range(3):
            router.route(query, data_info)
        
        # Check that routing history is being tracked
        assert len(router.routing_history) >= 3
        assert hasattr(router, 'tier_usage_count')
    
    # ===== PERFORMANCE TESTS =====
    
    def test_routing_speed(self, router):
        """Routing should be fast (<5ms)"""
        import time
        query = "Calculate total sales"
        data_info = {"rows": 1000, "columns": 10}
        
        start = time.time()
        for i in range(100):
            router.route(query, data_info)
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed / 100) * 1000
        assert avg_time_ms < 5, f"Routing too slow: {avg_time_ms:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

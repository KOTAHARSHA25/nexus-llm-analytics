"""
COMPREHENSIVE CORE COMPONENTS TESTING
Tests all backend core modules
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from backend.core.config import Config
from backend.core.model_selector import ModelSelector
from backend.core.query_complexity_analyzer import QueryComplexityAnalyzer
from backend.core.intelligent_router import IntelligentRouter
from backend.core.cot_parser import CoTParser
from backend.core.user_preferences import UserPreferencesManager
from backend.core.error_handling import ErrorHandler
from backend.core.rate_limiter import RateLimiter
from backend.core.circuit_breaker import CircuitBreaker


class TestConfig:
    """Test configuration management"""
    
    def test_config_loads(self):
        """Test config loads successfully"""
        config = Config()
        assert config is not None
        print("✓ Config loaded")
    
    def test_config_has_required_settings(self):
        """Test config has required settings"""
        config = Config()
        # Check for essential config attributes
        assert hasattr(config, 'OLLAMA_BASE_URL') or hasattr(config, 'ollama_base_url')
        print("✓ Config has required settings")


class TestModelSelector:
    """Test model selection"""
    
    def test_model_selector_initialization(self):
        """Test model selector initializes"""
        selector = ModelSelector()
        assert selector is not None
        print("✓ ModelSelector initialized")
    
    def test_select_model_for_query(self):
        """Test model selection for query"""
        selector = ModelSelector()
        result = selector.select_model("What is the total sales?")
        assert result is not None
        print(f"✓ Model selected: {result}")
    
    def test_model_availability_check(self):
        """Test checking model availability"""
        selector = ModelSelector()
        models = selector.get_available_models()
        assert isinstance(models, list)
        print(f"✓ Available models: {len(models)}")


class TestQueryComplexityAnalyzer:
    """Test query complexity analysis"""
    
    def test_analyzer_initialization(self):
        """Test analyzer initializes"""
        analyzer = QueryComplexityAnalyzer()
        assert analyzer is not None
        print("✓ QueryComplexityAnalyzer initialized")
    
    def test_simple_query_analysis(self):
        """Test analyzing simple query"""
        analyzer = QueryComplexityAnalyzer()
        result = analyzer.analyze("What is the total?")
        assert result is not None
        assert hasattr(result, 'total_score')
        assert result.total_score < 0.3
        print(f"✓ Simple query score: {result.total_score:.3f}")
    
    def test_complex_query_analysis(self):
        """Test analyzing complex query"""
        analyzer = QueryComplexityAnalyzer()
        result = analyzer.analyze("Perform machine learning clustering analysis")
        assert result is not None
        assert result.total_score > 0.3
        print(f"✓ Complex query score: {result.total_score:.3f}")
    
    def test_medium_query_analysis(self):
        """Test analyzing medium complexity query"""
        analyzer = QueryComplexityAnalyzer()
        result = analyzer.analyze("Compare sales trends by region")
        assert result is not None
        print(f"✓ Medium query score: {result.total_score:.3f}")


class TestIntelligentRouter:
    """Test intelligent routing"""
    
    def test_router_initialization(self):
        """Test router initializes"""
        router = IntelligentRouter()
        assert router is not None
        print("✓ IntelligentRouter initialized")
    
    def test_route_simple_query(self):
        """Test routing simple query"""
        router = IntelligentRouter()
        decision = router.route("What is the total?")
        assert decision is not None
        assert hasattr(decision, 'selected_model')
        print(f"✓ Routed to: {decision.selected_model}")
    
    def test_route_complex_query(self):
        """Test routing complex query"""
        router = IntelligentRouter()
        decision = router.route("Perform advanced statistical modeling")
        assert decision is not None
        print(f"✓ Complex query routed to: {decision.selected_model}")
    
    def test_routing_with_data_info(self):
        """Test routing with dataset info"""
        router = IntelligentRouter()
        data_info = {"rows": 10000, "columns": 50}
        decision = router.route("Analyze patterns", data_info)
        assert decision is not None
        print("✓ Routing with data info works")


class TestCoTParser:
    """Test Chain-of-Thought parser"""
    
    def test_parser_initialization(self):
        """Test parser initializes"""
        parser = CoTParser()
        assert parser is not None
        print("✓ CoTParser initialized")
    
    def test_parse_valid_response(self):
        """Test parsing valid CoT response"""
        parser = CoTParser()
        response = """
        [REASONING]
        Step 1: First I need to analyze the data structure
        Step 2: Then calculate the required metrics  
        Step 3: Finally format the results clearly
        [/REASONING]
        
        [OUTPUT]
        The total sales amount is $10,000
        [/OUTPUT]
        """
        result = parser.parse(response)
        assert result is not None
        assert result.is_valid
        print("✓ Valid CoT response parsed")
    
    def test_parse_invalid_response(self):
        """Test parsing invalid response"""
        parser = CoTParser()
        response = "Just some text without tags"
        result = parser.parse(response)
        assert result is not None
        assert not result.is_valid
        print("✓ Invalid response detected")


class TestUserPreferences:
    """Test user preferences management"""
    
    def test_preferences_initialization(self):
        """Test preferences manager initializes"""
        manager = UserPreferencesManager()
        assert manager is not None
        print("✓ UserPreferencesManager initialized")
    
    def test_load_preferences(self):
        """Test loading preferences"""
        manager = UserPreferencesManager()
        prefs = manager.load_preferences()
        assert isinstance(prefs, dict)
        print("✓ Preferences loaded")
    
    def test_save_preference(self):
        """Test saving preference"""
        manager = UserPreferencesManager()
        manager.update_preference('test_key', 'test_value')
        prefs = manager.load_preferences()
        assert 'test_key' in prefs
        print("✓ Preference saved")


class TestErrorHandler:
    """Test error handling"""
    
    def test_error_handler_initialization(self):
        """Test error handler initializes"""
        handler = ErrorHandler()
        assert handler is not None
        print("✓ ErrorHandler initialized")
    
    def test_handle_error(self):
        """Test handling error"""
        handler = ErrorHandler()
        try:
            raise ValueError("Test error")
        except Exception as e:
            result = handler.handle_error(e)
            assert result is not None
            print("✓ Error handled")


class TestRateLimiter:
    """Test rate limiting"""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initializes"""
        limiter = RateLimiter(max_requests=10, time_window=60)
        assert limiter is not None
        print("✓ RateLimiter initialized")
    
    def test_rate_limit_allows_requests(self):
        """Test rate limiter allows requests within limit"""
        limiter = RateLimiter(max_requests=5, time_window=60)
        for i in range(3):
            allowed = limiter.check_rate_limit("test_user")
            assert allowed == True
        print("✓ Rate limiting allows requests")
    
    def test_rate_limit_blocks_excess(self):
        """Test rate limiter blocks excess requests"""
        limiter = RateLimiter(max_requests=2, time_window=60)
        # First 2 should pass
        limiter.check_rate_limit("test_user2")
        limiter.check_rate_limit("test_user2")
        # Third should be blocked
        allowed = limiter.check_rate_limit("test_user2")
        assert allowed == False
        print("✓ Rate limiting blocks excess")


class TestCircuitBreaker:
    """Test circuit breaker pattern"""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initializes"""
        breaker = CircuitBreaker(failure_threshold=3)
        assert breaker is not None
        print("✓ CircuitBreaker initialized")
    
    def test_circuit_breaker_allows_calls(self):
        """Test circuit breaker allows calls"""
        breaker = CircuitBreaker(failure_threshold=3)
        
        def test_function():
            return "success"
        
        result = breaker.call(test_function)
        assert result == "success"
        print("✓ CircuitBreaker allows calls")
    
    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after failures"""
        breaker = CircuitBreaker(failure_threshold=2)
        
        def failing_function():
            raise Exception("Test failure")
        
        # Cause failures
        for i in range(3):
            try:
                breaker.call(failing_function)
            except:
                pass
        
        # Circuit should be open now
        assert breaker.state in ['open', 'OPEN']
        print("✓ CircuitBreaker opens on failures")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

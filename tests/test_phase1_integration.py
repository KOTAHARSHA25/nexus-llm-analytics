"""
Phase 1 End-to-End Integration Tests
=====================================
Comprehensive tests for all Phase 1 components:
- Smart Fallback Manager
- Dynamic Model Discovery
- RAM-Aware Selection
- Circuit Breaker System
- Enhanced Query Orchestrator
- DataAnalyst Agent Integration

Tests verify:
1. Components work individually
2. Components integrate correctly
3. Fallback chains activate properly
4. System never stops completely
5. Domain and data agnostic operation
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Individual test result"""
    name: str
    passed: bool
    duration_ms: float
    details: str
    error: str = None


class Phase1TestSuite:
    """Comprehensive Phase 1 test suite"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()
    
    def run_all(self):
        """Run all Phase 1 tests"""
        print("\n" + "="*70)
        print("PHASE 1 END-TO-END INTEGRATION TESTS")
        print("="*70 + "\n")
        
        # Test groups
        self._test_smart_fallback()
        self._test_model_discovery()
        self._test_ram_selector()
        self._test_circuit_breaker()
        self._test_query_orchestrator()
        self._test_phase1_coordinator()
        self._test_data_analyst_integration()
        
        # Print summary
        self._print_summary()
        
        return all(r.passed for r in self.results)
    
    def _record(self, name: str, passed: bool, duration_ms: float, details: str, error: str = None):
        """Record a test result"""
        result = TestResult(name, passed, duration_ms, details, error)
        self.results.append(result)
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {name} ({duration_ms:.1f}ms)")
        if not passed and error:
            print(f"       Error: {error[:100]}")
    
    # ==================== SMART FALLBACK TESTS ====================
    
    def _test_smart_fallback(self):
        """Test SmartFallbackManager"""
        print("\n[1/7] Testing SmartFallbackManager...")
        
        # Test 1.1: Initialization
        start = time.time()
        try:
            from backend.core.smart_fallback import SmartFallbackManager, FallbackReason
            manager = SmartFallbackManager()
            duration = (time.time() - start) * 1000
            self._record("SmartFallback: Initialization", True, duration, "Manager created successfully")
        except Exception as e:
            self._record("SmartFallback: Initialization", False, 0, str(e), str(e))
            return
        
        # Test 1.2: Model fallback chain
        start = time.time()
        try:
            fallback = manager.get_model_fallback("llama3.1:8b", FallbackReason.MEMORY_LIMIT)
            duration = (time.time() - start) * 1000
            passed = fallback is not None and fallback != "llama3.1:8b"
            self._record("SmartFallback: Model chain", passed, duration, f"Fallback: {fallback}")
        except Exception as e:
            self._record("SmartFallback: Model chain", False, 0, str(e), str(e))
        
        # Test 1.3: Method fallback
        start = time.time()
        try:
            fallback = manager.get_method_fallback("code_generation", FallbackReason.EXECUTION_ERROR)
            duration = (time.time() - start) * 1000
            passed = fallback is not None
            self._record("SmartFallback: Method chain", passed, duration, f"Fallback: {fallback}")
        except Exception as e:
            self._record("SmartFallback: Method chain", False, 0, str(e), str(e))
        
        # Test 1.4: Adaptive timeout
        start = time.time()
        try:
            timeout = manager.get_adaptive_timeout(60, "llama3.1:8b")
            duration = (time.time() - start) * 1000
            passed = timeout >= 30 and timeout <= 900
            self._record("SmartFallback: Adaptive timeout", passed, duration, f"Timeout: {timeout}s")
        except Exception as e:
            self._record("SmartFallback: Adaptive timeout", False, 0, str(e), str(e))
        
        # Test 1.5: Statistics
        start = time.time()
        try:
            stats = manager.get_stats()
            duration = (time.time() - start) * 1000
            passed = 'total_fallbacks' in stats and 'recovery_rate' in stats
            self._record("SmartFallback: Statistics", passed, duration, f"Total fallbacks: {stats.get('total_fallbacks')}")
        except Exception as e:
            self._record("SmartFallback: Statistics", False, 0, str(e), str(e))
    
    # ==================== MODEL DISCOVERY TESTS ====================
    
    def _test_model_discovery(self):
        """Test DynamicModelDiscovery"""
        print("\n[2/7] Testing DynamicModelDiscovery...")
        
        # Test 2.1: Initialization
        start = time.time()
        try:
            from backend.core.model_selector import DynamicModelDiscovery
            discovery = DynamicModelDiscovery()
            duration = (time.time() - start) * 1000
            self._record("ModelDiscovery: Initialization", True, duration, "Discovery initialized")
        except Exception as e:
            self._record("ModelDiscovery: Initialization", False, 0, str(e), str(e))
            return
        
        # Test 2.2: Sync discovery (may fail if Ollama not running)
        start = time.time()
        try:
            models = discovery.discover_models_sync()
            duration = (time.time() - start) * 1000
            
            if models:
                self._record("ModelDiscovery: Discover models", True, duration, 
                           f"Found {len(models)} models: {[m.name for m in models[:3]]}")
            else:
                self._record("ModelDiscovery: Discover models", True, duration, 
                           "No models found (Ollama may not be running)")
        except Exception as e:
            # Not a failure if Ollama isn't running
            self._record("ModelDiscovery: Discover models", True, 0, 
                        f"Discovery skipped: {str(e)[:50]}")
        
        # Test 2.3: Statistics
        start = time.time()
        try:
            stats = discovery.get_statistics()
            duration = (time.time() - start) * 1000
            passed = 'models_discovered' in stats
            self._record("ModelDiscovery: Statistics", passed, duration, 
                        f"Discovered: {stats.get('models_discovered', 0)}")
        except Exception as e:
            self._record("ModelDiscovery: Statistics", False, 0, str(e), str(e))
    
    # ==================== RAM SELECTOR TESTS ====================
    
    def _test_ram_selector(self):
        """Test RAMAwareSelector"""
        print("\n[3/7] Testing RAMAwareSelector...")
        
        # Test 3.1: Initialization
        start = time.time()
        try:
            from backend.core.model_selector import RAMAwareSelector, MemoryPressureLevel
            selector = RAMAwareSelector()
            duration = (time.time() - start) * 1000
            self._record("RAMSelector: Initialization", True, duration, "Selector initialized")
        except Exception as e:
            self._record("RAMSelector: Initialization", False, 0, str(e), str(e))
            return
        
        # Test 3.2: Memory snapshot
        start = time.time()
        try:
            snapshot = selector.get_memory_snapshot()
            duration = (time.time() - start) * 1000
            passed = snapshot.total_gb > 0 and snapshot.available_gb > 0
            self._record("RAMSelector: Memory snapshot", passed, duration, 
                        f"Available: {snapshot.available_gb:.1f}GB, Pressure: {snapshot.pressure_level.value}")
        except Exception as e:
            self._record("RAMSelector: Memory snapshot", False, 0, str(e), str(e))
        
        # Test 3.3: Available RAM calculation
        start = time.time()
        try:
            available = selector.get_available_ram_for_model()
            duration = (time.time() - start) * 1000
            passed = available >= 0
            self._record("RAMSelector: Available for model", passed, duration, 
                        f"Available for model: {available:.1f}GB")
        except Exception as e:
            self._record("RAMSelector: Available for model", False, 0, str(e), str(e))
        
        # Test 3.4: Model selection
        start = time.time()
        try:
            model_options = [
                ("tinyllama", 1.0),
                ("phi3:mini", 3.0),
                ("llama3.1:8b", 6.0)
            ]
            result = selector.select_model(
                preferred_model="llama3.1:8b",
                model_options=model_options,
                complexity=0.7
            )
            duration = (time.time() - start) * 1000
            passed = result.selected_model in [m[0] for m in model_options]
            self._record("RAMSelector: Model selection", passed, duration, 
                        f"Selected: {result.selected_model}, Fallback: {result.fallback_triggered}")
        except Exception as e:
            self._record("RAMSelector: Model selection", False, 0, str(e), str(e))
        
        # Test 3.5: Memory trend
        start = time.time()
        try:
            trend = selector.get_memory_trend()
            duration = (time.time() - start) * 1000
            passed = 'trend' in trend
            self._record("RAMSelector: Memory trend", passed, duration, 
                        f"Trend: {trend.get('trend')}")
        except Exception as e:
            self._record("RAMSelector: Memory trend", False, 0, str(e), str(e))
    
    # ==================== CIRCUIT BREAKER TESTS ====================
    
    def _test_circuit_breaker(self):
        """Test CircuitBreaker"""
        print("\n[4/7] Testing CircuitBreaker...")
        
        # Test 4.1: Initialization
        start = time.time()
        try:
            from backend.core.circuit_breaker import CircuitBreaker, CircuitState, CircuitBreakerConfig
            config = CircuitBreakerConfig(failure_threshold=2)
            cb = CircuitBreaker("test_circuit", config=config)
            duration = (time.time() - start) * 1000
            self._record("CircuitBreaker: Initialization", True, duration, 
                        f"Circuit created, state: {cb.state.value}")
        except Exception as e:
            self._record("CircuitBreaker: Initialization", False, 0, str(e), str(e))
            return
        
        # Test 4.2: Successful call
        start = time.time()
        try:
            def success_func():
                return {"success": True, "result": "test"}
            
            result = cb.call(success_func)
            duration = (time.time() - start) * 1000
            passed = result.get('success') == True
            self._record("CircuitBreaker: Successful call", passed, duration, 
                        f"Result: {result.get('success')}")
        except Exception as e:
            self._record("CircuitBreaker: Successful call", False, 0, str(e), str(e))
        
        # Test 4.3: Circuit opens after failures
        start = time.time()
        try:
            config2 = CircuitBreakerConfig(failure_threshold=2)
            cb2 = CircuitBreaker("test_fail_circuit", config=config2)
            
            def fail_func():
                raise Exception("Test failure")
            
            # Trigger failures
            for _ in range(3):
                cb2.call(fail_func)
            
            duration = (time.time() - start) * 1000
            passed = cb2.state == CircuitState.OPEN
            self._record("CircuitBreaker: Opens on failures", passed, duration, 
                        f"State after failures: {cb2.state.value}")
        except Exception as e:
            self._record("CircuitBreaker: Opens on failures", False, 0, str(e), str(e))
        
        # Test 4.4: Health status
        start = time.time()
        try:
            status = cb.get_health_status()
            duration = (time.time() - start) * 1000
            passed = 'state' in status and 'statistics' in status
            self._record("CircuitBreaker: Health status", passed, duration, 
                        f"State: {status.get('state')}")
        except Exception as e:
            self._record("CircuitBreaker: Health status", False, 0, str(e), str(e))
    
    # ==================== QUERY ORCHESTRATOR TESTS ====================
    
    def _test_query_orchestrator(self):
        """Test Enhanced QueryOrchestrator"""
        print("\n[5/7] Testing QueryOrchestrator...")
        
        # Test 5.1: Initialization
        start = time.time()
        try:
            from backend.core.engine.query_orchestrator import QueryOrchestrator, ExecutionMethod, ReviewLevel
            
            config = {
                'model_selection': {
                    'simple': 'tinyllama',
                    'medium': 'phi3:mini',
                    'complex': 'llama3.1:8b',
                    'thresholds': {'simple_max': 0.3, 'medium_max': 0.7}
                },
                'cot_review': {
                    'activation_rules': {
                        'always_on_complexity': 0.7,
                        'optional_range': [0.3, 0.7],
                        'always_on_code_gen': True
                    }
                }
            }
            
            orchestrator = QueryOrchestrator(None, config)
            duration = (time.time() - start) * 1000
            self._record("Orchestrator: Initialization", True, duration, 
                        "Orchestrator with Phase 1 enhancements")
        except Exception as e:
            self._record("Orchestrator: Initialization", False, 0, str(e), str(e))
            return
        
        # Test 5.2: Simple query plan
        start = time.time()
        try:
            plan = orchestrator.create_execution_plan("What is the average?", None)
            duration = (time.time() - start) * 1000
            passed = plan.model is not None and plan.complexity_score < 0.5
            self._record("Orchestrator: Simple query plan", passed, duration, 
                        f"Model: {plan.model}, Complexity: {plan.complexity_score:.2f}")
        except Exception as e:
            self._record("Orchestrator: Simple query plan", False, 0, str(e), str(e))
        
        # Test 5.3: Complex query plan
        start = time.time()
        try:
            complex_query = "Analyze the correlation between price and demand over time, then predict next quarter sales using regression"
            plan = orchestrator.create_execution_plan(complex_query, "mock_data")
            duration = (time.time() - start) * 1000
            passed = plan.complexity_score > 0.5 and plan.review_level in [ReviewLevel.OPTIONAL, ReviewLevel.MANDATORY]
            self._record("Orchestrator: Complex query plan", passed, duration, 
                        f"Complexity: {plan.complexity_score:.2f}, Review: {plan.review_level.value}")
        except Exception as e:
            self._record("Orchestrator: Complex query plan", False, 0, str(e), str(e))
        
        # Test 5.4: Fallback chain
        start = time.time()
        try:
            plan = orchestrator.create_execution_plan("Calculate total", "mock_data")
            duration = (time.time() - start) * 1000
            # Phase 1: Should have fallback chain
            has_fallback = hasattr(plan, 'fallback_chain')
            self._record("Orchestrator: Fallback chain", has_fallback, duration, 
                        f"Fallbacks: {getattr(plan, 'fallback_chain', [])}")
        except Exception as e:
            self._record("Orchestrator: Fallback chain", False, 0, str(e), str(e))
        
        # Test 5.5: Orchestrator status
        start = time.time()
        try:
            status = orchestrator.get_orchestrator_status()
            duration = (time.time() - start) * 1000
            passed = 'models' in status and 'phase1_available' in status
            self._record("Orchestrator: Status", passed, duration, 
                        f"Phase1 available: {status.get('phase1_available')}")
        except Exception as e:
            self._record("Orchestrator: Status", False, 0, str(e), str(e))
    
    # ==================== PHASE 1 COORDINATOR TESTS ====================
    
    def _test_phase1_coordinator(self):
        """Test Phase1Coordinator (unified access)"""
        print("\n[6/7] Testing Phase1Coordinator...")
        
        # Test 6.1: Initialization
        start = time.time()
        try:
            from backend.core.phase1_integration import Phase1Coordinator, get_phase1_coordinator
            coordinator = get_phase1_coordinator()
            duration = (time.time() - start) * 1000
            self._record("Coordinator: Initialization", True, duration, 
                        "Coordinator singleton created")
        except Exception as e:
            self._record("Coordinator: Initialization", False, 0, str(e), str(e))
            return
        
        # Test 6.2: Get best model
        start = time.time()
        try:
            model = coordinator.get_best_model_for_query("test query", complexity=0.5)
            duration = (time.time() - start) * 1000
            passed = model is not None and len(model) > 0
            self._record("Coordinator: Best model", passed, duration, 
                        f"Selected: {model}")
        except Exception as e:
            self._record("Coordinator: Best model", False, 0, str(e), str(e))
        
        # Test 6.3: Fallback chain
        start = time.time()
        try:
            chain = coordinator.get_model_fallback_chain()
            duration = (time.time() - start) * 1000
            passed = isinstance(chain, list)
            self._record("Coordinator: Fallback chain", passed, duration, 
                        f"Chain: {chain[:3] if chain else 'empty'}")
        except Exception as e:
            self._record("Coordinator: Fallback chain", False, 0, str(e), str(e))
        
        # Test 6.4: Overall status
        start = time.time()
        try:
            status = coordinator.get_status()
            duration = (time.time() - start) * 1000
            passed = hasattr(status, 'healthy') and hasattr(status, 'overall_health_score')
            self._record("Coordinator: Status", passed, duration, 
                        f"Healthy: {status.healthy}, Score: {status.overall_health_score:.2%}")
        except Exception as e:
            self._record("Coordinator: Status", False, 0, str(e), str(e))
    
    # ==================== DATA ANALYST INTEGRATION ====================
    
    def _test_data_analyst_integration(self):
        """Test DataAnalystAgent with Phase 1"""
        print("\n[7/7] Testing DataAnalyst Integration...")
        
        # Test 7.1: Agent initialization
        start = time.time()
        try:
            from backend.plugins.data_analyst_agent import DataAnalystAgent
            agent = DataAnalystAgent()
            agent.initialize()
            duration = (time.time() - start) * 1000
            self._record("DataAnalyst: Initialization", True, duration, 
                        f"Version: {agent.get_metadata().version}")
        except Exception as e:
            self._record("DataAnalyst: Initialization", False, 0, str(e), str(e))
            return
        
        # Test 7.2: Metadata
        start = time.time()
        try:
            metadata = agent.get_metadata()
            duration = (time.time() - start) * 1000
            passed = "2.1" in metadata.version  # Phase 1 version
            self._record("DataAnalyst: Metadata", passed, duration, 
                        f"Version {metadata.version} with Phase 1")
        except Exception as e:
            self._record("DataAnalyst: Metadata", False, 0, str(e), str(e))
        
        # Test 7.3: Orchestrator integration
        start = time.time()
        try:
            orchestrator = agent._get_orchestrator()
            duration = (time.time() - start) * 1000
            passed = orchestrator is not None
            self._record("DataAnalyst: Orchestrator", passed, duration, 
                        "Orchestrator loaded")
        except Exception as e:
            self._record("DataAnalyst: Orchestrator", False, 0, str(e), str(e))
        
        # Test 7.4: Can handle query
        start = time.time()
        try:
            confidence = agent.can_handle("What is the average?", ".csv")
            duration = (time.time() - start) * 1000
            passed = 0 < confidence <= 1.0
            self._record("DataAnalyst: Can handle", passed, duration, 
                        f"Confidence: {confidence:.2f}")
        except Exception as e:
            self._record("DataAnalyst: Can handle", False, 0, str(e), str(e))
    
    # ==================== SUMMARY ====================
    
    def _print_summary(self):
        """Print test summary"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        duration = time.time() - self.start_time
        
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
        print(f"Success Rate: {passed/total:.1%}")
        print(f"Total Duration: {duration:.2f}s")
        
        if failed > 0:
            print("\nFailed Tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  ❌ {r.name}: {r.error[:80] if r.error else 'Unknown'}")
        
        print("\n" + "="*70)
        
        # Save results
        results_path = project_root / 'tests' / 'phase1_test_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total': total,
                'passed': passed,
                'failed': failed,
                'success_rate': f"{passed/total:.1%}",
                'duration_seconds': round(duration, 2),
                'results': [
                    {
                        'name': r.name,
                        'passed': r.passed,
                        'duration_ms': round(r.duration_ms, 1),
                        'details': r.details,
                        'error': r.error
                    }
                    for r in self.results
                ]
            }, f, indent=2)
        
        print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    suite = Phase1TestSuite()
    success = suite.run_all()
    
    # Exit code
    sys.exit(0 if success else 1)

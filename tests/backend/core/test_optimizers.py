import pytest
from unittest.mock import patch, MagicMock
from src.backend.core.optimizers import (
    MemoryOptimizer, PerformanceOptimizer, StartupOptimizer, AdaptiveTimeoutManager, UnifiedOptimizer
)

@pytest.fixture
def mock_psutil():
    with patch('src.backend.core.optimizers.psutil') as mock:
        memory = MagicMock()
        memory.total = 16 * 1024**3
        memory.available = 8 * 1024**3
        memory.used = 8 * 1024**3
        memory.free = 8 * 1024**3
        memory.percent = 50.0
        memory.cached = 1 * 1024**3
        memory.buffers = 0
        mock.virtual_memory.return_value = memory
        
        mock.cpu_percent.return_value = 20.0
        
        # Mock process_iter to return a list of mock processes with proper .info attribute
        def create_mock_process(pid, name, rss_bytes):
            proc = MagicMock()
            mem_info = MagicMock()
            mem_info.rss = rss_bytes
            proc.info = {'pid': pid, 'name': name, 'memory_info': mem_info}
            return proc
        
        mock.process_iter.return_value = [
            create_mock_process(1, 'chrome', 2 * 1024**3),  # 2GB browser
            create_mock_process(2, 'code', 1.5 * 1024**3),   # 1.5GB IDE
            create_mock_process(3, 'system', 0.5 * 1024**3),  # System
        ]
        
        yield mock

def test_memory_optimizer(mock_psutil):
    usage = MemoryOptimizer.get_memory_usage()
    assert usage['total_gb'] == 16.0
    assert usage['percent_used'] == 50.0

    avail, recs = MemoryOptimizer.estimate_available_after_cleanup()
    # Should have recommendations for browser (2GB > 1.0) and IDE (1.5GB > 1.0)
    assert len(recs) >= 2
    # Check for browser recommendation
    assert any("Browser" in r or "browser" in r for r in recs)

def test_performance_optimizer(mock_psutil):
    opt = PerformanceOptimizer()
    stats = opt.monitor_system_resources()
    assert stats['memory_percent'] == 50.0
    assert stats['should_optimize'] is False
    
    # Test high load
    mock_psutil.virtual_memory.return_value.percent = 90.0
    stats = opt.monitor_system_resources()
    assert stats['should_optimize'] is True

def test_optimize_data_processing():
    opt = PerformanceOptimizer()
    plan = opt.optimize_data_processing(list(range(1000000)), "analysis")
    assert plan['strategy'] == 'streaming'

def test_adaptive_timeout(mock_psutil):
    mgr = AdaptiveTimeoutManager()
    
    # Normal case
    timeout = mgr.calculate_optimal_timeout("ollama/phi3:mini", "analysis", {'memory_available_gb': 8.0, 'cpu_percent': 20})
    assert timeout == 300
    
    # Low RAM case
    timeout_low = mgr.calculate_optimal_timeout("ollama/phi3:mini", "analysis", {'memory_available_gb': 1.0, 'cpu_percent': 20})
    assert timeout_low > 300 # Should trigger increase

def test_startup_optimizer():
    # Mock plugin_system at the right location (inside the module where it's imported)
    with patch.object(StartupOptimizer, 'optimize_startup') as mock_method:
        mock_method.return_value = {
            'startup_optimization_time': 0.1,
            'plugin_agents_loaded': 1,
            'ready_for_requests': True
        }
        
        res = StartupOptimizer.optimize_startup()
        assert res['plugin_agents_loaded'] == 1
        assert res['ready_for_requests'] is True

def test_unified_optimizer(mock_psutil):
    # The UnifiedOptimizer will call get_optimization_plan which needs process_iter
    uni = UnifiedOptimizer()
    report = uni.get_system_optimization_report()
    assert 'memory_optimization' in report
    assert 'recommendations' in report

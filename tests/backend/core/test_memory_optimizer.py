import pytest
from unittest.mock import patch, MagicMock
from src.backend.core.memory_optimizer import MemoryOptimizer

@pytest.fixture
def mock_psutil():
    with patch('src.backend.core.memory_optimizer.psutil') as mock:
        # Mock virtual_memory
        mem = MagicMock()
        mem.total = 16 * 1024**3
        mem.available = 8 * 1024**3
        mem.used = 8 * 1024**3
        mem.free = 8 * 1024**3
        mem.percent = 50.0
        mem.cached = 1 * 1024**3
        mem.buffers = 0
        mock.virtual_memory.return_value = mem
        
        # Mock process_iter
        proc = MagicMock()
        proc.info = {'pid': 123, 'name': 'chrome.exe', 'memory_info': MagicMock(rss=500*1024*1024)}
        mock.process_iter.return_value = [proc]
        
        yield mock

def test_get_memory_usage(mock_psutil):
    usage = MemoryOptimizer.get_memory_usage()
    assert usage['total_gb'] == 16.0
    assert usage['available_gb'] == 8.0
    assert 'cached_gb' in usage

def test_estimate_available_after_cleanup(mock_psutil):
    # Setup processes - need actual numeric rss values
    def make_proc(pid, name, rss_bytes):
        p = MagicMock()
        mem = MagicMock()
        mem.rss = int(rss_bytes)  # Must be actual int for comparison
        p.info = {'pid': pid, 'name': name, 'memory_info': mem}
        return p
    
    procs = [
        make_proc(1, 'chrome', 2 * 1024**3),     # 2GB browser
        make_proc(2, 'code', 1.5 * 1024**3),      # 1.5GB IDE
        make_proc(3, 'system', 1 * 1024**3),      # System (ignored)
    ]
    
    mock_psutil.process_iter.return_value = procs
    
    avail, recs = MemoryOptimizer.estimate_available_after_cleanup()
    
    assert avail > 8.0  # Should be higher than current available due to potential savings
    assert any("browser" in r.lower() for r in recs)
    assert any("ide" in r.lower() for r in recs)

def test_get_optimization_plan(mock_psutil):
    plan = MemoryOptimizer.get_optimization_plan()
    assert 'current_memory' in plan
    assert 'top_processes' in plan
    assert 'model_compatibility' in plan
    
    # Check model compatibility logic
    # With 8GB available, llama3.1:8b (needs 6.0) should be feasible
    assert plan['model_compatibility']['llama3.1:8b']['current'] is True

def test_clear_system_cache():
    with patch('subprocess.run') as mock_run:
        with patch('os.unlink') as mock_unlink:
            with patch('os.listdir', return_value=['temp1.tmp']):
                with patch('os.path.isfile', return_value=True):
                     with patch('os.environ.get', return_value='/tmp'):
                        with patch('os.path.exists', return_value=True):
                            MemoryOptimizer.clear_system_cache()
                            mock_run.assert_called() # flushdns

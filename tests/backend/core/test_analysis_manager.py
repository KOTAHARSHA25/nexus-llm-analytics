import pytest
import time
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
import threading
from src.backend.core.analysis_manager import AnalysisManager, check_cancellation, analysis_manager

# Fixture for fresh AnalysisManager instance
@pytest.fixture
def manager():
    return AnalysisManager()

def test_start_analysis(manager):
    user_session = "test_session"
    analysis_id = manager.start_analysis(user_session)
    
    assert isinstance(analysis_id, str)
    assert len(analysis_id) > 0
    
    status = manager.get_analysis_status(analysis_id)
    assert status['id'] == analysis_id
    assert status['user_session'] == user_session
    assert status['status'] == 'running'
    assert status['stage'] == 'initializing'
    assert 'start_time' in status

def test_update_analysis_stage(manager):
    aid = manager.start_analysis("sess")
    manager.update_analysis_stage(aid, "processing")
    
    status = manager.get_analysis_status(aid)
    assert status['stage'] == "processing"
    assert 'last_update' in status

    # Test updating non-existent analysis (should not crash)
    manager.update_analysis_stage("fake_id", "processing")

def test_cancellation_flow(manager):
    aid = manager.start_analysis("sess")
    
    # Initially not cancelled
    assert not manager.is_cancelled(aid)
    
    # Cancel it
    result = manager.cancel_analysis(aid)
    assert result is True
    assert manager.is_cancelled(aid)
    
    status = manager.get_analysis_status(aid)
    assert status['status'] == 'cancelled'
    assert 'cancelled_time' in status

    # Cancel non-existent
    assert manager.cancel_analysis("fake") is False

def test_complete_analysis(manager):
    aid = manager.start_analysis("sess")
    
    # Complete it
    manager.complete_analysis(aid)
    
    status = manager.get_analysis_status(aid)
    assert status['status'] == 'completed'
    assert 'end_time' in status
    assert not manager.is_cancelled(aid)

    # Test completion of cancelled analysis (edge case: should remove from cancelled set)
    aid2 = manager.start_analysis("sess2")
    manager.cancel_analysis(aid2)
    assert manager.is_cancelled(aid2)
    
    manager.complete_analysis(aid2)
    assert status['status'] == 'completed'
    assert not manager.is_cancelled(aid2)

def test_get_running_analyses(manager):
    aid1 = manager.start_analysis("s1")
    aid2 = manager.start_analysis("s2")
    manager.cancel_analysis(aid2)
    aid3 = manager.start_analysis("s3")
    manager.complete_analysis(aid3)
    
    running = manager.get_running_analyses()
    assert aid1 in running
    assert aid2 not in running
    assert aid3 not in running
    assert len(running) == 1

def test_cleanup_old_analyses(manager):
    # Mock time.time to control "current" time
    with patch('time.time') as mock_time:
        # Start time = 1000
        mock_time.return_value = 1000.0
        aid_old = manager.start_analysis("old")
        aid_new = manager.start_analysis("new")
        
        # Complete/Cancel them so they are candidates for cleanup
        manager.complete_analysis(aid_old) # end_time = 1000
        manager.complete_analysis(aid_new) # end_time = 1000
        
        # Move time forward by 25 hours (24h + 1h)
        # cleanup checks (current - start_time) > max_age
        # We want "old" to be cleaned up, "new" to stay? 
        # Wait, the logic is based on age from START time.
        
        # Let's rewrite setup for clarity
        
        # Case 1: Very old analysis, completed
        manager._running_analyses['old_completed'] = {
            'id': 'old_completed',
            'start_time': 1000.0,
            'status': 'completed'
        }
        
        # Case 2: Very old analysis, cancelled
        manager._running_analyses['old_cancelled'] = {
            'id': 'old_cancelled',
            'start_time': 1000.0,
            'status': 'cancelled'
        }
        manager._cancelled_analyses.add('old_cancelled')
        
        # Case 3: Recent analysis, completed
        manager._running_analyses['recent_completed'] = {
            'id': 'recent_completed',
            'start_time': 100000.0, # Large start time
            'status': 'completed'
        }
        
        # Case 4: Old analysis, BUT still running (should not be cleaned)
        manager._running_analyses['old_running'] = {
            'id': 'old_running',
            'start_time': 1000.0,
            'status': 'running'
        }
        
        # Set current time to 1000 + 25 hours * 3600 = 1000 + 90000 = 91000
        # Wait, if start time is 1000, and current is 91000, age is 90000 (25h).
        # Max age default 24h = 86400.
        
        mock_time.return_value = 1000.0 + (25 * 3600) + 10 # 25h+ past start
        
        manager.cleanup_old_analyses(max_age_hours=24)
        
        # Check results
        current_ids = manager._running_analyses.keys()
        assert 'old_completed' not in current_ids
        assert 'old_cancelled' not in current_ids
        assert 'old_cancelled' not in manager._cancelled_analyses
        
        assert 'recent_completed' in current_ids # 100000 > current, age is negative (future), so keep
        assert 'old_running' in current_ids # Status is running

def test_check_cancellation_helper():
    # Helper uses the global instance, so we mock method on that global instance
    # or we can mock the global instance itself in the module
    
    with patch('src.backend.core.analysis_manager.analysis_manager') as mock_global:
        mock_global.is_cancelled.return_value = False
        # Should not raise
        check_cancellation("some_id")
        
        mock_global.is_cancelled.return_value = True
        # Should raise 499
        with pytest.raises(HTTPException) as exc:
            check_cancellation("some_id")
        assert exc.value.status_code == 499

def test_thread_safety(manager):
    # Basic concurrent access test
    def worker():
        for _ in range(100):
            aid = manager.start_analysis("sess")
            manager.update_analysis_stage(aid, "stage2")
            manager.cancel_analysis(aid)
            
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Just ensure no exceptions and consistent state (e.g., cancelled set matches status)
    # This is a basic sanity check against race conditions corrupting dicts
    for aid, info in manager._running_analyses.items():
        if info['status'] == 'cancelled':
            assert aid in manager._cancelled_analyses
        else:
            assert aid not in manager._cancelled_analyses

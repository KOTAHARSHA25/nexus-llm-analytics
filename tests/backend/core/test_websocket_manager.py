"""
Tests for src/backend/core/websocket_manager.py

Tests MessageType enum and ConnectionManager class.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock


class TestMessageType:
    """Test MessageType enum"""
    
    def test_message_types(self):
        from src.backend.core.websocket_manager import MessageType
        
        assert MessageType.ANALYSIS_START.value == "analysis_start"
        assert MessageType.ANALYSIS_PROGRESS.value == "analysis_progress"
        assert MessageType.ANALYSIS_COMPLETE.value == "analysis_complete"
        assert MessageType.ANALYSIS_ERROR.value == "analysis_error"
        assert MessageType.HEARTBEAT.value == "heartbeat"
        assert MessageType.NOTIFICATION.value == "notification"
        assert MessageType.STATUS_UPDATE.value == "status_update"


class TestConnectionManager:
    """Test ConnectionManager class"""
    
    @pytest.fixture
    def manager(self):
        from src.backend.core.websocket_manager import ConnectionManager
        return ConnectionManager()
    
    def test_init(self, manager):
        assert manager.active_connections == {}
        assert manager.connection_metadata == {}
    
    def test_get_connection_count_empty(self, manager):
        assert manager.get_connection_count() == 0
    
    def test_get_connected_clients_empty(self, manager):
        assert manager.get_connected_clients() == []
    
    @pytest.mark.asyncio
    async def test_connect(self, manager):
        """Test connection registration"""
        mock_ws = AsyncMock()
        
        await manager.connect(mock_ws, "client1", {"user": "test"})
        
        assert "client1" in manager.active_connections
        assert manager.get_connection_count() == 1
    
    def test_disconnect(self, manager):
        """Test disconnection"""
        mock_ws = MagicMock()
        manager.active_connections["client1"] = mock_ws
        manager.connection_metadata["client1"] = {}
        
        manager.disconnect("client1")
        
        assert "client1" not in manager.active_connections
    
    def test_disconnect_nonexistent(self, manager):
        """Disconnecting non-existent client should not raise"""
        manager.disconnect("nonexistent")


class TestAnalysisProgressTracker:
    """Test AnalysisProgressTracker class"""
    
    @pytest.fixture
    def tracker(self):
        from src.backend.core.websocket_manager import AnalysisProgressTracker, ConnectionManager
        mock_manager = MagicMock(spec=ConnectionManager)
        mock_manager.send_personal_message = AsyncMock()
        return AnalysisProgressTracker(mock_manager)
    
    def test_init(self, tracker):
        assert tracker.connection_manager is not None
    
    @pytest.mark.asyncio
    async def test_start_analysis(self, tracker):
        """Test starting analysis tracking"""
        await tracker.start_analysis("analysis1", "client1", "test query")
        
        assert "analysis1" in tracker.active_analyses
        assert tracker.active_analyses["analysis1"]["client_id"] == "client1"
    
    @pytest.mark.asyncio
    async def test_update_progress(self, tracker):
        """Test updating analysis progress"""
        await tracker.start_analysis("analysis1", "client1", "test query")
        await tracker.update_progress("analysis1", 50, "Halfway done")
        
        tracker.connection_manager.send_personal_message.assert_called()
    
    @pytest.mark.asyncio
    async def test_complete_analysis(self, tracker):
        """Test completing analysis"""
        await tracker.start_analysis("analysis1", "client1", "test query")
        await tracker.complete_analysis("analysis1", {"result": "data"}, 1.5)
        
        assert "analysis1" not in tracker.active_analyses
    
    @pytest.mark.asyncio
    async def test_report_error(self, tracker):
        """Test reporting analysis error"""
        await tracker.start_analysis("analysis1", "client1", "test query")
        await tracker.report_error("analysis1", "Something went wrong")
        
        tracker.connection_manager.send_personal_message.assert_called()

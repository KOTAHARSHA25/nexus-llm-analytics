import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from backend.main import app
from backend.core.swarm import SwarmContext, SwarmEvent
from backend.services.analysis_service import AnalysisService

client = TestClient(app)

@pytest.fixture
def mock_swarm():
    swarm = SwarmContext()
    # Add some dummy data
    swarm.add_task("task_1", "Test Task", [], "DataAnalyst")
    swarm.update_task_status("task_1", "done", "DataAnalyst", result="Success")
    swarm.publish(SwarmEvent.INSIGHT_FOUND, "Analyst", {"summary": "Test Insight"})
    return swarm

@pytest.fixture
def mock_service(mock_swarm):
    from types import SimpleNamespace
    service = MagicMock(spec=AnalysisService)
    service.orchestrator = MagicMock()
    service.orchestrator.swarm_context = mock_swarm
    service.registry = MagicMock()
    
    # Use SimpleNamespace for metadata to act like a real object
    cap = SimpleNamespace(name="ANALYSIS")
    meta = SimpleNamespace(
        name="DataAnalyst",
        version="1.0",
        description="Test Agent",
        capabilities=[cap],
        priority=10
    )
    
    agent = MagicMock()
    agent.metadata = meta
    
    service.registry.agents = {
        "DataAnalyst": agent
    }
    return service

def test_get_agents(mock_service):
    with patch("backend.api.swarm.get_analysis_service", return_value=mock_service):
        response = client.get("/api/swarm/agents")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "DataAnalyst"
        assert "ANALYSIS" in data[0]["capabilities"]

def test_get_tasks(mock_service):
    with patch("backend.api.swarm.get_analysis_service", return_value=mock_service):
        response = client.get("/api/swarm/tasks")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "task_1"
        assert data[0]["status"] == "done"

def test_get_events(mock_service):
    with patch("backend.api.swarm.get_analysis_service", return_value=mock_service):
        response = client.get("/api/swarm/events")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert data[0]["type"] == "INSIGHT_FOUND"
        assert data[0]["content"]["summary"] == "Test Insight"

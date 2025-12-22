
import pytest
from unittest.mock import MagicMock, patch
from backend.plugins.visualizer_agent import VisualizerAgent

def test_visualizer_capability():
    """Test standard plugin capabilities"""
    agent = VisualizerAgent()
    meta = agent.get_metadata()
    assert meta.name == "Visualizer"
    
    # Test keywords
    assert agent.can_handle("Show me a plot of sales") > 0.5
    assert agent.can_handle("Draw a chart") > 0.5
    assert agent.can_handle("Calculate average") == 0.0

@patch("crewai.Crew")
def test_visualizer_execute(mock_crew_cls):
    """Test execution flow"""
    # Mock Crew instance
    mock_crew = MagicMock()
    mock_crew.kickoff.return_value = "import plotly.express as px\nfig = px.bar(...)"
    mock_crew_cls.return_value = mock_crew
    
    agent = VisualizerAgent()
    # Mock initializer (property usage in execute)
    agent.initializer = MagicMock()
    
    # Execute
    result = agent.execute("Make a bar chart", data=[{"a": 1, "b": 2}])
    
    assert result["success"] is True
    assert "import plotly" in result["result"]
    assert result["metadata"]["agent"] == "Visualizer"
    
    # Verify Crew was called
    mock_crew_cls.assert_called_once()
    mock_crew.kickoff.assert_called_once()

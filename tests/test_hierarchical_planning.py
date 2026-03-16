import pytest
from unittest.mock import MagicMock
from backend.core.engine.query_orchestrator import QueryOrchestrator

class TestHierarchicalPlanning:
    def test_decompose_query(self):
        orchestrator = QueryOrchestrator()
        
        # Mock LLM client
        mock_client = MagicMock()
        mock_response = {
            "response": """
            [
              {
                "id": "task1", 
                "description": "Task 1", 
                "agent": "AgentA", 
                "dependencies": []
              },
              {
                "id": "task2", 
                "description": "Task 2", 
                "agent": "AgentB", 
                "dependencies": ["task1"]
              }
            ]
            """
        }
        mock_client.generate.return_value = mock_response
        
        tasks = orchestrator.decompose_query_to_swarm("Execute complex query", llm_client=mock_client)
        
        assert len(tasks) == 2
        assert tasks[0]["id"] == "task1"
        assert tasks[1]["id"] == "task2"
        assert tasks[1]["dependencies"] == ["task1"]
        
        # Verify tasks added to SwarmContext
        assert "task1" in orchestrator.swarm_context._task_graph
        assert "task2" in orchestrator.swarm_context._task_graph

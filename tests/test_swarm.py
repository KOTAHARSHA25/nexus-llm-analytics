import pytest
from backend.core.swarm import SwarmContext, SwarmEvent, SwarmMessage

class TestSwarmContext:
    def test_initialization(self):
        swarm = SwarmContext()
        assert swarm._shared_memory == {}
        assert swarm._task_graph == {}
        assert swarm._message_history == []
        assert swarm._subscribers == {}

    def test_shared_memory(self):
        swarm = SwarmContext()
        swarm.write_shared("test_key", "test_value", "TestAgent")
        assert swarm.read_shared("test_key") == "test_value"
        assert swarm.read_shared("non_existent") is None

    def test_publish_subscribe(self):
        swarm = SwarmContext()
        
        received_messages = []
        def callback(msg):
            received_messages.append(msg)
            
        swarm.subscribe(SwarmEvent.INSIGHT_FOUND, callback)
        
        # Publish
        swarm.publish(SwarmEvent.INSIGHT_FOUND, "TestAgent", {"data": 123})
        
        assert len(received_messages) == 1
        assert received_messages[0].type == SwarmEvent.INSIGHT_FOUND
        assert received_messages[0].source_agent == "TestAgent"
        assert received_messages[0].content == {"data": 123}
        
    def test_task_management(self):
        swarm = SwarmContext()
        
        # Add task
        swarm.add_task("task1", "Do something", assigned_to="AgentA")
        assert "task1" in swarm._task_graph
        assert swarm._task_graph["task1"]["status"] == "pending"
        
        # Update task status
        swarm.update_task_status("task1", "done", "AgentA", result="Success")
        assert swarm._task_graph["task1"]["status"] == "done"
        
        # Add dependent task
        swarm.add_task("task2", "Do dependent thing", dependencies=["task1"])
        assert "task1" in swarm._task_graph["task2"]["dependencies"]
        
        # Check pending tasks
        pending = swarm.get_pending_tasks()
        assert len(pending) == 1
        assert pending[0]["id"] == "task2"

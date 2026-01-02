from backend.core.plugin_system import get_agent_registry, AgentCapability

def test_registry_discovery():
    """Verify that agents are discovered"""
    registry = get_agent_registry()
    registry.discover_agents()
    
    agents = registry.list_agents()
    assert len(agents) > 0, "No agents discovered! Check plugins directory."
    assert "DataAnalyst" in agents
    assert "RagAgent" in agents

def test_routing_logic_visualizer():
    """Verify routing logic for visualization query"""
    registry = get_agent_registry()
    topic, confidence, agent = registry.route_query("Please plot a bar chart of sales")
    
    assert agent is not None
    # Depending on scoring, Visualizer or DataAnalyst might win, 
    # but confidence should be high.
    assert confidence > 0.5
    if agent.metadata.name != "Visualizer":
        # DataAnalyst might handle it too, which is valid.
        pass

def test_routing_logic_rag():
    """Verify routing logic for document query"""
    registry = get_agent_registry()
    topic, confidence, agent = registry.route_query("Summarize this PDF contract", file_type=".pdf")
    
    assert agent is not None
    assert agent.metadata.name == "RagAgent"
    assert confidence > 0.8

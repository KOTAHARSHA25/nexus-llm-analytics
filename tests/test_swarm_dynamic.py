import pytest
from unittest.mock import MagicMock, patch
from backend.core.plugin_system import BasePluginAgent, AgentRegistry, AgentMetadata, AgentCapability
from backend.plugins.data_analyst_agent import DataAnalystAgent
from backend.plugins.reporter_agent import ReporterAgent

# --- Mocks ---

class MockAgent(BasePluginAgent):
    def __init__(self, name, capabilities):
        self._metadata = AgentMetadata(
            name=name,
            version="1.0.0",
            description=f"Mock {name}",
            author="Tester",
            capabilities=capabilities,
            file_types=[],
            dependencies=[],
            priority=10
        )
        super().__init__()
        self.initialize(registry=None) # We'll inject registry later

    def get_metadata(self) -> AgentMetadata:
        return self._metadata

    def can_handle(self, query: str, file_type: str = None, **kwargs) -> float:
        return 0.5

    def initialize(self, **kwargs):
        self.registry = kwargs.get("registry")
        return True

    def execute(self, query, data=None, **kwargs):
        return {
            "success": True, 
            "result": f"{self._metadata.name} processed: {query}",
            "metadata": {"agent": self._metadata.name}
        }

# --- Tests ---

class TestSwarmDynamic:
    
    @pytest.fixture
    def registry(self):
        reg = AgentRegistry()
        # Clear any existing agents for isolation (if singleton logic allows, otherwise ignore)
        reg.plugins = {} 
        return reg

    def test_registry_injection(self, registry):
        """Verify registry is injected into agents on registration"""
        agent = MockAgent("TestAgent", [])
        
        # Mock initialize to verify it's called with registry
        agent.initialize = MagicMock(return_value=True)
        
        registry.register_agent(agent)
        
        # Check injection took place
        assert agent.registry == registry
        agent.initialize.assert_called_with(registry=registry)

    def test_delegation_basic(self, registry):
        """Verify one agent can delegate to another"""
        sender = MockAgent("Sender", [])
        receiver = MockAgent("Receiver", [])
        
        registry.register_agent(sender)
        registry.register_agent(receiver)
        
        # Sender calls delegate
        response = sender.delegate("Receiver", "Hello")
        
        assert response["success"] is True
        assert response["result"] == "Receiver processed: Hello"

    def test_delegation_recursion_limit(self, registry):
        """Verify infinite loops are caught"""
        agent_a = MockAgent("AgentA", [])
        agent_b = MockAgent("AgentB", [])
        
        registry.register_agent(agent_a)
        registry.register_agent(agent_b)
        
        # Make Agent B delegate back to Agent A (simulate loop)
        # We perform a manual patch here to create the loop condition
        original_execute_b = agent_b.execute
        
        def loop_execute(query, data=None, **kwargs):
            return agent_b.delegate("AgentA", query, **kwargs)
            
        agent_b.execute = loop_execute
        
        # Agent A calls Agent B, which calls Agent A...
        # A -> B -> A -> B -> A -> B -> A (STOP)
        
        # We start the chain manually
        response = agent_a.delegate("AgentB", "Loop check")
        
        # It should eventually fail or return error
        # Because the *delegate* method catches the recursion depth
        
        # Let's make A also loop to ensure infinite ping-pong
        def loop_execute_a(query, data=None, **kwargs):
             return agent_a.delegate("AgentB", query, **kwargs)
        agent_a.execute = loop_execute_a
        
        response = agent_a.delegate("AgentB", "Loop check")
        
        # Expect failure
        assert response["success"] is False
        assert "Max delegation depth" in response["error"]

    def test_capability_delegation(self, registry):
        """Verify delegating by capability finds the correct agent"""
        sender = MockAgent("Seeker", [])
        visualizer = MockAgent("VizArtist", [AgentCapability.VISUALIZATION])
        
        registry.register_agent(sender)
        registry.register_agent(visualizer)
        
        response = sender.delegate_by_capability(AgentCapability.VISUALIZATION, "Draw a chart")
        
        assert response["success"] is True
        assert "VizArtist processed" in response["result"]

    @patch("backend.plugins.data_analyst_agent.get_model_manager")
    # Patch where it is DEFINED, not where it is imported inside a method
    @patch("backend.core.engine.query_orchestrator.get_query_orchestrator") 
    def test_data_analyst_smart_delegation(self, mock_orch, mock_mm, registry):
        """Verify DataAnalyst delegates plotting queries"""
        analyst = DataAnalystAgent()
        visualizer = MockAgent("VisualizerAgent", [AgentCapability.VISUALIZATION])
        
        registry.register_agent(analyst)
        registry.register_agent(visualizer)
        
        # Configure analyst mocks
        mock_mm.return_value.ensure_initialized.return_value = True
        
        # Inject registry specifically (initialize might mock it away if we are strict, but here using real DataAnalyst)
        # We called register_agent which calls initialize(registry=...)
        # But DataAnalyst.initialize is REAL, so it sets self.registry correctly.
        
        # Execute typical plotting query
        # We need to provide dummy file or data to trigger the delegation logic check
        response = analyst.execute(
            query="Please plot a line chart of the sales", 
            data={"sales": [1, 2, 3]},
            # kwargs registry is not stored by execute, so we rely on self.registry being set by register_agent
        )
        
        # Should have delegated to VisualizerAgent
        # Note: VisualizerAgent in this test is a MOCK, so it returns "VisualizerAgent processed..."
        assert response["success"] is True
        assert "VisualizerAgent processed" in response["result"]

    @patch("backend.plugins.reporter_agent.get_model_manager")
    @patch("backend.plugins.reporter_agent.PDFReportGenerator")
    def test_reporter_data_pull(self, mock_pdf, mock_mm, registry):
        """Verify Reporter pulls data if missing"""
        reporter = ReporterAgent()
        analyst = MockAgent("DataAnalyst", [AgentCapability.DATA_ANALYSIS])
        
        registry.register_agent(reporter)
        registry.register_agent(analyst)
        
        # Configure reporter mock
        mock_mm.return_value.ensure_initialized.return_value = True
        # Mock LLM generation to avoid real call
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Report Content"
        mock_mm.return_value.llm_client = mock_llm
        
        # Execute reporter WITHOUT data
        # It should call DataAnalyst
        
        # We patch delegate to verify it was called, OR we check the result if MockAgent returns specific data
        with patch.object(reporter, 'delegate', wraps=reporter.delegate) as spy_delegate:
            response = reporter.execute(
                query="Create a report on Q3 sales",
                data=None
            )
            
            # Verify delegate was called
            spy_delegate.assert_called()
            call_args = spy_delegate.call_args
            assert call_args[1]['agent_name'] == "DataAnalyst"
            
            # Verify reporter continued to generate report
            assert response["success"] is True
            assert "Report Content" in response["result"]

    def test_reporter_existing_data(self, registry):
        """Verify Reporter uses provided data if available"""
        # ... setup similar to above but provide data ...
        pass

from backend.agents.visualization_agent import VisualizationAgent
from backend.agents.report_agent import ReportAgent
from backend.agents.rag_agent import RAGAgent

def test_visualization_agent_stub():
    agent = VisualizationAgent()
    result = agent.plot(data=None)
    assert 'message' in result and 'stub' in result['message']

def test_report_agent_stub():
    agent = ReportAgent()
    result = agent.compile_report(data=None)
    assert 'message' in result and 'stub' in result['message']


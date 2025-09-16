from backend.agents.data_agent import DataAgent
from backend.agents.review_agent import ReviewAgent
from backend.agents.rag_agent import RAGAgent
from backend.core.sandbox import Sandbox
from backend.core.utils import AgentRegistry, friendly_error
from backend.core.llm_client import LLMClient
import os

# CrewAI Controller Agent: Orchestrates and decomposes tasks



class ControllerAgent:
    """Primary orchestrator. Receives user queries and delegates sub-tasks to specialized agents."""
    def __init__(self, agent_config=None):
        self.registry = AgentRegistry()
        # Register default agents
        self.registry.register('review_agent', ReviewAgent())
        self.registry.register('sandbox', Sandbox())
        llm_client = LLMClient()
        data_agent = DataAgent(
            review_agent=self.registry.get('review_agent'),
            sandbox=self.registry.get('sandbox'),
            llm_client=llm_client
        )
        self.registry.register('data_agent', data_agent)
        persist_directory = os.getenv("CHROMADB_PERSIST_DIRECTORY", "./chroma_db")
        self.registry.register('rag_agent', RAGAgent(persist_directory=persist_directory, llm_client=llm_client))
        # Optionally register/override agents from config
        if agent_config:
            for name, agent in agent_config.items():
                self.registry.register(name, agent)

        # Scalable query routing
        self.query_handlers = {
            "summarize": data_agent.summarize,
            "describe": data_agent.describe,
            "value_counts": data_agent.value_counts,
            "filter": data_agent.filter,
        }

    def handle_query(self, query, filename=None, **kwargs):
        data_agent = self.registry.get('data_agent')
        rag_agent = self.registry.get('rag_agent')
        if not data_agent or not rag_agent:
            return friendly_error("Required agents not registered.", suggestion="Check agent configuration or contact support.")
        if not filename:
            return friendly_error("No file provided.", suggestion="Please upload or specify a data file.")

        # Route to RAG agent for unstructured data (PDF/TXT) or explicit 'rag' query
        if filename.lower().endswith(('.pdf', '.txt')) or query == "rag":
            # For RAG, just pass the query string
            return rag_agent.retrieve(query, n_results=kwargs.get('n_results', 3))

        # Otherwise, use DataAgent for structured data
        load_result = data_agent.load_file(filename)
        if isinstance(load_result, dict) and load_result.get("error"):
            return friendly_error(load_result.get("error"), suggestion="Check the file name and format.")

        # Query routing for structured data
        handler = self.query_handlers.get(query)
        if not handler:
            return friendly_error(f"Unknown query: {query}", suggestion="Check the query type or refer to the documentation.")

        # Argument validation and dispatch
        try:
            if query == "value_counts":
                if "column" not in kwargs:
                    return friendly_error("'column' parameter required for value_counts.", suggestion="Specify the column name to analyze value counts.")
                return handler(kwargs["column"])
            elif query == "filter":
                if "column" not in kwargs or "value" not in kwargs:
                    return friendly_error("'column' and 'value' parameters required for filter.", suggestion="Provide both column and value for filtering.")
                return handler(kwargs["column"], kwargs["value"])
            else:
                return handler()
        except Exception as e:
            return friendly_error(f"Error executing query '{query}': {e}", suggestion="Check your parameters and try again.")

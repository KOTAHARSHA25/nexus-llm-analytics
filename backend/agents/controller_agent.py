from backend.agents.data_agent import DataAgent

# CrewAI Controller Agent: Orchestrates and decomposes tasks

class ControllerAgent:
    """Primary orchestrator. Receives user queries and delegates sub-tasks to specialized agents."""
    def __init__(self):
        self.data_agent = DataAgent()

    def handle_query(self, query, filename=None, **kwargs):
        if filename:
            load_result = self.data_agent.load_file(filename)
            if isinstance(load_result, dict) and load_result.get("error"):
                return load_result
        else:
            return {"error": "No file provided."}

        # Query routing
        if query == "summarize":
            return self.data_agent.summarize()
        elif query == "describe":
            return self.data_agent.describe()
        elif query == "value_counts":
            column = kwargs.get("column")
            if not column:
                return {"error": "'column' parameter required for value_counts."}
            return self.data_agent.value_counts(column)
        elif query == "filter":
            column = kwargs.get("column")
            value = kwargs.get("value")
            if not column or value is None:
                return {"error": "'column' and 'value' parameters required for filter."}
            return self.data_agent.filter(column, value)
        else:
            return {"error": f"Unknown query: {query}"}

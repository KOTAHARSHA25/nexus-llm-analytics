# CrewAI Visualization Agent: Generates interactive charts using Plotly

class VisualizationAgent:
    """Agent for generating data visualizations."""
    def __init__(self, llm_client=None):
        from backend.core.llm_client import LLMClient
        self.llm_client = llm_client or LLMClient()

    def plot(self, data, chart_type="bar", **kwargs):
        """
        Generate Plotly chart code using LLM (Llama 3.1 8B) from user data and chart type.
        Returns dict with 'code' (Plotly code as string) and 'summary' (text summary).
        """
        prompt = (
            f"You are a Python data visualization expert. Given the following pandas DataFrame (as CSV) and a chart type, generate Python code using Plotly Express to create the chart. "
            f"Also provide a one-sentence summary of the chart.\n\n"
            f"Data (CSV):\n{data.to_csv(index=False)}\n\n"
            f"Chart type: {chart_type}\n"
            f"Respond in JSON: {{'code': <plotly_code>, 'summary': <summary>}}"
        )
        llm_result = self.llm_client.generate_primary(prompt)
        import json
        try:
            llm_json = json.loads(llm_result.get("response", ""))
            return {
                "code": llm_json.get("code", ""),
                "summary": llm_json.get("summary", "")
            }
        except Exception as e:
            return {
                "code": None,
                "summary": f"LLM output could not be parsed: {e}"
            }

# CrewAI Report Agent: Compiles outputs into downloadable reports

class ReportAgent:
    """Agent for compiling analysis and results into reports."""
    def __init__(self, llm_client=None):
        from backend.core.llm_client import LLMClient
        self.llm_client = llm_client or LLMClient()

    def compile_report(self, data, charts=None, summary=None):
        """
        Generate a report text and summary using LLM (Llama 3.1 8B) from data, charts, and summary.
        Returns dict with 'report_text' and 'summary'.
        """
        charts = charts or []
        summary = summary or ""
        prompt = (
            f"You are a data analyst. Given the following pandas DataFrame (as CSV), a list of chart descriptions, and a summary, write a professional report. "
            f"The report should include an executive summary, key findings, and recommendations.\n\n"
            f"Data (CSV):\n{data.to_csv(index=False)}\n\n"
            f"Charts: {charts}\n"
            f"Summary: {summary}\n"
            f"Respond in JSON: {{'report_text': <full_report>, 'summary': <one_sentence_summary>}}"
        )
        llm_result = self.llm_client.generate_primary(prompt)
        import json
        try:
            llm_json = json.loads(llm_result.get("response", ""))
            return {
                "report_text": llm_json.get("report_text", ""),
                "summary": llm_json.get("summary", "")
            }
        except Exception as e:
            return {
                "report_text": None,
                "summary": f"LLM output could not be parsed: {e}"
            }

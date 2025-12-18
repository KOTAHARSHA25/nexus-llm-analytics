# CrewAI Specialized Agents Module
# Consolidated module containing RAG, Visualization, and Report agents for better maintainability

from backend.core.chromadb_client import ChromaDBClient
from backend.core.llm_client import LLMClient
import json


class RAGAgent:
    """Retrieval-Augmented Generation agent for unstructured data."""
    
    def __init__(self, persist_directory="./test_chroma_db", llm_client=None):
        self.chroma = ChromaDBClient(persist_directory=persist_directory)
        self.llm_client = llm_client or LLMClient()

    def retrieve(self, query, n_results=3):
        """Retrieve top matching chunks from ChromaDB for a given query, then use LLM to summarize."""
        results = self.chroma.query(query_text=query, n_results=n_results)
        docs = results.get('documents', [[]])[0]
        ids = results.get('ids', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0] if 'metadatas' in results else [{}]*len(docs)
        
        # Use LLM to summarize the retrieved docs
        context = "\n\n".join(docs)
        prompt = f"Given the following context from a document, answer the user's query: '{query}'.\n\nContext:\n{context}"
        llm_result = self.llm_client.generate_primary(prompt)
        summary = llm_result.get("response")
        
        return {
            "query": query,
            "summary": summary,
            "chunks": [
                {"id": id_, "text": doc, "metadata": meta}
                for id_, doc, meta in zip(ids, docs, metadatas)
            ]
        }


class VisualizationAgent:
    """Agent for generating data visualizations."""
    
    def __init__(self, llm_client=None):
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


class ReportAgent:
    """Agent for compiling analysis and results into reports."""
    
    def __init__(self, llm_client=None):
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


# For backward compatibility, export individual agents
__all__ = ['RAGAgent', 'VisualizationAgent', 'ReportAgent']
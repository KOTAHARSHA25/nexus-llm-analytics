# CrewAI Base Infrastructure for Nexus LLM Analytics
# This module provides the foundation for proper CrewAI integration

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from langchain_ollama import OllamaLLM
from typing import Any
import logging

def create_base_llm(model: str = None) -> OllamaLLM:
    """Create a base LLM instance using langchain-ollama"""
    if model is None:
        model = "llama3.1:8b"
    
    # Remove ollama/ prefix if present (OllamaLLM doesn't need it)
    clean_model = model.replace("ollama/", "")
    
    return OllamaLLM(
        model=clean_model,
        base_url="http://localhost:11434"
    )

class DataAnalysisTool(BaseTool):
    """Tool for data analysis operations"""
    
    name: str = "data_analysis"
    description: str = "Analyze structured data files (CSV, JSON) and perform statistical operations"
    sandbox: Any = None
    
    def __init__(self, sandbox, **kwargs):
        super().__init__(sandbox=sandbox, **kwargs)
    
    def _run(self, code: str, data: Any = None, **kwargs) -> str:
        """Execute data analysis code in sandbox"""
        try:
            result = self.sandbox.execute(code, data=data)
            if "error" in result:
                return f"Error: {result['error']}"
            return str(result.get("result", ""))
        except Exception as e:
            return f"Tool execution failed: {str(e)}"

class RAGTool(BaseTool):
    """Tool for Retrieval-Augmented Generation"""
    
    name: str = "rag_retrieval"
    description: str = "Retrieve and summarize information from unstructured documents using vector similarity"
    chroma_client: Any = None
    llm_client: Any = None
    
    def __init__(self, chroma_client, llm_client, **kwargs):
        super().__init__(chroma_client=chroma_client, llm_client=llm_client, **kwargs)
    
    def _run(self, query: str, n_results: int = 3, **kwargs) -> str:
        """Perform RAG retrieval and generation"""
        try:
            results = self.chroma_client.query(query_text=query, n_results=n_results)
            docs = results.get('documents', [[]])[0]
            
            if not docs:
                return "No relevant documents found."
            
            context = "\n\n".join(docs)
            prompt = f"Based on this context, answer the query: '{query}'\n\nContext:\n{context}"
            
            response = self.llm_client.generate_primary(prompt)
            return response.get("response", "No response generated")
        except Exception as e:
            return f"RAG tool failed: {str(e)}"

class VisualizationTool(BaseTool):
    """Tool for generating data visualizations"""
    
    name: str = "visualization"
    description: str = "Generate Plotly visualizations from data analysis results"
    llm_client: Any = None
    
    def __init__(self, llm_client, **kwargs):
        super().__init__(llm_client=llm_client, **kwargs)
    
    def _run(self, data_summary: str, chart_type: str = "auto", **kwargs) -> str:
        """Generate visualization code"""
        try:
            prompt = f"""Generate Python Plotly code for visualization:
Data Summary: {data_summary}
Chart Type: {chart_type}

Return only the Python code that creates a Plotly figure."""
            
            response = self.llm_client.generate_primary(prompt)
            return response.get("response", "No visualization generated")
        except Exception as e:
            return f"Visualization tool failed: {str(e)}"

# create_base_llm function is defined above

def create_analysis_tools(sandbox, chroma_client, llm_client):
    """Factory function to create all analysis tools"""
    return [
        DataAnalysisTool(sandbox=sandbox),
        RAGTool(chroma_client=chroma_client, llm_client=llm_client),
        VisualizationTool(llm_client=llm_client)
    ]
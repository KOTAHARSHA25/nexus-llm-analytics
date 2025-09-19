# CrewAI Base Infrastructure for Nexus LLM Analytics
# This module provides the foundation for proper CrewAI integration

from crewai.tools import BaseTool
from crewai import LLM
from typing import Any
import os

def create_base_llm(model: str = None) -> LLM:
    """Create a base LLM instance using CrewAI's LLM class with LiteLLM"""
    if model is None:
        model = "llama3.1:8b"
    
    # Add ollama/ prefix if not present (LiteLLM needs provider specification)
    if not model.startswith("ollama/"):
        model = f"ollama/{model}"
    
    # Set required environment variables for LiteLLM
    os.environ["OPENAI_API_KEY"] = "not-needed"
    os.environ["OPENAI_API_BASE"] = "http://localhost:11434"
    
    # Use CrewAI's LLM class with extended timeout configuration
    return LLM(
        model=model,
        base_url="http://localhost:11434",
        timeout=1200,  # 20 minutes timeout for complex analysis
        max_retries=3
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
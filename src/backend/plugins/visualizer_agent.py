# Visualizer Agent Plugin
# Handles data visualization tasks

import sys
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from backend.core.plugin_system import BasePluginAgent, AgentMetadata, AgentCapability
from backend.agents.model_initializer import get_model_initializer

class VisualizerAgent(BasePluginAgent):
    """
    Visualizer Agent Plugin.
    Generates Plotly/Matplotlib code for data visualization.
    """
    
    def get_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="Visualizer",
            version="1.0.0",
            description="Generates interactive data visualizations (Plotly)",
            author="Nexus Team",
            capabilities=[AgentCapability.VISUALIZATION],
            file_types=[], # Can handle any data as input, doesn't load files directly usually
            dependencies=["plotly"],
            priority=20
        )
    
    def initialize(self, **kwargs) -> bool:
        self.initializer = get_model_initializer()
        return True
    
    def can_handle(self, query: str, file_type: Optional[str] = None, **kwargs) -> float:
        # Check for visualization keywords
        keywords = ["plot", "chart", "graph", "visualize", "diagram", "trend line"]
        if any(k in query.lower() for k in keywords):
            return 0.8
        return 0.0

    def execute(self, query: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """Execute visualization task"""
        try:
            self.initializer.ensure_initialized()
            
            # Construct data context
            data_context = ""
            if data:
                # If data is a dataframe, give a summary
                if hasattr(data, "head"):
                    data_context = f"Data Sample:\n{data.head().to_markdown()}\nColumns: {list(data.columns)}\nTypes: {data.dtypes.to_dict()}"
                else:
                    data_context = str(data)[:2000]

            system_prompt = """You are a Data Visualization Expert using Python and Plotly Express.
Your goal is to write a single, self-contained Python code block that generates an interactive Plotly chart.
The code must define a figure named 'fig' and not show it (the system will handle display).
Do NOT use pandas.read_csv() - assume the data is already available in a variable named 'df'.
"""

            user_prompt = f"""
Create a Plotly visualization for the following request: "{query}"

DATA CONTEXT:
{data_context}

REQUIREMENTS:
1. Use `import plotly.express as px`.
2. Assume the dataframe is named `df`.
3. Create the figure and assign it to variable `fig`.
4. Do NOT call `fig.show()`.
5. Return ONLY the valid Python code in a markdown block.
"""
            
            # Use primary model for code generation
            response = self.initializer.llm_client.generate(
                prompt=user_prompt,
                system=system_prompt,
                model=self.initializer.primary_llm.model
            )
            
            result_text = response.get('response', str(response)) if isinstance(response, dict) else str(response)
            
            # Clean up code block
            code_block = result_text
            if "```python" in code_block:
                code_block = code_block.split("```python")[1].split("```")[0].strip()
            elif "```" in code_block:
                 code_block = code_block.split("```")[1].split("```")[0].strip()
            
            return {
                "success": True,
                "result": code_block,
                "metadata": {"agent": "Visualizer", "mode": "direct_generation"}
            }
            
        except Exception as e:
            logging.error(f"Visualizer execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Visualizer Agent Plugin
# Handles data visualization tasks
# Phase 3.4: Now executes generated code in sandbox

import sys
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from backend.core.plugin_system import BasePluginAgent, AgentMetadata, AgentCapability
from backend.agents.model_manager import get_model_manager
from backend.core.security.sandbox import EnhancedSandbox


class VisualizerAgent(BasePluginAgent):
    """
    Visualizer Agent Plugin.
    Generates Plotly/Matplotlib code for data visualization.
    Phase 3.4: Now executes generated code in sandbox for immediate results.
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config=config)
        self.sandbox = None  # Lazy initialization
    
    def get_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="Visualizer",
            version="2.0.0",  # Updated for Phase 3.4
            description="Generates and executes interactive data visualizations (Plotly)",
            author="Nexus Team",
            capabilities=[AgentCapability.VISUALIZATION],
            file_types=[],
            dependencies=["plotly"],
            priority=20
        )
    
    def initialize(self, **kwargs) -> bool:
        self.initializer = get_model_manager()
        return True
    
    def _get_sandbox(self) -> EnhancedSandbox:
        """Lazy initialization of sandbox"""
        if self.sandbox is None:
            self.sandbox = EnhancedSandbox(max_memory_mb=256, max_cpu_seconds=30)
        return self.sandbox
    
    def can_handle(self, query: str, file_type: Optional[str] = None, **kwargs) -> float:
        # Check for visualization keywords
        keywords = ["plot", "chart", "graph", "visualize", "diagram", "trend line", "histogram", "scatter", "bar chart", "pie chart", "heatmap"]
        if any(k in query.lower() for k in keywords):
            return 0.8
        return 0.0

    def _generate_viz_code(self, query: str, data: Any = None) -> str:
        """Generate visualization code using LLM"""
        self.initializer.ensure_initialized()
        
        # Construct data context
        data_context = ""
        if data is not None:
            if hasattr(data, "head"):
                data_context = f"Data Sample:\n{data.head().to_markdown()}\nColumns: {list(data.columns)}\nTypes: {data.dtypes.to_dict()}"
            else:
                data_context = str(data)[:2000]

        system_prompt = """You are a Data Visualization Expert using Python and Plotly Express.
Your goal is to write a single, self-contained Python code block that generates an interactive Plotly chart.
The code must define a figure named 'fig' and not show it (the system will handle display).
Do NOT use pandas.read_csv() - assume the data is already available in a variable named 'df'.
Do NOT call fig.show() - the system handles rendering.
"""

        user_prompt = f"""
Create a Plotly visualization for the following request: "{query}"

DATA CONTEXT:
{data_context}

REQUIREMENTS:
1. Use `import plotly.express as px` or `import plotly.graph_objects as go`.
2. Assume the dataframe is named `df`.
3. Create the figure and assign it to variable `fig`.
4. Do NOT call `fig.show()`.
5. Return ONLY the valid Python code in a markdown block.
"""
        
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
        
        return code_block

    def execute(self, query: str, data: Any = None, execute_code: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Execute visualization task.
        
        Phase 3.4: Now executes generated code in sandbox for immediate visualization.
        
        Args:
            query: The visualization request
            data: DataFrame or data to visualize
            execute_code: If True, execute code in sandbox (default). If False, return code only.
            **kwargs: Additional arguments
            
        Returns:
            Dict with success, result (figure JSON or code), and metadata
        """
        try:
            # Generate visualization code
            code = self._generate_viz_code(query, data)
            
            if not execute_code:
                # Return code only (legacy behavior)
                return {
                    "success": True,
                    "result": code,
                    "type": "code",
                    "metadata": {"agent": "Visualizer", "mode": "code_generation"}
                }
            
            # Phase 3.4: Execute in sandbox
            sandbox = self._get_sandbox()
            
            # Prepare execution context with data
            additional_globals = {}
            if data is not None:
                # Deep copy to prevent modification
                import copy
                try:
                    additional_globals['df'] = copy.deepcopy(data)
                except Exception:
                    additional_globals['df'] = data
            
            # Execute the visualization code
            exec_result = sandbox.execute(code, additional_globals)
            
            if exec_result.get('success'):
                # Check if figure was created
                result_vars = exec_result.get('result', {})
                
                if isinstance(result_vars, dict) and 'fig' in result_vars:
                    fig = result_vars['fig']
                    
                    # Convert Plotly figure to JSON for frontend
                    try:
                        fig_json = fig.to_json()
                        return {
                            "success": True,
                            "result": fig_json,
                            "type": "visualization",
                            "code": code,
                            "metadata": {
                                "agent": "Visualizer",
                                "mode": "executed",
                                "execution_time": exec_result.get('execution_time', 0)
                            }
                        }
                    except AttributeError:
                        # Figure doesn't have to_json, try to serialize
                        return {
                            "success": True,
                            "result": str(fig),
                            "type": "visualization",
                            "code": code,
                            "metadata": {"agent": "Visualizer", "mode": "executed"}
                        }
                else:
                    # No figure created, return code
                    return {
                        "success": True,
                        "result": code,
                        "type": "code",
                        "warning": "Code executed but no 'fig' variable found",
                        "execution_result": exec_result.get('result'),
                        "metadata": {"agent": "Visualizer", "mode": "code_only"}
                    }
            else:
                # Execution failed, return error with code
                return {
                    "success": False,
                    "error": exec_result.get('error', 'Unknown execution error'),
                    "code": code,
                    "type": "error",
                    "metadata": {"agent": "Visualizer", "mode": "execution_failed"}
                }
            
        except Exception as e:
            logging.error(f"Visualizer execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "type": "error"
            }


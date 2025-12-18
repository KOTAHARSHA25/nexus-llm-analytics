# Visualization API endpoint for generating interactive charts
# Handles chart generation requests and returns Plotly JSON
# Enhanced with LIDA-inspired goal-based visualization generation

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import re
import ast
import importlib
import base64
import io
import matplotlib.pyplot as plt
from backend.core.crew_singleton import get_crew_manager
from backend.core.sandbox import EnhancedSandbox
from backend.utils.data_utils import read_dataframe, create_data_summary, clean_code_snippet

router = APIRouter()
logger = logging.getLogger(__name__)

class VisualizationRequest(BaseModel):
    data_summary: str
    chart_type: str = "auto"
    filename: Optional[str] = None
    columns: Optional[List[str]] = None
    custom_params: Optional[Dict[str, Any]] = None

class ChartExecutionRequest(BaseModel):
    plotly_code: str
    data: Dict[str, Any]

class GoalBasedVisualizationRequest(BaseModel):
    """LIDA-inspired goal-based visualization request"""
    filename: str
    goal: Optional[str] = None  # e.g., "Show distribution of sales by region"
    library: str = "plotly"  # plotly, matplotlib, seaborn
    n_goals: int = 3  # Number of goals to generate if not specified

def execute_plotly_code(code: str, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Safely execute Plotly code and return the figure as JSON
    """
    sandbox = EnhancedSandbox(max_memory_mb=512, max_cpu_seconds=60)
    
    # Prepare the execution environment with data and Plotly
    try:
        result = sandbox.execute(code, data=data)
        
        if "error" in result:
            return {"error": result["error"]}
        
        # Look for a figure object in the results
        local_vars = result.get("result", {})
        
        # Common variable names for Plotly figures
        fig_candidates = ['fig', 'figure', 'chart', 'plot']
        
        for var_name in fig_candidates:
            if var_name in local_vars:
                fig = local_vars[var_name]
                
                # Convert Plotly figure to JSON
                if hasattr(fig, 'to_json'):
                    return {
                        "success": True,
                        "figure_json": fig.to_json(),
                        "chart_type": "plotly"
                    }
                elif hasattr(fig, 'to_dict'):
                    return {
                        "success": True,
                        "figure_json": json.dumps(fig.to_dict()),
                        "chart_type": "plotly"
                    }
        
        return {"error": "No Plotly figure found in executed code"}
        
    except Exception as e:
        logging.error(f"Plotly code execution failed: {e}")
        return {"error": f"Chart generation failed: {str(e)}"}

def generate_auto_chart(df: pd.DataFrame, columns: List[str] = None) -> Dict[str, Any]:
    """
    Automatically generate an appropriate chart based on data characteristics
    """
    try:
        if df.empty:
            return {"error": "Dataset is empty"}
        
        # Use specified columns or all columns
        if columns:
            df = df[columns]
        
        # Determine the best chart type based on data
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Scatter plot for two numeric variables
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                           title=f"{numeric_cols[1]} vs {numeric_cols[0]}")
        elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
            # Bar chart for categorical vs numeric
            fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0],
                        title=f"{numeric_cols[0]} by {categorical_cols[0]}")
        elif len(numeric_cols) == 1:
            # Histogram for single numeric variable
            fig = px.histogram(df, x=numeric_cols[0],
                             title=f"Distribution of {numeric_cols[0]}")
        elif len(categorical_cols) >= 1:
            # Count plot for categorical data
            value_counts = df[categorical_cols[0]].value_counts()
            fig = px.bar(x=value_counts.index, y=value_counts.values,
                        title=f"Count of {categorical_cols[0]}")
        else:
            return {"error": "Unable to determine appropriate chart type for this data"}
        
        return {
            "success": True,
            "figure_json": fig.to_json(),
            "chart_type": "auto_generated"
        }
        
    except Exception as e:
        logging.error(f"Auto chart generation failed: {e}")
        return {"error": f"Auto chart generation failed: {str(e)}"}

@router.post("/generate")
async def generate_visualization(request: VisualizationRequest):
    """
    Generate a visualization using CrewAI or auto-generation
    """
    logging.info(f"[VISUALIZE] Request: {request.chart_type}, filename: {request.filename}")
    
    try:
        if request.filename:
            # Use centralized path resolver
            from backend.utils.data_utils import DataPathResolver
            
            filepath = DataPathResolver.resolve_data_file(request.filename)
            
            if not filepath:
                raise HTTPException(status_code=404, detail=f"File '{request.filename}' not found. Please upload the file first.")
            
            filepath = str(filepath)
            logging.info(f"[VISUALIZE] Loading data from: {filepath}")
            
            if request.filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif request.filename.endswith('.json'):
                df = pd.read_json(filepath)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type for visualization")
            
            # Use auto-generation if chart_type is "auto"
            if request.chart_type == "auto":
                result = generate_auto_chart(df, request.columns)
                logging.info(f"[VISUALIZE] Auto-generated chart result: {result.get('success', False)}")
                return result
            
            # Use CrewAI for custom chart generation
            crew_manager = get_crew_manager()
            result = crew_manager.create_visualization(
                data_summary=request.data_summary or df.head().to_string(),
                chart_type=request.chart_type
            )
            
            if result.get("success"):
                # Execute the generated Plotly code
                plotly_code = result.get("visualization_code", "")
                chart_result = execute_plotly_code(plotly_code, df)
                
                if chart_result.get("success"):
                    return {
                        "success": True,
                        "figure_json": chart_result["figure_json"],
                        "generated_code": plotly_code,
                        "chart_type": request.chart_type
                    }
                else:
                    return chart_result
            else:
                return result
        
        else:
            raise HTTPException(status_code=400, detail="Filename is required for visualization")
            
    except Exception as e:
        logging.error(f"Visualization generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

@router.post("/execute")
async def execute_chart_code(request: ChartExecutionRequest):
    """
    Execute custom Plotly code with provided data
    """
    logging.info("[VISUALIZE] Executing custom Plotly code")
    
    try:
        # Convert data dict to DataFrame
        df = pd.DataFrame(request.data)
        
        # Execute the Plotly code
        result = execute_plotly_code(request.plotly_code, df)
        
        logging.info(f"[VISUALIZE] Code execution result: {result.get('success', False)}")
        return result
        
    except Exception as e:
        logging.error(f"Chart code execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Code execution failed: {str(e)}")

def preprocess_visualization_code(code: str, library: str = "plotly") -> str:
    """
    Preprocess generated visualization code (uses centralized function).
    Removes markdown artifacts and ensures proper structure.
    """
    from backend.utils.data_utils import preprocess_visualization_code as preprocess
    return preprocess(code, library=library)


def execute_matplotlib_code(code: str, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute matplotlib/seaborn visualization code (LIDA-inspired).
    Returns base64-encoded PNG image.
    """
    try:
        # Create execution environment
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        exec_locals = {
            'pd': pd,
            'data': data,
            'plt': plt,
            'np': __import__('numpy'),
        }
        
        # Also import seaborn if mentioned
        if 'seaborn' in code or 'sns' in code:
            import seaborn as sns
            exec_locals['sns'] = sns
        
        # Execute code
        exec(code, exec_locals)
        
        # Get figure from chart variable or current figure
        chart = exec_locals.get('chart') or exec_locals.get('fig') or plt.gcf()
        
        # Convert to base64 PNG
        buf = io.BytesIO()
        plt.grid(color="lightgray", linestyle="dashed", zorder=-10)
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode("ascii")
        plt.close('all')
        
        return {
            "success": True,
            "image_base64": plot_data,
            "chart_type": "matplotlib"
        }
        
    except Exception as e:
        logger.error(f"Matplotlib code execution failed: {e}")
        plt.close('all')  # Clean up
        return {"error": f"Chart generation failed: {str(e)}"}


@router.post("/goal-based")
async def generate_goal_based_visualization(request: GoalBasedVisualizationRequest):
    """
    Generate visualization based on a data goal (LIDA-inspired approach).
    Automatically analyzes data and generates appropriate visualizations.
    """
    logger.info(f"[VISUALIZE] Goal-based request: {request.goal}, file: {request.filename}")
    
    try:
        # Use centralized path resolver
        from backend.utils.data_utils import DataPathResolver
        
        filepath = DataPathResolver.resolve_data_file(request.filename)
        
        if not filepath:
            raise HTTPException(status_code=404, detail=f"File '{request.filename}' not found")
        
        filepath = str(filepath)
        logger.info(f"[VISUALIZE] Loading data from: {filepath}")
        df = read_dataframe(filepath)
        
        # Create comprehensive data summary (LIDA approach)
        data_summary = create_data_summary(df, request.filename)
        
        # Use CrewAI to generate visualization goals and code
        crew_manager = get_crew_manager()
        
        if request.goal:
            # User specified a goal
            prompt = f"""
            Dataset Summary: {json.dumps(data_summary, indent=2)}
            
            User Goal: {request.goal}
            
            Generate {request.library} visualization code that accomplishes this goal.
            The code must:
            1. Use only fields that exist in the dataset
            2. Apply appropriate transformations
            3. Follow visualization best practices
            4. Be complete and executable
            5. Store the final figure in a variable named 'chart' or 'fig'
            
            Return ONLY the Python code wrapped in ```python ``` markers.
            """
        else:
            # Auto-generate goals
            prompt = f"""
            Dataset Summary: {json.dumps(data_summary, indent=2)}
            
            Generate {request.n_goals} insightful visualization goals for this dataset.
            For each goal, create {request.library} code that:
            1. Uses only existing dataset fields
            2. Applies appropriate transformations
            3. Follows visualization best practices
            4. Is complete and executable
            
            Return the code for the most important visualization.
            """
        
        # Get visualization code from CrewAI
        result = crew_manager.analyze_structured_data(
            query=prompt,
            data_summary=data_summary
        )
        
        # Extract code from result
        code = result.get("analysis", "")
        code = preprocess_visualization_code(code)
        
        logger.info(f"[VISUALIZE] Generated code length: {len(code)}")
        
        # Execute based on library
        if request.library in ["matplotlib", "seaborn"]:
            chart_result = execute_matplotlib_code(code, df)
        else:  # plotly or default
            chart_result = execute_plotly_code(code, df)
        
        if chart_result.get("success"):
            return {
                "success": True,
                "visualization": chart_result,
                "generated_code": code,
                "data_summary": data_summary,
                "goal": request.goal
            }
        else:
            return chart_result
            
    except Exception as e:
        logger.error(f"Goal-based visualization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


@router.get("/types")
async def get_supported_chart_types():
    """
    Return list of supported chart types
    """
    chart_types = {
        "auto": "Automatically choose best chart type",
        "bar": "Bar chart for categorical data",
        "line": "Line chart for time series or continuous data",
        "scatter": "Scatter plot for correlation analysis",
        "histogram": "Histogram for distribution analysis",
        "box": "Box plot for statistical summary",
        "violin": "Violin plot for distribution shape",
        "pie": "Pie chart for proportional data",
        "heatmap": "Heatmap for correlation matrix",
        "treemap": "Treemap for hierarchical data",
        "sunburst": "Sunburst chart for hierarchical data",
        "parallel_coordinates": "Parallel coordinates for multivariate analysis",
        "radar": "Radar chart for multi-dimensional data"
    }
    
    return {
        "success": True,
        "chart_types": chart_types
    }
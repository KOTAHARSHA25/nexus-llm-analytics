# Visualization API endpoint for generating interactive charts
# Handles chart generation requests and returns Plotly JSON

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
from agents.crew_manager import CrewManager
from core.sandbox import EnhancedSandbox

router = APIRouter()

class VisualizationRequest(BaseModel):
    data_summary: str
    chart_type: str = "auto"
    filename: Optional[str] = None
    columns: Optional[List[str]] = None
    custom_params: Optional[Dict[str, Any]] = None

class ChartExecutionRequest(BaseModel):
    plotly_code: str
    data: Dict[str, Any]

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
            # Load data from file
            import os
            filepath = os.path.join(os.path.dirname(__file__), '..', 'data', request.filename)
            
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
            crew_manager = CrewManager()
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
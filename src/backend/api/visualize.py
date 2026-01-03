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

from backend.core.security.sandbox import EnhancedSandbox
from backend.utils.data_utils import read_dataframe, create_data_summary, clean_code_snippet
from backend.visualization.dynamic_charts import ChartTypeAnalyzer, DynamicChartGenerator

router = APIRouter()
logger = logging.getLogger(__name__)

# Debug mode flag - set to False to reduce console noise
_DEBUG_VISUALIZE = False

def _debug_print(msg: str):
    """Print debug message only if debug mode is enabled, with ASCII-safe output."""
    if _DEBUG_VISUALIZE:
        # Remove emojis for Windows console compatibility
        safe_msg = ''.join(char if ord(char) < 128 else '?' for char in msg)
        print(safe_msg)

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
    analysis_context: Optional[str] = None  # Analysis results to guide visualization

def execute_plotly_code(code: str, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Safely execute Plotly code and return the figure as JSON
    
    Note: Uses direct exec() instead of sandbox because Plotly modules
    cannot pass through the sandbox's type safety checks. This is similar
    to how matplotlib execution is handled (see execute_matplotlib_code).
    """
    try:
        # Log the code being executed
        logging.info(f"[PLOTLY] Executing code:\n{code[:500]}")
        
        # Create execution environment with Plotly modules and data
        exec_locals = {
            'pd': pd,
            'data': data,
            'px': px,
            'go': go,
            'plotly': plotly,
            'np': __import__('numpy'),
        }
        
        # Execute the Plotly code
        exec(code, exec_locals)
        
        # Log what variables were created
        var_names = [k for k in exec_locals.keys() if not k.startswith('_') and k not in ['pd', 'data', 'px', 'go', 'plotly', 'np']]
        logging.info(f"[PLOTLY] Variables created: {var_names}")
        
        # Look for a figure object in the results
        # Common variable names for Plotly figures
        fig_candidates = ['fig', 'figure', 'chart', 'plot']
        
        for var_name in fig_candidates:
            if var_name in exec_locals:
                fig = exec_locals[var_name]
                logging.info(f"[PLOTLY] Found figure in variable: {var_name}")
                
                # Convert Plotly figure to JSON with proper encoding
                # CRITICAL: Use PlotlyJSONEncoder to handle numpy arrays
                if hasattr(fig, 'to_dict'):
                    logging.info(f"[PLOTLY] Using to_dict() method")
                    from plotly.utils import PlotlyJSONEncoder
                    fig_dict = fig.to_dict()
                    # Use Plotly's custom encoder to handle numpy arrays
                    figure_json = json.dumps(fig_dict, cls=PlotlyJSONEncoder)
                    logging.info(f"[PLOTLY] JSON length: {len(figure_json)}")
                    return {
                        "success": True,
                        "figure_json": figure_json,
                        "chart_type": "plotly"
                    }
                elif hasattr(fig, 'to_json'):
                    logging.info(f"[PLOTLY] Using to_json() method (fallback)")
                    # This may use binary encoding - not ideal
                    return {
                        "success": True,
                        "figure_json": fig.to_json(),
                        "chart_type": "plotly"
                    }
        
        logging.warning(f"[PLOTLY] No figure found. Available variables: {var_names}")
        return {"error": "No Plotly figure found in executed code"}
        
    except Exception as e:
        logging.error(f"Plotly code execution failed: {e}")
        logging.error(f"Code that failed:\n{code}")
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
    Generate a visualization using intelligent agent routing or auto-generation
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
            
            # Use Visualizer Agent for custom chart generation
            from backend.core.plugin_system import get_agent_registry
            registry = get_agent_registry()
            viz_agent = registry.get_agent("Visualizer")
            
            if not viz_agent:
                 raise HTTPException(status_code=500, detail="Visualizer Agent not found")

            query = f"Create a {request.chart_type} chart for the data."
            if request.chart_type == "auto":
                query = "Create the most appropriate chart for this data."

            execution_result = viz_agent.execute(
                query=query,
                data=request.data_summary or df.head().to_string()
            )
            
            if execution_result.get("success"):
                # Extract code from the result
                raw_result = execution_result.get("result", "")
                from backend.utils.data_utils import clean_code_snippet
                plotly_code = clean_code_snippet(raw_result)
                
                # Create a result dict similar to what was expected
                result = {
                    "success": True,
                    "visualization_code": plotly_code
                }
            else:
                result = {"success": False, "error": execution_result.get("error")}
            
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

# NOTE: /execute endpoint removed (Dec 2025)
# Custom Plotly code execution is now handled by the main POST / endpoint
# or via the sandbox in /api/viz/edit for modifications

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
    Generate visualization using DETERMINISTIC template-based approach.
    Follows IMMUTABLE RULES: 100% accuracy, dynamic data handling, no hardcoding.
    
    NEW FEATURES:
    - Auto-suggests best chart types based on data structure
    - Allows user to choose specific chart type
    - Template-based (no LLM code generation = 100% reliable)
    """
    logger.debug(f"[VISUALIZE] Goal-based request: {request.goal}, file: {request.filename}")
    
    try:
        # Use centralized path resolver
        from backend.utils.data_utils import DataPathResolver
        
        filepath = DataPathResolver.resolve_data_file(request.filename)
        
        if not filepath:
            raise HTTPException(status_code=404, detail=f"File '{request.filename}' not found")
        
        filepath = str(filepath)
        logger.debug(f"[VISUALIZE] Loading data from: {filepath}")
        df = read_dataframe(filepath)
        
        # Analyze data structure (DYNAMIC - works with any data)
        analyzer = ChartTypeAnalyzer()
        data_analysis = analyzer.analyze_data_structure(df)
        suggestions = analyzer.suggest_chart_types(df, data_analysis)
        
        logger.debug(f"[VISUALIZE] Data analysis: {data_analysis}")
        logger.debug(f"[VISUALIZE] Chart suggestions: {len(suggestions)}")
        
        # Determine chart type using HYBRID approach:
        # 1. Data structure analysis (deterministic, reliable)
        # 2. LLM for filtering only (simpler task)
        chart_type = None
        chart_params = {}
        
        # FIRST: Get smart suggestions based on actual data structure
        logger.debug(f"[VISUALIZE] Analyzing data structure for chart suggestions...")
        
        if request.goal and len(request.goal.strip()) > 10:
            # Use LLM ONLY for filtering (simpler, more reliable)
            logger.debug(f"[VISUALIZE] Using LLM for filtering logic: {request.goal}")
            
            from backend.core.plugin_system import get_agent_registry
            registry = get_agent_registry()
            analyst_agent = registry.get_agent("DataAnalyst") # or "Time Series" or whatever, but DataAnalyst is safe fallback
            
            # STEP 1: Simple and direct - let LLM generate filter + chart suggestion together
            if request.analysis_context or request.goal:
                pass  # Verbose logging disabled
                
                # SIMPLIFIED: LLM only handles FILTERING (easier task)
                # Show sample values to help LLM understand the data
                sample_values = {}
                for col in df.columns:
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) <= 10:  # Only show if not too many unique values
                        sample_values[col] = unique_vals[:5].tolist()
                
                filter_query = f"""SIMPLE TASK: Does this question need data filtering?

USER ASKED: "{request.goal}"

DATA AVAILABLE:
Columns: {list(df.columns)}
Total rows: {len(df)}
Sample values: {sample_values}

YOUR ONLY JOB: Determine if filtering is needed based on the question and available data.

FILTERING LOGIC:
1. Check if question mentions SPECIFIC values that appear in the data
2. Look for phrases like "only", "specific", "just", "filter to", etc.
3. Check if question asks for ALL/TOTAL/COMPARE → probably no filter needed

STEPS:
1. Read the question carefully
2. Check if any SPECIFIC values from the data are mentioned
3. If yes → generate filter code using df[df['actual_column_name'].str.contains('ActualValue', case=False, na=False)]
4. If no specific values mentioned → set needs_filter=false

IMPORTANT: Use the ACTUAL column names and values from the data provided above!

RESPONSE (JSON only):
{{
  "needs_filter": true/false,
  "filter_code": "df[df['actual_column_name'].str.contains('ActualValue', case=False, na=False)]",
  "explanation": "Brief reason"
}}

NOW PROCESS THIS QUESTION!"""

                filter_result = {"success": False}
                if analyst_agent:
                    filter_result = analyst_agent.execute(
                        query=filter_query,
                        data=df.head().to_string() # Context provided in query mostly
                    )
                
                if filter_result.get("success") and filter_result.get("result"):
                    result_text = filter_result["result"]
                    
                    try:
                        # Parse JSON - handle markdown wrapping
                        if "```json" in result_text:
                            result_text = result_text.split("```json")[1].split("```")[0]
                        elif "```" in result_text:
                            result_text = result_text.split("```")[1].split("```")[0]
                        
                        filter_config = json.loads(result_text.strip())
                        
                        # Apply filter if specified
                        if filter_config.get("needs_filter") and filter_config.get("filter_code"):
                            filter_code = filter_config["filter_code"]
                            original_rows = len(df)
                            
                            try:
                                local_namespace = {'df': df, 'pd': pd}
                                filtered_df = eval(filter_code, {"__builtins__": {}}, local_namespace)
                                
                                if isinstance(filtered_df, pd.DataFrame) and len(filtered_df) > 0:
                                    df = filtered_df
                                else:
                                    pass  # Filter returned empty/invalid result
                            except Exception as e:
                                logger.debug(f"Filter execution error: {e}")
                        
                    except Exception as e:
                        logger.debug(f"Filter parse error: {e}")
        
        # SMART CHART SELECTION: Parse user's question FIRST, then use data structure
        if chart_type is None and suggestions:
            # Re-analyze with potentially filtered data
            fresh_analysis = analyzer.analyze_data_structure(df)
            fresh_suggestions = analyzer.suggest_chart_types(df, fresh_analysis)
            
            # STEP 1: Check if user's question indicates specific chart type
            question_lower = request.goal.lower() if request.goal else ""
            user_intent_type = None
            
            # PRIORITY: Check if user explicitly requests a chart type
            # This overrides all other detection logic
            if 'line chart' in question_lower or 'create a line' in question_lower or 'make a line' in question_lower:
                user_intent_type = 'line'
            elif 'bar chart' in question_lower or 'create a bar' in question_lower or 'make a bar' in question_lower:
                user_intent_type = 'bar'
            elif 'scatter plot' in question_lower or 'scatter chart' in question_lower or 'create a scatter' in question_lower:
                user_intent_type = 'scatter'
            elif 'histogram' in question_lower or 'create a histogram' in question_lower:
                user_intent_type = 'histogram'
            elif 'pie chart' in question_lower or 'create a pie' in question_lower or 'make a pie' in question_lower:
                user_intent_type = 'pie'
            
            # If no explicit chart type, use keyword-based detection
            # Compare BY (categorical) -> BAR CHART
            elif ('compare' in question_lower or 'comparison' in question_lower) and ('by' in question_lower or 'across' in question_lower):
                user_intent_type = 'bar'
            
            # Correlation/Relationship -> SCATTER
            elif any(word in question_lower for word in ['correlation', 'relationship', ' vs ', ' vs.', 'against']):
                user_intent_type = 'scatter'
            
            # Distribution/Spread -> HISTOGRAM
            elif any(word in question_lower for word in ['distribution', 'spread', 'range', 'frequency']):
                user_intent_type = 'histogram'
            
            # Trend/Time -> LINE (only if data supports it)
            elif any(word in question_lower for word in ['trend', 'over time', 'growth', 'change over']):
                # Check if data actually has datetime or sequential data
                has_datetime = len(fresh_analysis.get('datetime_columns', [])) > 0
                if has_datetime:
                    user_intent_type = 'line'
            
            # Share/Proportion -> PIE
            elif any(word in question_lower for word in ['share', 'proportion', 'percentage', 'breakdown of', 'diversification', 'composition', 'split', 'allocation']):
                user_intent_type = 'pie'
            
            # STEP 2: Find matching suggestion or use intent
            if user_intent_type:
                # Look for matching suggestion
                matching = [s for s in fresh_suggestions if s['type'] == user_intent_type]
                if matching:
                    best_suggestion = matching[0]
                else:
                    # Force the user's preferred type even if not in top suggestions
                    best_suggestion = {'type': user_intent_type}
            else:
                # No clear intent, use highest priority
                best_suggestion = fresh_suggestions[0]
            
            chart_type = best_suggestion['type']
            
            # SMART COLUMN DETECTION: Parse column names from question
            def extract_columns_from_question(question: str, df_columns: list) -> list:
                """Extract column references from natural language question"""
                question_lower = question.lower()
                detected = []
                
                # Check each column name against question (handle underscores/spaces)
                for col in df_columns:
                    col_variants = [
                        col.lower(),
                        col.lower().replace('_', ' '),
                        col.lower().replace('_', '')
                    ]
                    if any(variant in question_lower for variant in col_variants):
                        detected.append(col)
                
                return detected
            
            # Detect columns mentioned in question
            mentioned_columns = extract_columns_from_question(request.goal, df.columns.tolist())
            
            # Extract suggested columns
            param_mapping = {
                'x_column': 'x_col',
                'y_column': 'y_col',
                'values_column': 'values_col',
                'names_column': 'names_col'
            }
            chart_params = {param_mapping[k]: v for k, v in best_suggestion.items() 
                          if k in param_mapping}
            
            # OVERRIDE with mentioned columns for specific chart types
            if user_intent_type == 'histogram' and mentioned_columns:
                # For histogram, use the first numeric column mentioned
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                histogram_col = next((col for col in mentioned_columns if col in numeric_cols), None)
                if histogram_col:
                    chart_params['x_col'] = histogram_col
                    chart_params.pop('y_col', None)
            
            elif user_intent_type == 'scatter' and len(mentioned_columns) >= 2:
                # For scatter, use two mentioned numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                scatter_cols = [col for col in mentioned_columns if col in numeric_cols]
                if len(scatter_cols) >= 2:
                    chart_params['x_col'] = scatter_cols[0]
                    chart_params['y_col'] = scatter_cols[1]
            
            elif user_intent_type == 'line' and len(mentioned_columns) >= 2:
                # For line chart: x-axis (categorical or datetime) vs y-axis (numeric)
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                # Find x-axis column (categorical, datetime, or numeric)
                line_x = next((col for col in mentioned_columns if col in categorical_cols + datetime_cols), None)
                # Find y-axis column (numeric)
                line_y = next((col for col in mentioned_columns if col in numeric_cols), None)
                
                if line_x and line_y:
                    chart_params['x_col'] = line_x
                    chart_params['y_col'] = line_y
                elif line_y:
                    # If only numeric mentioned, use first categorical/datetime for x
                    if categorical_cols or datetime_cols:
                        chart_params['x_col'] = (categorical_cols + datetime_cols)[0]
                    chart_params['y_col'] = line_y
            
            elif user_intent_type == 'bar' and len(mentioned_columns) >= 2:
                # For bar chart: categorical column (x) vs numeric column (y)
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                # Find categorical and numeric columns from mentioned
                bar_categorical = next((col for col in mentioned_columns if col in categorical_cols), None)
                bar_numeric = next((col for col in mentioned_columns if col in numeric_cols), None)
                
                if bar_categorical and bar_numeric:
                    chart_params['x_col'] = bar_categorical
                    chart_params['y_col'] = bar_numeric
                elif bar_categorical:
                    # If only categorical mentioned, use first numeric for y
                    chart_params['x_col'] = bar_categorical
                    if numeric_cols:
                        chart_params['y_col'] = numeric_cols[0]
                elif bar_numeric:
                    # If only numeric mentioned, use first categorical for x
                    if categorical_cols:
                        chart_params['x_col'] = categorical_cols[0]
                    chart_params['y_col'] = bar_numeric
            
            elif user_intent_type == 'pie' and len(mentioned_columns) >= 2:
                # For pie chart: categorical column (names) vs numeric column (values)
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                # Find categorical and numeric columns from mentioned
                pie_categorical = next((col for col in mentioned_columns if col in categorical_cols), None)
                pie_numeric = next((col for col in mentioned_columns if col in numeric_cols), None)
                
                if pie_categorical and pie_numeric:
                    chart_params['names_col'] = pie_categorical  # DynamicChartGenerator uses 'names_col'
                    chart_params['values_col'] = pie_numeric
                elif pie_categorical:
                    # If only categorical mentioned, use first numeric for values
                    chart_params['names_col'] = pie_categorical
                    if numeric_cols:
                        chart_params['values_col'] = numeric_cols[0]
                elif pie_numeric:
                    # If only numeric mentioned, use first categorical for names
                    if categorical_cols:
                        chart_params['names_col'] = categorical_cols[0]
                    chart_params['values_col'] = pie_numeric
        
        # Final fallback
        if chart_type is None:
            chart_type = 'bar'
        
        logger.debug(f"[VISUALIZE] Selected chart type: {chart_type}")
        logger.debug(f"[VISUALIZE] Chart parameters: {chart_params}")
        
        # Generate chart using deterministic template
        generator = DynamicChartGenerator()
        fig = generator.create_chart(df, chart_type, **chart_params)
        
        # Convert binary-encoded data to plain arrays for JSON serialization
        fig_dict = fig.to_dict()
        
        # Process each trace to ensure proper serialization
        for trace_idx, trace in enumerate(fig_dict.get('data', [])):
            # Check each potential data key
            for data_key in ['x', 'y', 'z', 'lat', 'lon', 'values', 'text', 'marker', 'customdata']:
                if data_key in trace:
                    val = trace[data_key]
                    
                    # Case 1: Binary encoded (dict with dtype/bdata)
                    if isinstance(val, dict) and 'dtype' in val and 'bdata' in val:
                        logger.warning(f"Binary encoding detected in trace[{trace_idx}]['{data_key}'], converting to list")
                        
                        # Extract original data from DataFrame based on chart type
                        fixed = False
                        
                        # SCATTER CHART
                        if chart_type == 'scatter':
                            if data_key == 'x':
                                col_name = chart_params.get('x_col') or df.select_dtypes(include=['number']).columns[0]
                                trace[data_key] = df[col_name].tolist()
                                fixed = True
                            elif data_key == 'y':
                                col_name = chart_params.get('y_col') or df.select_dtypes(include=['number']).columns[1]
                                trace[data_key] = df[col_name].tolist()
                                fixed = True
                        
                        # HISTOGRAM
                        elif chart_type == 'histogram':
                            if data_key == 'x':
                                col_name = chart_params.get('x_col') or chart_params.get('column')
                                if col_name and col_name in df.columns:
                                    trace[data_key] = df[col_name].tolist()
                                    fixed = True
                        
                        # LINE CHART
                        elif chart_type == 'line':
                            if data_key == 'x':
                                col_name = chart_params.get('x_col')
                                if col_name and col_name in df.columns:
                                    trace[data_key] = df[col_name].tolist()
                                    fixed = True
                            elif data_key == 'y':
                                col_name = chart_params.get('y_col')
                                if col_name and col_name in df.columns:
                                    trace[data_key] = df[col_name].tolist()
                                    fixed = True
                        
                        # BAR CHART
                        elif chart_type == 'bar':
                            if data_key == 'x':
                                col_name = chart_params.get('x_col') or chart_params.get('category_col')
                                if col_name and col_name in df.columns:
                                    trace[data_key] = df[col_name].tolist()
                                    fixed = True
                            elif data_key == 'y':
                                col_name = chart_params.get('y_col') or chart_params.get('value_col')
                                if col_name and col_name in df.columns:
                                    trace[data_key] = df[col_name].tolist()
                                    fixed = True
                        
                        # PIE CHART
                        elif chart_type == 'pie':
                            if data_key == 'labels':
                                col_name = chart_params.get('names_col') or chart_params.get('labels_col') or chart_params.get('category_col')
                                if col_name and col_name in df.columns:
                                    trace[data_key] = df[col_name].tolist()
                                    fixed = True
                            elif data_key == 'values':
                                col_name = chart_params.get('values_col') or chart_params.get('value_col')
                                if col_name and col_name in df.columns:
                                    trace[data_key] = df[col_name].tolist()
                                    fixed = True
                        
                        # If not fixed, use empty list as fallback
                        if not fixed:
                            logger.warning(f"Cannot auto-fix {data_key} for {chart_type}, using empty list")
                            trace[data_key] = []
                    
                    # Case 2: Numpy array or Pandas Series
                    elif hasattr(val, 'tolist'):
                        trace[data_key] = val.tolist()
        
        # Recursively convert ALL numpy arrays to lists for JSON serialization
        def convert_numpy_to_list(obj):
            """Recursively convert numpy arrays to Python lists"""
            import numpy as np
            
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_to_list(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_list(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_to_list(item) for item in obj)
            elif hasattr(obj, 'tolist'):  # Pandas Series or other array-like
                return obj.tolist()
            else:
                return obj
        
        fig_dict = convert_numpy_to_list(fig_dict)
        
        # Use standard json.dumps for guaranteed plain arrays
        figure_json = json.dumps(fig_dict)
        
        return {
            "success": True,
            "visualization": {
                "success": True,
                "figure_json": figure_json,
                "chart_type": chart_type
            },
            "data_analysis": data_analysis,
            "suggestions": suggestions[:5],  # Top 5 suggestions
            "selected_chart": {
                "type": chart_type,
                "reason": next((s['reason'] for s in suggestions if s['type'] == chart_type), 
                             f"User requested {chart_type} chart")
            },
            "goal": request.goal
        }
        
    except Exception as e:
        logger.error(f"Goal-based visualization failed: {e}")
        logger.error(f"Traceback:", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


@router.get("/types")
async def get_supported_chart_types():
    """
    Return list of supported chart types
    """
    chart_types = {
        "auto": "Automatically choose best chart type based on data",
        "bar": "Bar chart for comparing categorical data",
        "line": "Line chart for time series or trends",
        "scatter": "Scatter plot for correlation analysis",
        "histogram": "Histogram for distribution analysis",
        "box": "Box plot for statistical summary",
        "pie": "Pie chart for proportional data"
    }
    
    return {
        "success": True,
        "chart_types": chart_types
    }


@router.post("/suggestions")
async def get_chart_suggestions(request: dict):
    """
    Get intelligent chart suggestions based on data structure.
    
    Request body:
    {
        "filename": "data.csv"
    }
    
    Returns:
    - Data analysis (column types, counts)
    - Ranked chart suggestions with reasoning
    - Recommended chart type
    """
    filename = request.get("filename")
    
    if not filename:
        raise HTTPException(status_code=400, detail="filename is required")
    
    try:
        from backend.utils.data_utils import DataPathResolver
        
        filepath = DataPathResolver.resolve_data_file(filename)
        if not filepath:
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
        
        df = read_dataframe(str(filepath))
        
        # Analyze data structure
        analyzer = ChartTypeAnalyzer()
        data_analysis = analyzer.analyze_data_structure(df)
        suggestions = analyzer.suggest_chart_types(df, data_analysis)
        
        return {
            "success": True,
            "filename": filename,
            "data_analysis": {
                "rows": data_analysis['row_count'],
                "columns": data_analysis['column_count'],
                "numeric_columns": data_analysis['numeric_columns'],
                "categorical_columns": data_analysis['categorical_columns'],
                "datetime_columns": data_analysis['datetime_columns']
            },
            "suggestions": suggestions,
            "recommended": suggestions[0] if suggestions else None
        }
        
    except Exception as e:
        logger.error(f"Chart suggestions failed: {e}")
        raise HTTPException(status_code=500, detail=f"Suggestions failed: {str(e)}")
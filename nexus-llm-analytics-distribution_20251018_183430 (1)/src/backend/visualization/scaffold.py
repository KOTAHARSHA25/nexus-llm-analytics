"""
Chart Scaffold Templates for Multiple Visualization Libraries.
Integrated from Microsoft LIDA - provides library-specific code templates with best practices.
"""

from typing import Tuple
from pydantic import BaseModel


class VisualizationGoal(BaseModel):
    """Lightweight goal model for scaffold"""
    question: str = ""
    visualization: str = ""
    index: int = 0


class ChartScaffold:
    """
    Return code scaffold for charts in multiple visualization libraries.
    Based on LIDA's ChartScaffold with enhancements.
    
    Provides templates and instructions for:
    - matplotlib: Static plots
    - seaborn: Statistical visualizations  
    - plotly: Interactive charts
    - altair: Declarative visualizations
    - ggplot: Grammar of graphics (plotnine)
    """
    
    def __init__(self):
        pass
    
    def get_template(self, goal: VisualizationGoal, library: str) -> Tuple[str, dict]:
        """
        Get code template and instructions for a specific visualization library.
        
        Args:
            goal: Visualization goal with question and visualization type
            library: Target visualization library (matplotlib, seaborn, plotly, altair, ggplot)
            
        Returns:
            Tuple of (template_code, instructions_dict)
        """
        
        # General instructions that apply to all libraries
        general_instructions = f"""
        If the solution requires a single value (e.g. max, min, median, first, last etc), ALWAYS add a line (axvline or axhline) to the chart, ALWAYS with a legend containing the single value (formatted with 0.2F). 
        
        If using a <field> where semantic_type=date, YOU MUST APPLY the following transform before using that column:
        i) convert date fields to date types using data['<field>'] = pd.to_datetime(data['<field>'], errors='coerce')
        ii) drop the rows with NaT values: data = data[pd.notna(data['<field>'])]
        iii) convert field to right time format for plotting
        
        ALWAYS make sure the x-axis labels are legible (e.g., rotate when needed).
        
        Solve the task carefully by completing ONLY the <imports> AND <stub> section.
        Given the dataset summary, the plot(data) method should generate a {library} chart ({goal.visualization}) that addresses this goal: {goal.question}.
        
        DO NOT WRITE ANY CODE TO LOAD THE DATA. The data is already loaded and available in the variable data.
        """
        
        matplotlib_instructions = f"{general_instructions} DO NOT include plt.show(). The plot method must return a matplotlib object (plt). Think step by step.\n"
        
        if library == "matplotlib":
            instructions = {
                "role": "assistant",
                "content": f"{matplotlib_instructions}. Use BaseMap for charts that require a map."
            }
            template = f"""
import matplotlib.pyplot as plt
import pandas as pd
<imports>

# plan - step by step approach to solve the visualization goal
# 1. Load and prepare data (already done)
# 2. Apply necessary transformations
# 3. Create the visualization
# 4. Add labels, title, and styling

def plot(data: pd.DataFrame):
    <stub>  # only modify this section
    plt.title('{goal.question}', wrap=True)
    plt.xlabel('X Label')
    plt.ylabel('Y Label')
    plt.legend()
    plt.tight_layout()
    return plt

chart = plot(data)  # data already contains the data to be plotted. Always include this line. No additional code beyond this line.
"""
        
        elif library == "seaborn":
            instructions = {
                "role": "assistant",
                "content": f"{matplotlib_instructions}. Seaborn provides high-level statistical plotting. Use appropriate seaborn functions for statistical visualizations."
            }
            template = f"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
<imports>

# solution plan
# i.  Identify data requirements
# ii. Apply transformations if needed
# iii. Create seaborn visualization
# iv. Customize appearance

def plot(data: pd.DataFrame):
    sns.set_style("whitegrid")
    <stub>  # only modify this section
    plt.title('{goal.question}', wrap=True)
    plt.xlabel('X Label')
    plt.ylabel('Y Label')
    plt.tight_layout()
    return plt

chart = plot(data)  # data already contains the data to be plotted. Always include this line. No additional code beyond this line.
"""
        
        elif library == "plotly":
            instructions = {
                "role": "system",
                "content": f"""{general_instructions}
                
                If calculating metrics such as mean, median, mode, etc. ALWAYS use the option 'numeric_only=True' when applicable and available.
                AVOID visualizations that require nbformat library.
                DO NOT include fig.show().
                The plot method must return a plotly figure object (fig).
                Think step by step.
                """
            }
            template = """
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
<imports>

def plot(data: pd.DataFrame):
    # <stub> only modify this section
    # Example: fig = px.scatter(data, x='column1', y='column2', title='My Chart')
    
    fig.update_layout(
        title_text='Chart Title',
        xaxis_title='X Axis Label',
        yaxis_title='Y Axis Label',
        template='plotly_white'
    )
    
    return fig

chart = plot(data)  # variable data already contains the data to be plotted and should not be loaded again. Always include this line. No additional code beyond this line.
"""
        
        elif library == "altair":
            instructions = {
                "role": "system",
                "content": f"""{general_instructions}
                
                Always add a type that is BASED on semantic_type to each field such as :Q (quantitative), :O (ordinal), :N (nominal), :T (temporal), :G (geojson).
                Use :T if semantic_type is year or date.
                The plot method must return an altair Chart object.
                Think step by step.
                """
            }
            template = """
import altair as alt
import pandas as pd
<imports>

def plot(data: pd.DataFrame):
    chart = <stub>  # only modify this section
    # Example: alt.Chart(data).mark_bar().encode(x='column1:N', y='column2:Q')
    
    return chart

chart = plot(data)  # data already contains the data to be plotted. Always include this line. No additional code beyond this line.
"""
        
        elif library == "ggplot":
            instructions = {
                "role": "assistant",
                "content": f"{general_instructions}. The plot method must return a ggplot object (chart). Think step by step.\n"
            }
            template = """
import plotnine as p9
import pandas as pd
<imports>

def plot(data: pd.DataFrame):
    chart = (
        <stub>  # only modify this section
        # Example: p9.ggplot(data, p9.aes(x='column1', y='column2')) + p9.geom_point()
    )
    
    return chart

chart = plot(data)  # data already contains the data to be plotted. Always include this line. No additional code beyond this line.
"""
        
        else:
            raise ValueError(
                f"Unsupported library: {library}. Choose from 'matplotlib', 'seaborn', 'plotly', 'altair', 'ggplot'."
            )
        
        return template, instructions
    
    def get_best_practices(self, library: str) -> str:
        """
        Get visualization best practices for a specific library.
        
        Args:
            library: Visualization library name
            
        Returns:
            String containing best practices
        """
        practices = {
            "matplotlib": """
            - Use appropriate plot types (bar for categories, line for trends, scatter for correlations)
            - Always add axis labels and title
            - Use legend when multiple series
            - Rotate x-axis labels if they overlap
            - Use tight_layout() to prevent label cutoff
            - Choose appropriate figure size
            - Use color wisely (avoid rainbow colors for continuous data)
            """,
            
            "seaborn": """
            - Use statistical plots (violinplot, boxplot, etc.) for distributions
            - Set appropriate style (whitegrid, darkgrid, white, dark, ticks)
            - Use hue parameter for categorical grouping
            - Use FacetGrid for multi-plot layouts
            - Leverage built-in color palettes
            - Use appropriate context (paper, notebook, talk, poster)
            """,
            
            "plotly": """
            - Use interactive features (hover, zoom, pan)
            - Add appropriate title and axis labels
            - Use update_layout() for customization
            - Choose appropriate template (plotly, plotly_white, plotly_dark, etc.)
            - Use colors from plotly color sequences
            - Add annotations for important insights
            - Use subplots for multiple visualizations
            """,
            
            "altair": """
            - Specify data types explicitly (:Q, :O, :N, :T, :G)
            - Use mark_* methods appropriately (mark_bar, mark_line, mark_point, etc.)
            - Use encode() to map data to visual properties
            - Use interactive selections when appropriate
            - Keep visualizations declarative and readable
            - Use transform_* methods for data transformations
            """,
            
            "ggplot": """
            - Follow grammar of graphics principles
            - Use appropriate geom_* layers
            - Use aes() for aesthetic mappings
            - Use facet_wrap() or facet_grid() for multi-panel plots
            - Use theme_* for styling
            - Use appropriate scales (scale_color_*, scale_fill_*, etc.)
            """
        }
        
        return practices.get(library, "No specific best practices available for this library.")

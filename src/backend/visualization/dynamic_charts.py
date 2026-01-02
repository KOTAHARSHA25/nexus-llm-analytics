"""
Dynamic Chart Generator - Template-Based (100% Deterministic)
Follows IMMUTABLE RULES:
- Works with ANY data structure
- No hardcoded column names
- Template-based (not LLM-generated)
- Suggests best chart type based on data characteristics
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ChartTypeAnalyzer:
    """Dynamically analyze data and suggest appropriate chart types"""
    
    @staticmethod
    def analyze_data_structure(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze DataFrame structure to understand what visualizations are possible.
        DYNAMIC - works with ANY data structure.
        """
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Try to detect date-like strings
        for col in categorical_cols:
            if df[col].dtype == 'object':
                # Check if values look like dates
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    try:
                        # Use format='mixed' to handle various date formats without warnings
                        pd.to_datetime(sample, errors='raise', format='mixed')
                        datetime_cols.append(col)
                        categorical_cols.remove(col)
                    except (ValueError, TypeError, pd.errors.ParserError):
                        pass  # Not a valid datetime format
        
        analysis = {
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'datetime_columns': datetime_cols,
            'row_count': len(df),
            'column_count': len(df.columns),
            'has_numeric': len(numeric_cols) > 0,
            'has_categorical': len(categorical_cols) > 0,
            'has_datetime': len(datetime_cols) > 0,
        }
        
        return analysis
    
    @staticmethod
    def suggest_chart_types(df: pd.DataFrame, analysis: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Suggest appropriate chart types based on data structure.
        Returns list of suggestions with reasoning.
        """
        if analysis is None:
            analysis = ChartTypeAnalyzer.analyze_data_structure(df)
        
        suggestions = []
        
        num_cols = analysis['numeric_columns']
        cat_cols = analysis['categorical_columns']
        date_cols = analysis['datetime_columns']
        
        # Bar Chart: categorical vs numeric
        if cat_cols and num_cols:
            suggestions.append({
                'type': 'bar',
                'priority': 90,
                'reason': f'Compare {num_cols[0]} across {cat_cols[0]} categories',
                'x_column': cat_cols[0],
                'y_column': num_cols[0],
                'use_case': 'Comparing values across categories'
            })
        
        # Line Chart: time series or sequential numeric
        if date_cols and num_cols:
            suggestions.append({
                'type': 'line',
                'priority': 95,
                'reason': f'Show trend of {num_cols[0]} over {date_cols[0]}',
                'x_column': date_cols[0],
                'y_column': num_cols[0],
                'use_case': 'Time series analysis'
            })
        elif len(num_cols) >= 2:
            suggestions.append({
                'type': 'line',
                'priority': 70,
                'reason': f'Show relationship between {num_cols[0]} and {num_cols[1]}',
                'x_column': num_cols[0],
                'y_column': num_cols[1],
                'use_case': 'Trend analysis'
            })
        
        # Scatter: two numeric columns (correlation)
        if len(num_cols) >= 2:
            suggestions.append({
                'type': 'scatter',
                'priority': 85,
                'reason': f'Explore correlation between {num_cols[0]} and {num_cols[1]}',
                'x_column': num_cols[0],
                'y_column': num_cols[1],
                'use_case': 'Correlation analysis'
            })
        
        # Pie Chart: categorical with numeric (proportions)
        if cat_cols and num_cols and len(df[cat_cols[0]].unique()) <= 10:
            suggestions.append({
                'type': 'pie',
                'priority': 75,
                'reason': f'Show distribution of {num_cols[0]} by {cat_cols[0]}',
                'values_column': num_cols[0],
                'names_column': cat_cols[0],
                'use_case': 'Part-to-whole relationships'
            })
        
        # Histogram: single numeric (distribution)
        if num_cols:
            suggestions.append({
                'type': 'histogram',
                'priority': 80,
                'reason': f'Show distribution of {num_cols[0]}',
                'x_column': num_cols[0],
                'use_case': 'Distribution analysis'
            })
        
        # Box Plot: numeric with optional categorical grouping
        if num_cols:
            if cat_cols:
                suggestions.append({
                    'type': 'box',
                    'priority': 70,
                    'reason': f'Compare {num_cols[0]} distribution across {cat_cols[0]}',
                    'x_column': cat_cols[0],
                    'y_column': num_cols[0],
                    'use_case': 'Statistical comparison'
                })
            else:
                suggestions.append({
                    'type': 'box',
                    'priority': 65,
                    'reason': f'Show statistical distribution of {num_cols[0]}',
                    'y_column': num_cols[0],
                    'use_case': 'Statistical summary'
                })
        
        # Sort by priority
        suggestions.sort(key=lambda x: x['priority'], reverse=True)
        
        return suggestions


class DynamicChartGenerator:
    """
    Generate charts dynamically based on data structure.
    NO hardcoded column names - works with ANY data.
    """
    
    @staticmethod
    def create_bar_chart(df: pd.DataFrame, x_col: str = None, y_col: str = None, 
                        title: str = None, color_col: str = None) -> go.Figure:
        """
        Create bar chart dynamically.
        If x_col/y_col not specified, auto-detect from data structure.
        """
        analysis = ChartTypeAnalyzer.analyze_data_structure(df)
        
        # Auto-detect columns if not specified
        if x_col is None:
            x_col = analysis['categorical_columns'][0] if analysis['categorical_columns'] else df.columns[0]
        if y_col is None:
            y_col = analysis['numeric_columns'][0] if analysis['numeric_columns'] else df.columns[-1]
        
        if title is None:
            title = f'{y_col.title()} by {x_col.title()}'
        
        fig = px.bar(df, x=x_col, y=y_col, title=title, color=color_col)
        fig.update_layout(template='plotly_white', xaxis_title=x_col.title(), yaxis_title=y_col.title())
        
        return fig
    
    @staticmethod
    def create_line_chart(df: pd.DataFrame, x_col: str = None, y_col: str = None,
                         title: str = None, color_col: str = None) -> go.Figure:
        """Create line chart dynamically"""
        analysis = ChartTypeAnalyzer.analyze_data_structure(df)
        
        # Auto-detect: prefer datetime for x-axis
        if x_col is None:
            if analysis['datetime_columns']:
                x_col = analysis['datetime_columns'][0]
            else:
                x_col = df.columns[0]
        
        if y_col is None:
            y_col = analysis['numeric_columns'][0] if analysis['numeric_columns'] else df.columns[-1]
        
        if title is None:
            title = f'{y_col.title()} over {x_col.title()}'
        
        fig = px.line(df, x=x_col, y=y_col, title=title, color=color_col, markers=True)
        fig.update_layout(template='plotly_white', xaxis_title=x_col.title(), yaxis_title=y_col.title())
        
        return fig
    
    @staticmethod
    def create_scatter_chart(df: pd.DataFrame, x_col: str = None, y_col: str = None,
                           title: str = None, color_col: str = None, size_col: str = None) -> go.Figure:
        """Create scatter plot dynamically"""
        analysis = ChartTypeAnalyzer.analyze_data_structure(df)
        
        # Auto-detect: use first two numeric columns
        num_cols = analysis['numeric_columns']
        if x_col is None:
            x_col = num_cols[0] if len(num_cols) > 0 else df.columns[0]
        if y_col is None:
            y_col = num_cols[1] if len(num_cols) > 1 else num_cols[0] if num_cols else df.columns[-1]
        
        if title is None:
            title = f'{y_col.title()} vs {x_col.title()}'
        
        fig = px.scatter(df, x=x_col, y=y_col, title=title, color=color_col, size=size_col)
        
        # Ensure markers are visible even without size parameter
        if size_col is None:
            fig.update_traces(marker=dict(size=10, opacity=0.7))
        
        fig.update_layout(template='plotly_white', xaxis_title=x_col.title(), yaxis_title=y_col.title())
        
        return fig
    
    @staticmethod
    def create_pie_chart(df: pd.DataFrame, values_col: str = None, names_col: str = None,
                        title: str = None) -> go.Figure:
        """Create pie chart dynamically"""
        analysis = ChartTypeAnalyzer.analyze_data_structure(df)
        
        if values_col is None:
            values_col = analysis['numeric_columns'][0] if analysis['numeric_columns'] else df.columns[-1]
        if names_col is None:
            names_col = analysis['categorical_columns'][0] if analysis['categorical_columns'] else df.columns[0]
        
        if title is None:
            title = f'{values_col.title()} Distribution by {names_col.title()}'
        
        # Aggregate data by category
        aggregated = df.groupby(names_col)[values_col].sum().reset_index()
        
        fig = px.pie(aggregated, values=values_col, names=names_col, title=title)
        fig.update_layout(template='plotly_white')
        
        return fig
    
    @staticmethod
    def create_histogram(df: pd.DataFrame, x_col: str = None, title: str = None,
                        nbins: int = 30) -> go.Figure:
        """Create histogram dynamically"""
        analysis = ChartTypeAnalyzer.analyze_data_structure(df)
        
        if x_col is None:
            x_col = analysis['numeric_columns'][0] if analysis['numeric_columns'] else df.columns[0]
        
        if title is None:
            title = f'Distribution of {x_col.title()}'
        
        fig = px.histogram(df, x=x_col, title=title, nbins=nbins)
        fig.update_layout(template='plotly_white', xaxis_title=x_col.title(), yaxis_title='Count')
        
        return fig
    
    @staticmethod
    def create_box_plot(df: pd.DataFrame, y_col: str = None, x_col: str = None,
                       title: str = None) -> go.Figure:
        """Create box plot dynamically"""
        analysis = ChartTypeAnalyzer.analyze_data_structure(df)
        
        if y_col is None:
            y_col = analysis['numeric_columns'][0] if analysis['numeric_columns'] else df.columns[-1]
        
        if title is None:
            if x_col:
                title = f'{y_col.title()} Distribution by {x_col.title()}'
            else:
                title = f'{y_col.title()} Distribution'
        
        fig = px.box(df, y=y_col, x=x_col, title=title)
        fig.update_layout(template='plotly_white')
        
        return fig
    
    @staticmethod
    def create_chart(df: pd.DataFrame, chart_type: str, **kwargs) -> go.Figure:
        """
        Universal chart creation method.
        Dynamically creates any chart type based on data structure.
        """
        generators = {
            'bar': DynamicChartGenerator.create_bar_chart,
            'line': DynamicChartGenerator.create_line_chart,
            'scatter': DynamicChartGenerator.create_scatter_chart,
            'pie': DynamicChartGenerator.create_pie_chart,
            'histogram': DynamicChartGenerator.create_histogram,
            'box': DynamicChartGenerator.create_box_plot,
        }
        
        if chart_type not in generators:
            raise ValueError(f"Unsupported chart type: {chart_type}. Choose from: {list(generators.keys())}")
        
        return generators[chart_type](df, **kwargs)

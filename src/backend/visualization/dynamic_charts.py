"""Dynamic Chart Generator — Nexus LLM Analytics
=================================================

Template-based, 100 % deterministic chart generation.

Immutable Rules
---------------
* Works with ANY data structure — no hardcoded column names.
* Template-based (not LLM-generated).
* Suggests best chart type based on data characteristics.

Classes
-------
ChartTypeAnalyzer
    Inspects DataFrame column types and suggests chart types.
DynamicChartGenerator
    Generates Plotly figures from column metadata.

v2.0 Enterprise Additions
-------------------------
* :class:`ChartGenerationMetrics` — tracks chart creation counts
  by type.
* :func:`get_chart_generator` — thread-safe singleton accessor.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


class ChartTypeAnalyzer:
    """Analyse data structures and suggest appropriate chart types.

    Detects numeric, categorical, and datetime columns and ranks
    chart suggestions by priority.  Fully deterministic — no LLM
    calls.

    Thread Safety:
        All methods are ``@staticmethod`` — inherently thread-safe.
    """
    
    @staticmethod
    def analyze_data_structure(df: pd.DataFrame) -> dict[str, Any]:
        """Analyze DataFrame structure to determine possible visualizations.

        Works dynamically with any data structure by detecting numeric,
        categorical, and datetime columns.

        Args:
            df: Input DataFrame to analyze.

        Returns:
            Dictionary describing column types and data characteristics.
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
    def suggest_chart_types(df: pd.DataFrame, analysis: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Suggest appropriate chart types based on data structure.

        Args:
            df: Input DataFrame to analyze.
            analysis: Pre-computed data analysis; computed on-the-fly if *None*.

        Returns:
            List of suggestion dicts sorted by priority (highest first).
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

    @staticmethod
    def suggest_chart_types(df: pd.DataFrame, analysis: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Suggest appropriate chart types based on data structure.

        Args:
            df: Input DataFrame to analyze.
            analysis: Pre-computed data analysis; computed on-the-fly if *None*.

        Returns:
            List of suggestion dicts sorted by priority (highest first).
        """
        if analysis is None:
            analysis = ChartTypeAnalyzer.analyze_data_structure(df)
        
        suggestions = []
        
        num_cols = analysis['numeric_columns']
        cat_cols = analysis['categorical_columns']
        date_cols = analysis['datetime_columns']
        
        # 1. Heatmap: numeric-only correlation
        if len(num_cols) >= 3:
            suggestions.append({
                'type': 'heatmap',
                'priority': 85,
                'reason': f'Analyze correlations between {len(num_cols)} numeric variables',
                'use_case': 'Correlation matrix'
            })

        # 2. Area Chart: time series composition
        if date_cols and num_cols:
            if cat_cols:
                suggestions.append({
                    'type': 'area',
                    'priority': 80,
                    'reason': f'Show composition of {num_cols[0]} over time by {cat_cols[0]}',
                    'x_column': date_cols[0],
                    'y_column': num_cols[0],
                    'color_column': cat_cols[0],
                    'use_case': 'Stacked time series'
                })
            else:
                suggestions.append({
                    'type': 'area',
                    'priority': 75,
                    'reason': f'Show trend volume of {num_cols[0]} over time',
                    'x_column': date_cols[0],
                    'y_column': num_cols[0],
                    'use_case': 'Volume trend'
                })

        # 3. Violin Plot: rich distribution
        if num_cols:
            if cat_cols:
                suggestions.append({
                    'type': 'violin',
                    'priority': 75,
                    'reason': f'Compare distribution density of {num_cols[0]} across {cat_cols[0]}',
                    'x_column': cat_cols[0],
                    'y_column': num_cols[0],
                    'use_case': 'Distribution density comparison'
                })
            else:
                suggestions.append({
                    'type': 'violin',
                    'priority': 60,
                    'reason': f'Show distribution density of {num_cols[0]}',
                    'y_column': num_cols[0],
                    'use_case': 'Distribution density'
                })

        # 4. Bubble Chart: 3 variables
        if len(num_cols) >= 3:
            suggestions.append({
                'type': 'bubble',
                'priority': 88,
                'reason': f'Multi-variable analysis: {num_cols[0]} vs {num_cols[1]} sized by {num_cols[2]}',
                'x_column': num_cols[0],
                'y_column': num_cols[1],
                'size_column': num_cols[2],
                'use_case': 'Multi-dimensional scatter'
            })

        # 5. Funnel Chart: categorical + decreasing numeric
        if cat_cols and num_cols:
            # Simple heuristic: if numeric values look roughly decreasing
            try:
                # Check first 5 rows for decreasing trend, just as a heuristic
                sample = df[num_cols[0]].head(5).tolist()
                is_decreasing = all(sample[i] >= sample[i+1] for i in range(len(sample)-1)) if len(sample) > 1 else False
                if is_decreasing:
                    suggestions.append({
                        'type': 'funnel',
                        'priority': 70,
                        'reason': f'Show funnel progression of {num_cols[0]} by {cat_cols[0]}',
                        'x_column': num_cols[0],
                        'y_column': cat_cols[0],
                        'use_case': 'Process/Conversion analysis'
                    })
            except Exception:
                pass

        # 6. Treemap & 7. Sunburst: Hierarchical (2+ categorical + 1 numeric)
        if len(cat_cols) >= 2 and num_cols:
            suggestions.append({
                'type': 'treemap',
                'priority': 82,
                'reason': f'Hierarchical view of {num_cols[0]} by {cat_cols[0]} > {cat_cols[1]}',
                'path_columns': cat_cols[:2],
                'values_column': num_cols[0],
                'use_case': 'Hierarchical part-to-whole'
            })
            suggestions.append({
                'type': 'sunburst',
                'priority': 80,
                'reason': f'Radial hierarchy of {num_cols[0]} by {cat_cols[0]} > {cat_cols[1]}',
                'path_columns': cat_cols[:2],
                'values_column': num_cols[0],
                'use_case': 'Radial hierarchy'
            })
        
        # --- Existing Chart Logic (Bar, Line, Scatter, Pie, Histogram, Box) ---
        
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
    def create_bar_chart(df: pd.DataFrame, x_col: str | None = None, y_col: str | None = None, 
                        title: str | None = None, color_col: str | None = None) -> go.Figure:
        """Create a bar chart, auto-detecting columns when not specified.

        Args:
            df: Source DataFrame.
            x_col: Column for x-axis (auto-detected if *None*).
            y_col: Column for y-axis (auto-detected if *None*).
            title: Chart title (auto-generated if *None*).
            color_col: Optional column for colour encoding.

        Returns:
            Plotly ``Figure`` with the rendered bar chart.
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
    def create_line_chart(df: pd.DataFrame, x_col: str | None = None, y_col: str | None = None,
                         title: str | None = None, color_col: str | None = None) -> go.Figure:
        """Create a line chart, preferring datetime columns for the x-axis.

        Args:
            df: Source DataFrame.
            x_col: Column for x-axis (auto-detected if *None*).
            y_col: Column for y-axis (auto-detected if *None*).
            title: Chart title (auto-generated if *None*).
            color_col: Optional column for colour encoding.

        Returns:
            Plotly ``Figure`` with the rendered line chart.
        """
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
    def create_scatter_chart(df: pd.DataFrame, x_col: str | None = None, y_col: str | None = None,
                           title: str | None = None, color_col: str | None = None, size_col: str | None = None) -> go.Figure:
        """Create a scatter plot for correlation analysis.

        Args:
            df: Source DataFrame.
            x_col: Column for x-axis (auto-detected if *None*).
            y_col: Column for y-axis (auto-detected if *None*).
            title: Chart title (auto-generated if *None*).
            color_col: Optional column for colour encoding.
            size_col: Optional column for marker size.

        Returns:
            Plotly ``Figure`` with the rendered scatter plot.
        """
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
    def create_pie_chart(df: pd.DataFrame, values_col: str | None = None, names_col: str | None = None,
                        title: str | None = None) -> go.Figure:
        """Create a pie chart showing part-to-whole relationships.

        Data is grouped by *names_col* and values are summed before plotting.

        Args:
            df: Source DataFrame.
            values_col: Numeric column for slice sizes (auto-detected if *None*).
            names_col: Categorical column for slice labels (auto-detected if *None*).
            title: Chart title (auto-generated if *None*).

        Returns:
            Plotly ``Figure`` with the rendered pie chart.
        """
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
    def create_histogram(df: pd.DataFrame, x_col: str | None = None, title: str | None = None,
                        nbins: int = 30) -> go.Figure:
        """Create a histogram for distribution analysis.

        Args:
            df: Source DataFrame.
            x_col: Numeric column to plot (auto-detected if *None*).
            title: Chart title (auto-generated if *None*).
            nbins: Number of histogram bins.

        Returns:
            Plotly ``Figure`` with the rendered histogram.
        """
        analysis = ChartTypeAnalyzer.analyze_data_structure(df)
        
        if x_col is None:
            x_col = analysis['numeric_columns'][0] if analysis['numeric_columns'] else df.columns[0]
        
        if title is None:
            title = f'Distribution of {x_col.title()}'
        
        fig = px.histogram(df, x=x_col, title=title, nbins=nbins)
        fig.update_layout(template='plotly_white', xaxis_title=x_col.title(), yaxis_title='Count')
        
        return fig
    
    @staticmethod
    def create_box_plot(df: pd.DataFrame, y_col: str | None = None, x_col: str | None = None,
                       title: str | None = None) -> go.Figure:
        """Create a box plot for statistical distribution analysis.

        Args:
            df: Source DataFrame.
            y_col: Numeric column for the distribution (auto-detected if *None*).
            x_col: Optional categorical column for grouping.
            title: Chart title (auto-generated if *None*).

        Returns:
            Plotly ``Figure`` with the rendered box plot.
        """
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
    def create_heatmap(df: pd.DataFrame, title: str | None = None) -> go.Figure:
        """Create a correlation heatmap for numeric columns.

        Args:
            df: Source DataFrame.
            title: Chart title (auto-generated if *None*).

        Returns:
            Plotly ``Figure`` with the rendered heatmap.
        """
        analysis = ChartTypeAnalyzer.analyze_data_structure(df)
        num_cols = analysis['numeric_columns']
        
        if len(num_cols) < 2:
            raise ValueError("Heatmap requires at least 2 numeric columns")
            
        corr_matrix = df[num_cols].corr()
        
        if title is None:
            title = 'Correlation Matrix'
            
        fig = px.imshow(corr_matrix, text_auto=True, title=title, aspect="auto")
        fig.update_layout(template='plotly_white')
        return fig

    @staticmethod
    def create_area_chart(df: pd.DataFrame, x_col: str | None = None, y_col: str | None = None,
                         color_col: str | None = None, title: str | None = None) -> go.Figure:
        """Create a stacked area chart.

        Args:
            df: Source DataFrame.
            x_col: Time/Sequence column.
            y_col: Value column.
            color_col: Grouping column for stacking.
            title: Chart title.

        Returns:
            Plotly ``Figure``.
        """
        analysis = ChartTypeAnalyzer.analyze_data_structure(df)
        
        if x_col is None:
            x_col = analysis['datetime_columns'][0] if analysis['datetime_columns'] else df.columns[0]
        if y_col is None:
            y_col = analysis['numeric_columns'][0] if analysis['numeric_columns'] else df.columns[-1]
            
        if title is None:
            title = f'{y_col.title()} (Stacked Area) by {x_col.title()}'
            
        fig = px.area(df, x=x_col, y=y_col, color=color_col, title=title)
        fig.update_layout(template='plotly_white')
        return fig

    @staticmethod
    def create_violin_plot(df: pd.DataFrame, y_col: str | None = None, x_col: str | None = None,
                          title: str | None = None) -> go.Figure:
        """Create a violin plot for distribution density.

        Args:
            df: Source DataFrame.
            y_col: Metric distribution.
            x_col: Optional grouping category.
            title: Chart title.

        Returns:
            Plotly ``Figure``.
        """
        analysis = ChartTypeAnalyzer.analyze_data_structure(df)
        
        if y_col is None:
            y_col = analysis['numeric_columns'][0] if analysis['numeric_columns'] else df.columns[-1]
            
        if title is None:
            title = f'{y_col.title()} Distribution Density'
            
        fig = px.violin(df, y=y_col, x=x_col, box=True, points="all", title=title)
        fig.update_layout(template='plotly_white')
        return fig

    @staticmethod
    def create_funnel_chart(df: pd.DataFrame, x_col: str | None = None, y_col: str | None = None,
                           title: str | None = None) -> go.Figure:
        """Create a funnel chart for process analysis.

        Args:
            df: Source DataFrame.
            x_col: Value column (width of funnel).
            y_col: Stage column (height/segments).
            title: Chart title.

        Returns:
            Plotly ``Figure``.
        """
        analysis = ChartTypeAnalyzer.analyze_data_structure(df)
        
        # Heuristic: y_col (stages) is usually categorical, x_col (values) is numeric
        if y_col is None:
            y_col = analysis['categorical_columns'][0] if analysis['categorical_columns'] else df.columns[0]
        if x_col is None:
            x_col = analysis['numeric_columns'][0] if analysis['numeric_columns'] else df.columns[-1]
            
        if title is None:
            title = f'Funnel Analysis: {y_col.title()}'
            
        fig = px.funnel(df, x=x_col, y=y_col, title=title)
        fig.update_layout(template='plotly_white')
        return fig

    @staticmethod
    def create_treemap(df: pd.DataFrame, path_cols: list[str] | None = None, values_col: str | None = None,
                      title: str | None = None) -> go.Figure:
        """Create a treemap for hierarchical data.

        Args:
            df: Source DataFrame.
            path_cols: List of categorical columns for hierarchy.
            values_col: Numeric column for box size.
            title: Chart title.

        Returns:
            Plotly ``Figure``.
        """
        analysis = ChartTypeAnalyzer.analyze_data_structure(df)
        
        if path_cols is None:
            path_cols = analysis['categorical_columns'][:2]
            
        if values_col is None:
            values_col = analysis['numeric_columns'][0] if analysis['numeric_columns'] else None
            
        if title is None:
            title = f'Hierarchical View ({", ".join(path_cols)})'
            
        fig = px.treemap(df, path=path_cols, values=values_col, title=title)
        fig.update_layout(template='plotly_white')
        return fig

    @staticmethod
    def create_sunburst(df: pd.DataFrame, path_cols: list[str] | None = None, values_col: str | None = None,
                       title: str | None = None) -> go.Figure:
        """Create a sunburst chart for radial hierarchy.

        Args:
            df: Source DataFrame.
            path_cols: List of categorical columns for hierarchy.
            values_col: Numeric column for segment size.
            title: Chart title.

        Returns:
            Plotly ``Figure``.
        """
        analysis = ChartTypeAnalyzer.analyze_data_structure(df)
        
        if path_cols is None:
            path_cols = analysis['categorical_columns'][:2]
            
        if values_col is None:
            values_col = analysis['numeric_columns'][0] if analysis['numeric_columns'] else None
            
        if title is None:
            title = f'Radial Hierarchy ({", ".join(path_cols)})'
            
        fig = px.sunburst(df, path=path_cols, values=values_col, title=title)
        fig.update_layout(template='plotly_white')
        return fig

    @staticmethod
    def create_bubble_chart(df: pd.DataFrame, x_col: str | None = None, y_col: str | None = None,
                           size_col: str | None = None, color_col: str | None = None, title: str | None = None) -> go.Figure:
        """Create a bubble chart (3D scatter).

        Args:
            df: Source DataFrame.
            x_col: X-axis metric.
            y_col: Y-axis metric.
            size_col: Bubble size metric.
            color_col: Optional grouping/color metric.
            title: Chart title.

        Returns:
            Plotly ``Figure``.
        """
        analysis = ChartTypeAnalyzer.analyze_data_structure(df)
        
        num_cols = analysis['numeric_columns']
        if x_col is None:
            x_col = num_cols[0] if len(num_cols) > 0 else df.columns[0]
        if y_col is None:
            y_col = num_cols[1] if len(num_cols) > 1 else df.columns[-1]
        if size_col is None:
            size_col = num_cols[2] if len(num_cols) > 2 else y_col
            
        if title is None:
            title = f'{y_col.title()} vs {x_col.title()} (Size: {size_col.title()})'
            
        fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=color_col, title=title)
        fig.update_layout(template='plotly_white')
        return fig

    @staticmethod
    def create_chart(df: pd.DataFrame, chart_type: str, **kwargs: Any) -> go.Figure:
        """Route chart creation to the appropriate generator method.

        Args:
            df: Source DataFrame.
            chart_type: One of ``bar``, ``line``, ``scatter``, ``pie``,
                ``histogram``, ``box``, ``heatmap``, ``area``, ``violin``,
                ``funnel``, ``treemap``, ``sunburst``, ``bubble``.
            **kwargs: Forwarded to the underlying chart method.

        Returns:
            Plotly ``Figure`` for the requested chart type.

        Raises:
            ValueError: If *chart_type* is not supported.
        """
        generators = {
            'bar': DynamicChartGenerator.create_bar_chart,
            'line': DynamicChartGenerator.create_line_chart,
            'scatter': DynamicChartGenerator.create_scatter_chart,
            'pie': DynamicChartGenerator.create_pie_chart,
            'histogram': DynamicChartGenerator.create_histogram,
            'box': DynamicChartGenerator.create_box_plot,
            'heatmap': DynamicChartGenerator.create_heatmap,
            'area': DynamicChartGenerator.create_area_chart,
            'violin': DynamicChartGenerator.create_violin_plot,
            'funnel': DynamicChartGenerator.create_funnel_chart,
            'treemap': DynamicChartGenerator.create_treemap,
            'sunburst': DynamicChartGenerator.create_sunburst,
            'bubble': DynamicChartGenerator.create_bubble_chart,
        }
        
        if chart_type not in generators:
            # Fallback for old/unknown types to simpler ones or error
            if chart_type == 'interactive_graph':
                 # Route to line or scatter generically
                 return DynamicChartGenerator.create_scatter_chart(df, **kwargs)
            raise ValueError(f"Unsupported chart type: {chart_type}. Choose from: {list(generators.keys())}")
        
        return generators[chart_type](df, **kwargs)


# =====================================================================
# v2.0 Enterprise Additions — appended; all v1.x code is unchanged
# =====================================================================

import threading
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class ChartGenerationMetrics:
    """Tracks chart creation counts grouped by chart type.

    v2.0 Enterprise Addition.
    """

    total_charts: int = 0
    charts_by_type: Counter = field(default_factory=Counter)

    def record(self, chart_type: str) -> None:
        self.total_charts += 1
        self.charts_by_type[chart_type] += 1

    def to_dict(self) -> dict:
        return {
            "total_charts": self.total_charts,
            "charts_by_type": dict(self.charts_by_type),
        }


_chart_generator_instance: "DynamicChartGenerator | None" = None
_chart_generator_lock = threading.Lock()


def get_chart_generator() -> "DynamicChartGenerator":
    """Return the process-wide :class:`DynamicChartGenerator` singleton.

    Thread Safety:
        Uses double-checked locking for safe lazy initialisation.
    """
    global _chart_generator_instance
    if _chart_generator_instance is None:
        with _chart_generator_lock:
            if _chart_generator_instance is None:
                _chart_generator_instance = DynamicChartGenerator()
    return _chart_generator_instance

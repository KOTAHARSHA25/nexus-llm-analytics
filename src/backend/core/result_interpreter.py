# Result Interpreter Utility
# Provides domain-agnostic, human-readable interpretation of analysis results

import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np

class ResultInterpreter:
    """
    Domain-agnostic result interpreter that converts analysis results 
    into human-readable, meaningful text.
    
    This class provides consistent formatting across all agents while
    preserving all important information in a readable format.
    """
    
    @staticmethod
    def interpret(
        result: Dict[str, Any],
        query: str = "",
        agent_name: str = "",
        operation: str = ""
    ) -> str:
        """
        Main interpretation entry point.
        
        Args:
            result: The raw result dictionary from an agent
            query: The original user query (for context)
            agent_name: Name of the agent that produced the result
            operation: The operation performed
            
        Returns:
            Human-readable interpretation string
        """
        lines = []
        
        # Header
        if operation:
            lines.append(f"## {ResultInterpreter._format_operation_name(operation)} Results\n")
        
        # Process the result based on its structure
        if isinstance(result, dict):
            lines.extend(ResultInterpreter._interpret_dict(result, query, indent=0))
        elif isinstance(result, (list, tuple)):
            lines.extend(ResultInterpreter._interpret_list(result, indent=0))
        else:
            lines.append(str(result))
        
        interpretation = "\n".join(lines)
        
        # If interpretation is too short, provide a summary
        if len(interpretation.strip()) < 50:
            interpretation = f"Analysis completed successfully. {interpretation}"
        
        return interpretation
    
    @staticmethod
    def _format_operation_name(operation: str) -> str:
        """Convert operation_name to Operation Name"""
        return operation.replace('_', ' ').title()
    
    @staticmethod
    def _format_key(key: str) -> str:
        """Convert key_name to readable Key Name"""
        # Skip internal/metadata keys
        if key.startswith('_') or key in ['success', 'agent', 'operation', 'metadata']:
            return ""
        return key.replace('_', ' ').title()
    
    @staticmethod
    def _format_value(value: Any, precision: int = 2) -> str:
        """Format a value for human readability"""
        if value is None:
            return "Not available"
        elif isinstance(value, bool):
            return "Yes" if value else "No"
        elif isinstance(value, float):
            if abs(value) >= 1000000:
                return f"{value:,.0f}"
            elif abs(value) >= 100:
                return f"{value:,.{precision}f}"
            elif abs(value) < 0.01 and value != 0:
                return f"{value:.4f}"
            else:
                return f"{value:.{precision}f}"
        elif isinstance(value, int):
            return f"{value:,}"
        elif isinstance(value, str):
            return value
        elif isinstance(value, (list, tuple)):
            if len(value) <= 5:
                return ", ".join(str(v) for v in value)
            else:
                return f"{', '.join(str(v) for v in value[:5])}... ({len(value)} total)"
        else:
            return str(value)
    
    @staticmethod
    def _interpret_dict(data: Dict[str, Any], query: str = "", indent: int = 0) -> List[str]:
        """Recursively interpret a dictionary into readable lines"""
        lines = []
        prefix = "  " * indent
        
        # Skip metadata keys
        skip_keys = {'success', 'agent', 'operation', 'metadata', 'error', 'type'}
        
        # First, handle special known structures
        if 'summary' in data or 'overview' in data:
            summary = data.get('summary') or data.get('overview')
            if isinstance(summary, str):
                lines.append(f"{prefix}**Summary:** {summary}\n")
        
        # Process each key-value pair
        for key, value in data.items():
            if key in skip_keys or key.startswith('_'):
                continue
            
            formatted_key = ResultInterpreter._format_key(key)
            if not formatted_key:
                continue
            
            if isinstance(value, dict):
                # Check if it's a simple metrics dict (all scalar values)
                if all(isinstance(v, (int, float, str, bool, type(None))) for v in value.values()):
                    lines.append(f"{prefix}### {formatted_key}")
                    for k, v in value.items():
                        sub_key = ResultInterpreter._format_key(k)
                        if sub_key:
                            lines.append(f"{prefix}• **{sub_key}:** {ResultInterpreter._format_value(v)}")
                    lines.append("")
                else:
                    # Nested structure
                    lines.append(f"{prefix}### {formatted_key}")
                    lines.extend(ResultInterpreter._interpret_dict(value, query, indent + 1))
            
            elif isinstance(value, (list, tuple)):
                lines.append(f"{prefix}### {formatted_key}")
                lines.extend(ResultInterpreter._interpret_list(value, indent + 1))
            
            elif isinstance(value, (int, float)):
                # Format numbers nicely
                lines.append(f"{prefix}• **{formatted_key}:** {ResultInterpreter._format_value(value)}")
            
            elif isinstance(value, str):
                if len(value) > 100:
                    lines.append(f"{prefix}**{formatted_key}:**")
                    lines.append(f"{prefix}{value}")
                else:
                    lines.append(f"{prefix}• **{formatted_key}:** {value}")
            
            elif isinstance(value, bool):
                lines.append(f"{prefix}• **{formatted_key}:** {'Yes' if value else 'No'}")
            
            elif value is not None:
                lines.append(f"{prefix}• **{formatted_key}:** {ResultInterpreter._format_value(value)}")
        
        return lines
    
    @staticmethod
    def _interpret_list(data: List[Any], indent: int = 0) -> List[str]:
        """Interpret a list into readable lines"""
        lines = []
        prefix = "  " * indent
        
        if not data:
            lines.append(f"{prefix}No items found.")
            return lines
        
        # Check if list contains dicts (table-like data)
        if all(isinstance(item, dict) for item in data):
            for i, item in enumerate(data[:10], 1):  # Limit to first 10
                lines.append(f"{prefix}{i}.")
                for k, v in item.items():
                    formatted_key = ResultInterpreter._format_key(k)
                    if formatted_key:
                        lines.append(f"{prefix}   • **{formatted_key}:** {ResultInterpreter._format_value(v)}")
            if len(data) > 10:
                lines.append(f"{prefix}... and {len(data) - 10} more items")
        else:
            # Simple list
            for item in data[:20]:
                lines.append(f"{prefix}• {ResultInterpreter._format_value(item)}")
            if len(data) > 20:
                lines.append(f"{prefix}... and {len(data) - 20} more items")
        
        return lines
    
    @staticmethod
    def generate_data_summary(
        data_info: Dict[str, Any],
        numeric_summary: Dict[str, Any] = None,
        categorical_summary: Dict[str, Any] = None,
        query: str = ""
    ) -> str:
        """
        Generate a comprehensive, query-aware data summary.
        
        This is the main method for creating human-readable analysis results.
        """
        lines = []
        query_lower = query.lower() if query else ""
        
        # Dataset overview
        shape = data_info.get("shape", (0, 0))
        lines.append("## Analysis Results\n")
        lines.append(f"**Dataset:** {shape[0]:,} records across {shape[1]} columns")
        
        # Missing data
        missing = data_info.get("missing_values", {})
        total_missing = sum(missing.values()) if missing else 0
        if total_missing > 0:
            lines.append(f"**Data Quality:** {total_missing:,} missing values detected")
        else:
            lines.append("**Data Quality:** Complete (no missing values)")
        
        lines.append("")
        
        # Answer common query patterns
        if numeric_summary:
            # Check what the user is asking for
            wants_total = any(t in query_lower for t in ["total", "sum", "all", "overall"])
            wants_average = any(t in query_lower for t in ["average", "mean", "typical"])
            wants_highest = any(t in query_lower for t in ["highest", "maximum", "max", "top", "best"])
            wants_lowest = any(t in query_lower for t in ["lowest", "minimum", "min", "bottom", "worst"])
            wants_count = any(t in query_lower for t in ["how many", "count", "number of"])
            
            lines.append("### Key Metrics\n")
            
            for col, stats in numeric_summary.items():
                col_display = col.replace('_', ' ').title()
                
                # Always show totals prominently for numeric columns
                if 'sum' in stats:
                    lines.append(f"**Total {col_display}:** {stats['sum']:,.2f}")
                
                lines.append(f"**{col_display}:**")
                
                if wants_total or wants_average or not (wants_highest or wants_lowest):
                    if 'mean' in stats:
                        lines.append(f"  • Average: {stats['mean']:,.2f}")
                    if 'median' in stats:
                        lines.append(f"  • Median: {stats['median']:,.2f}")
                
                if wants_highest or not (wants_total or wants_average):
                    if 'max' in stats:
                        lines.append(f"  • Maximum: {stats['max']:,.2f}")
                
                if wants_lowest or not (wants_total or wants_average):
                    if 'min' in stats:
                        lines.append(f"  • Minimum: {stats['min']:,.2f}")
                
                if 'count' in stats:
                    lines.append(f"  • Count: {stats['count']:,}")
                
                if 'std' in stats and stats['std'] > 0:
                    lines.append(f"  • Std Dev: {stats['std']:,.2f}")
                
                lines.append("")
        
        # Categorical breakdown
        if categorical_summary:
            lines.append("### Category Breakdown\n")
            
            for col, cat_stats in categorical_summary.items():
                col_display = col.replace('_', ' ').title()
                lines.append(f"**{col_display}:**")
                lines.append(f"  • Unique values: {cat_stats.get('unique_count', 'N/A')}")
                
                if 'most_frequent' in cat_stats:
                    lines.append(f"  • Most common: {cat_stats['most_frequent']} ({cat_stats.get('most_frequent_count', 'N/A'):,} occurrences)")
                
                # Show value distribution
                value_counts = cat_stats.get('value_counts', {})
                if value_counts:
                    lines.append("  • Distribution:")
                    for val, count in list(value_counts.items())[:8]:
                        lines.append(f"    - {val}: {count:,}")
                    if len(value_counts) > 8:
                        lines.append(f"    - ... and {len(value_counts) - 8} more categories")
                
                lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def generate_grouped_summary(
        data,  # DataFrame
        group_col: str,
        value_col: str,
        query: str = ""
    ) -> str:
        """Generate a summary of data grouped by a category"""
        lines = []
        query_lower = query.lower() if query else ""
        
        try:
            import pandas as pd
            
            if not isinstance(data, pd.DataFrame):
                return "Unable to generate grouped summary - invalid data format"
            
            # Perform grouping
            grouped = data.groupby(group_col)[value_col].agg(['sum', 'mean', 'count', 'min', 'max'])
            grouped = grouped.sort_values('sum', ascending=False)
            
            lines.append(f"### {value_col.replace('_', ' ').title()} by {group_col.replace('_', ' ').title()}\n")
            
            # Find highest/lowest
            highest_group = grouped['sum'].idxmax()
            lowest_group = grouped['sum'].idxmin()
            total_sum = grouped['sum'].sum()
            
            lines.append(f"**Total {value_col.replace('_', ' ').title()}:** {total_sum:,.2f}")
            lines.append(f"**Highest:** {highest_group} ({grouped.loc[highest_group, 'sum']:,.2f})")
            lines.append(f"**Lowest:** {lowest_group} ({grouped.loc[lowest_group, 'sum']:,.2f})")
            lines.append("")
            
            # Show all groups
            lines.append("**Breakdown:**")
            for group_name, row in grouped.iterrows():
                pct = (row['sum'] / total_sum * 100) if total_sum > 0 else 0
                lines.append(f"• **{group_name}:** {row['sum']:,.2f} ({pct:.1f}%)")
                lines.append(f"  - Average: {row['mean']:,.2f}, Count: {row['count']:,.0f}")
            
            return "\n".join(lines)
            
        except Exception as e:
            logging.error(f"Grouped summary generation failed: {e}")
            return f"Unable to generate grouped summary: {str(e)}"


def interpret_result(result: Dict[str, Any], query: str = "", agent_name: str = "") -> str:
    """
    Convenience function to interpret any agent result.
    
    This is the main function that should be called from agents.
    """
    return ResultInterpreter.interpret(
        result=result,
        query=query,
        agent_name=agent_name,
        operation=result.get('operation', '')
    )


def generate_analysis_interpretation(
    data_info: Dict,
    numeric_summary: Dict = None,
    categorical_summary: Dict = None,
    query: str = ""
) -> str:
    """
    Convenience function to generate data analysis interpretation.
    
    Use this for statistical/data analysis results.
    """
    return ResultInterpreter.generate_data_summary(
        data_info=data_info,
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        query=query
    )

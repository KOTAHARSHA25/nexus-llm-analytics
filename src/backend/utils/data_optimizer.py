"""
Data Optimizer Utility for LLM Consumption

Prepares complex data structures for efficient LLM processing by:
1. Flattening nested JSON/dict structures
2. Sampling large datasets
3. Generating schema summaries
4. Creating statistical overviews

Author: Nexus LLM Analytics Team
Date: October 19, 2025
"""

import json
import pandas as pd
from typing import Dict, Any, List, Union, Tuple
from pathlib import Path
import numpy as np
import logging


class DataOptimizer:
    """Optimizes data for LLM consumption"""
    
    def __init__(self, max_rows: int = 100, max_depth: int = 3, max_chars: int = 8000):
        """
        Initialize data optimizer
        
        Args:
            max_rows: Maximum number of rows to sample
            max_depth: Maximum nesting depth to preserve
            max_chars: Maximum characters in preview (Def: 8000 for better context)
        """
        self.max_rows = max_rows
        self.max_depth = max_depth
        self.max_chars = max_chars
    
    def optimize_for_llm(self, filepath: str, file_type: str = None) -> Dict[str, Any]:
        """
        Main optimization function - prepares data for LLM
        
        Args:
            filepath: Path to data file
            file_type: Type of file (json, csv, excel)
            
        Returns:
            Dict with: schema, sample, stats, preview
        """
        filepath = Path(filepath)
        
        # Auto-detect file type if not provided
        if file_type is None:
            file_type = self._detect_file_type(filepath)
        
        try:
            # Load and optimize based on type
            if file_type == 'json':
                return self._optimize_json(filepath)
            elif file_type == 'csv':
                return self._optimize_csv(filepath)
            elif file_type in ['excel', 'xlsx', 'xls']:
                return self._optimize_excel(filepath)
            else:
                return self._basic_load(filepath)
                
        except Exception as e:
            logging.warning(f"Failed to parse {filepath.name} as {file_type}: {e}. Falling back to text mode.")
            return self._basic_load(filepath, error_context=str(e))
    
    def _detect_file_type(self, filepath: Path) -> str:
        """Detect file type from extension"""
        ext = filepath.suffix.lower()
        type_map = {
            '.json': 'json',
            '.csv': 'csv',
            '.xlsx': 'excel',
            '.xls': 'excel',
        }
        return type_map.get(ext, 'unknown')
    
    def _optimize_json(self, filepath: Path) -> Dict[str, Any]:
        """Optimize JSON data for LLM with robust error handling"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle empty data
        if not data:
            raise ValueError("Empty JSON file - no data to analyze")
        
        # Handle empty lists
        if isinstance(data, list) and len(data) == 0:
            raise ValueError("Empty JSON array - no records to analyze")
        
        # Handle empty objects
        if isinstance(data, dict) and len(data) == 0:
            raise ValueError("Empty JSON object - no data to analyze")
        
        # Check if nested structure
        is_nested = self._is_nested(data)
        
        # Always try to flatten to extract simple nested structures
        flattened = self._flatten_nested_json(data)
        
        # Convert to DataFrame for analysis
        if isinstance(flattened, list):
            if len(flattened) == 0:
                raise ValueError("No records after flattening - invalid structure")
            df = pd.DataFrame(flattened)
        else:
            df = pd.DataFrame([flattened])
        
        # Check for completely empty DataFrame
        if df.empty or len(df.columns) == 0:
            raise ValueError("No analyzable data found in JSON file")
        
        # Generate optimized output
        return self._generate_optimized_output(df, 'json', is_nested)
    
    def _optimize_csv(self, filepath: Path) -> Dict[str, Any]:
        """Optimize CSV data for LLM with special type handling"""
        df = pd.read_csv(filepath)
        # Clean and convert special types (currency, percentages, dates)
        df = self._clean_special_types(df)
        return self._generate_optimized_output(df, 'csv', False)
    
    def _optimize_excel(self, filepath: Path) -> Dict[str, Any]:
        """Optimize Excel data for LLM"""
        df = pd.read_excel(filepath)
        return self._generate_optimized_output(df, 'excel', False)
    
    def _clean_special_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and convert special data types to proper numeric/datetime types
        
        Handles:
        - Currency: $1,234.56 → 1234.56
        - Percentages: 15% → 0.15 (or 15.0 depending on usage)
        - Dates: Various formats → datetime
        
        Args:
            df: DataFrame with potentially messy types
            
        Returns:
            DataFrame with cleaned types
        """
        df = df.copy()  # Don't modify original
        
        for col in df.columns:
            # Skip if already numeric or datetime
            if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            
            # Try to detect and convert special types
            if df[col].dtype == 'object':
                # Get non-null sample
                sample = df[col].dropna().head(10) if len(df[col].dropna()) > 0 else pd.Series()
                
                if len(sample) == 0:
                    continue
                
                # Check if currency format (contains $ and numbers)
                sample_str = sample.astype(str)
                if sample_str.str.contains(r'\$', regex=True).any():
                    try:
                        # Remove $, commas, and convert to float
                        df[col] = df[col].str.replace('$', '', regex=False)
                        df[col] = df[col].str.replace(',', '', regex=False)
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        logging.info(f"✓ Converted currency column: '{col}' → numeric")
                        continue
                    except:
                        pass
                
                # Check if percentage format (contains % and numbers)
                if sample_str.str.contains(r'%', regex=True).any():
                    try:
                        # Remove % and convert to float (keep as percentage number, not decimal)
                        # E.g., "15%" → 15.0 (not 0.15) for easier LLM understanding
                        df[col] = df[col].str.replace('%', '', regex=False)
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        logging.info(f"✓ Converted percentage column: '{col}' → numeric (kept as %)")
                        continue
                    except:
                        pass
                
                # Check if date format (already handled by _detect_and_convert_dates, but double-check)
                if sample_str.str.contains(r'-|/|\d{4}', regex=True).any():
                    try:
                        # infer_datetime_format is deprecated - pandas now does this by default
                        converted = pd.to_datetime(df[col], errors='coerce')
                        # Only convert if >50% of values are valid dates
                        if converted.notna().sum() / len(df) > 0.5:
                            df[col] = converted
                            logging.info(f"✓ Converted date column: '{col}' → datetime")
                            continue
                    except:
                        pass
        
        return df
    
    def _basic_load(self, filepath: Path, error_context: str = None) -> Dict[str, Any]:
        """Basic file loading for unknown types or failed parses"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Simple stats for text
            lines = content.splitlines()
            line_count = len(lines)
            non_empty_lines = sum(1 for line in lines if line.strip())
            
            preview = f"File: {filepath.name} (treated as unstructured text)\n"
            if error_context:
                preview += f"⚠️ Note: Structured parsing failed: {error_context}\n"
            preview += f"Size: {len(content)} bytes, {line_count} lines\n"
            preview += "="*50 + "\n"
            preview += content[:self.max_chars]
            
            if len(content) > self.max_chars:
                preview += "\n... (truncated)"

            return {
                'schema': 'Unstructured Text / Unknown Format',
                'sample': [{'text': line} for line in lines[:10] if line.strip()],
                'stats': {
                    'file_size': len(content),
                    'line_count': line_count,
                    'non_empty_lines': non_empty_lines,
                    'is_fallback': True
                },
                'preview': preview,
                'is_optimized': False,
                'was_nested': False,
                'total_rows': line_count,
                'total_columns': 1,
                'file_type': 'text'
            }
        except Exception as e:
            # Absolute last resort
            return {
                'schema': 'Unreadable File',
                'sample': [],
                'stats': {'error': str(e)},
                'preview': f"Could not read file: {e}",
                'is_optimized': False,
                'total_rows': 0
            }
    
    def _is_nested(self, data: Any, current_depth: int = 0) -> bool:
        """Check if JSON structure is deeply nested"""
        if current_depth > 2:  # More than 2 levels is considered nested
            return True
        
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, (dict, list)):
                    if self._is_nested(value, current_depth + 1):
                        return True
        elif isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], (dict, list)):
                if self._is_nested(data[0], current_depth + 1):
                    return True
        
        return False
    
    def _flatten_nested_json(self, data: Any, parent_key: str = '', sep: str = '_') -> Union[Dict, List]:
        """
        Flatten nested JSON structure
        
        Args:
            data: Nested JSON data
            parent_key: Parent key for recursive flattening
            sep: Separator for flattened keys
            
        Returns:
            Flattened dictionary or list of dictionaries
        """
        # Special case: If root dict has single key pointing to list of dicts, extract that list
        if isinstance(data, dict) and len(data) == 1 and not parent_key:
            key, value = next(iter(data.items()))
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                # This is a simple nested structure like {"sales_data": [{...}, {...}]}
                # Return the list directly so pandas can convert it to a proper DataFrame
                return value
        
        if isinstance(data, list):
            # If list, flatten each item
            return [self._flatten_dict(item, parent_key, sep) if isinstance(item, dict) else item 
                    for item in data]
        elif isinstance(data, dict):
            return self._flatten_dict(data, parent_key, sep)
        else:
            return data
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten a nested dictionary with robust null and special character handling"""
        items = []
        for k, v in d.items():
            # Sanitize key names for DataFrame compatibility
            # Replace dashes, dots, spaces with underscores
            sanitized_key = str(k).replace('-', '_').replace('.', '_').replace(' ', '_')
            new_key = f"{parent_key}{sep}{sanitized_key}" if parent_key else sanitized_key
            
            # Handle null/None values
            if v is None:
                items.append((new_key, None))
            elif isinstance(v, dict):
                # Handle empty dicts
                if len(v) == 0:
                    items.append((new_key, None))
                else:
                    items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, list):
                # Handle empty lists
                if len(v) == 0:
                    items.append((new_key, None))
                # Handle lists - convert to string if complex
                elif isinstance(v[0], dict):
                    # List of dicts - flatten and number them
                    for i, item in enumerate(v):
                        items.extend(self._flatten_dict(item, f"{new_key}_{i}", sep).items())
                elif isinstance(v[0], list):
                    # Nested arrays - convert to string representation
                    items.append((new_key, str(v)))
                else:
                    # Simple list - keep as is (but limit length)
                    items.append((new_key, str(v) if len(str(v)) < 100 else f"{len(v)} items"))
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def _generate_optimized_output(self, df: pd.DataFrame, file_type: str, is_nested: bool) -> Dict[str, Any]:
        """Generate optimized output for LLM with enhanced type detection"""
        
        # Auto-detect and convert date columns
        df = self._detect_and_convert_dates(df)
        
        # Infer better types for mixed columns
        df = df.infer_objects()
        
        # Schema information
        schema = self._generate_schema(df)
        
        # Statistical summary
        stats = self._generate_stats(df)
        
        # Sample data (limited rows)
        sample = self._generate_sample(df)
        
        # Preview string (for LLM context)
        preview = self._generate_preview(df, schema, stats)
        
        return {
            'schema': schema,
            'sample': sample,
            'stats': stats,
            'preview': preview,
            'is_optimized': True,
            'was_nested': is_nested,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'file_type': file_type
        }
    
    def _detect_and_convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and convert date/timestamp columns"""
        for col in df.columns:
            # Skip if already datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            
            # Try to convert if column looks like dates
            try:
                # Sample first non-null value
                sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if sample is None:
                    continue
                
                # Check if it looks like a date
                sample_str = str(sample)
                if any(indicator in sample_str for indicator in ['-', '/', 'T', ':', 'AM', 'PM', 'Z']):
                    # Try to parse as datetime
                    df[col] = pd.to_datetime(df[col], errors='ignore', infer_datetime_format=True)
            except:
                # If conversion fails, keep original
                pass
        
        return df
    
    def _generate_schema(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate schema description"""
        schema = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            
            # Handle columns with unhashable types (lists, dicts, etc.)
            try:
                unique_count = df[col].nunique()
            except TypeError:
                # Column contains unhashable types (nested structures)
                unique_count = -1  # Indicate nested/complex data
            
            schema[col] = {
                'type': dtype,
                'null_count': int(null_count),
                'unique_values': int(unique_count),
                'sample_values': df[col].dropna().head(3).tolist() if len(df) > 0 else []
            }
        
        return schema
    
    def _generate_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical summary"""
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_rows': int(df.duplicated().sum()),
        }
        
        # Numeric statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Categorical statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            stats['categorical_summary'] = {}
            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                try:
                    stats['categorical_summary'][col] = {
                        'unique_count': int(df[col].nunique()),
                        'top_values': df[col].value_counts().head(3).to_dict()
                    }
                except TypeError:
                    # Column contains unhashable types (nested structures)
                    stats['categorical_summary'][col] = {
                        'unique_count': -1,
                        'top_values': {}
                    }
        
        return stats
    
    def _generate_sample(self, df: pd.DataFrame) -> List[Dict]:
        """Generate sample rows"""
        sample_size = min(self.max_rows, len(df))
        
        if len(df) <= sample_size:
            # Return all rows if small dataset
            return df.to_dict('records')
        else:
            # Sample from beginning, middle, and end
            start_rows = sample_size // 3
            mid_rows = sample_size // 3
            end_rows = sample_size - start_rows - mid_rows
            
            mid_index = len(df) // 2
            
            sample_df = pd.concat([
                df.head(start_rows),
                df.iloc[mid_index:mid_index + mid_rows],
                df.tail(end_rows)
            ])
            
            return sample_df.to_dict('records')
    
    def _generate_preview(self, df: pd.DataFrame, schema: Dict, stats: Dict) -> str:
        """Generate human-readable preview for LLM - adapts to data size and complexity"""
        preview_parts = []
        
        # Detect if this is a small, simple dataset
        is_small = len(df) <= 10
        is_simple = len(df.columns) <= 5
        has_numeric = len(df.select_dtypes(include=[np.number]).columns) > 0
        
        # FOR SMALL, SIMPLE DATASETS: Use minimal, clear format
        if is_small and is_simple:
            preview_parts.append(f"Data from file (Total: {len(df)} row{'s' if len(df) != 1 else ''}, {len(df.columns)} column{'s' if len(df.columns) != 1 else ''}):")
            preview_parts.append(f"")
            
            # Show ALL rows for small datasets (not just 3)
            preview_parts.append(df.to_string(index=False, max_cols=len(df.columns)))
            preview_parts.append(f"")
            
            # Simple column info
            preview_parts.append(f"Columns:")
            for col in df.columns:
                dtype = str(df[col].dtype)
                unique_count = df[col].nunique()
                preview_parts.append(f"- {col}: {dtype} ({unique_count} unique value{'s' if unique_count != 1 else ''})")
            
            return '\n'.join(preview_parts)
        
        # FOR LARGE/COMPLEX DATASETS: Use detailed statistics approach
        # Basic info
        preview_parts.append(f"Dataset Overview:")
        preview_parts.append(f"- Total Rows: {stats['total_rows']}")
        preview_parts.append(f"- Total Columns: {stats['total_columns']}")
        preview_parts.append(f"- Columns: {', '.join(stats['columns'][:10])}")
        if stats['total_columns'] > 10:
            preview_parts.append(f"  ... and {stats['total_columns'] - 10} more columns")
        
        # PRE-CALCULATED AGGREGATIONS (for accurate answers)
        preview_parts.append(f"\n{'='*70}")
        preview_parts.append(f"⚡ PRE-CALCULATED STATISTICS - COMPUTED FROM ALL {len(df):,} ROWS")
        preview_parts.append(f"{'='*70}")
        preview_parts.append(f"⚠️  CRITICAL: DO NOT calculate these yourself! Use these exact values!")
        preview_parts.append(f"")
        
        # Numeric aggregations with DYNAMIC prioritization
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # FIXED: Show OVERALL TOTALS for ALL numeric columns first (critical for accuracy)
        # This ensures questions like "total revenue?" get answered correctly
        preview_parts.append(f"\n🔢 OVERALL COLUMN STATISTICS (computed from ALL {len(df)} rows):")
        preview_parts.append(f"-" * 50)
        preview_parts.append(f"USE THESE FOR: 'total X?', 'average X?', 'maximum X?', 'minimum X?'")
        preview_parts.append(f"")
        for col in numeric_cols:
            try:
                total = df[col].sum()
                avg = df[col].mean()
                min_val = df[col].min()
                max_val = df[col].max()
                preview_parts.append(f"📊 {col.upper()}:")
                preview_parts.append(f"   • TOTAL (sum of all values): {total:,.2f}" if isinstance(total, float) else f"   • TOTAL (sum of all values): {total:,}")
                preview_parts.append(f"   • AVERAGE (mean): {avg:,.2f}" if isinstance(avg, float) else f"   • AVERAGE (mean): {avg:,}")
                preview_parts.append(f"   • MINIMUM (smallest single value): {min_val:,.2f}" if isinstance(min_val, float) else f"   • MINIMUM (smallest single value): {min_val:,}")
                preview_parts.append(f"   • MAXIMUM (largest single value): {max_val:,.2f}" if isinstance(max_val, float) else f"   • MAXIMUM (largest single value): {max_val:,}")
            except:
                pass
        preview_parts.append(f"")
        
        # DYNAMIC filtering: Identify meaningful numeric columns for detailed stats
        meaningful_numeric_cols = []
        
        for col in numeric_cols:
            col_lower = col.lower()
            
            # Skip ID-like columns (high uniqueness, low value for aggregation)
            try:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.95:  # 95%+ unique = likely an ID
                    continue
            except:
                pass
            
            # Skip year-only columns (years 1900-2100)
            try:
                col_min = df[col].min()
                col_max = df[col].max()
                if col_min >= 1900 and col_max <= 2100 and df[col].nunique() <= 200:
                    continue  # Likely year column
            except:
                pass
            
            meaningful_numeric_cols.append(col)
        
        # Show ALL meaningful numeric columns (up to 10)
        for col in meaningful_numeric_cols[:10]:
            try:
                total = df[col].sum()
                avg = df[col].mean()
                min_val = df[col].min()
                max_val = df[col].max()
                count = df[col].count()
                preview_parts.append(f"📊 {col.upper()}:")
                preview_parts.append(f"   • Total Sum: {total:,.2f}" if isinstance(total, float) else f"   • Total Sum: {total:,}")
                preview_parts.append(f"   • Average/Mean: {avg:,.2f}" if isinstance(avg, float) else f"   • Average/Mean: {avg:,}")
                preview_parts.append(f"   • Minimum: {min_val:,.2f}, Maximum: {max_val:,.2f}" if isinstance(min_val, float) else f"   • Minimum: {min_val}, Maximum: {max_val}")
                preview_parts.append(f"   • Count: {count:,} values")
                preview_parts.append(f"")
            except:
                pass
        
        # Categorical aggregations (counts) - DYNAMIC detection
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Filter out ID-like columns using DYNAMIC heuristics (no keywords)
        meaningful_cols = []
        for col in categorical_cols:
            col_lower = col.lower()
            unique_count = df[col].nunique()
            total_rows = len(df)
            
            # Skip ID columns: high uniqueness (>90% unique values)
            is_id_like = unique_count / total_rows > 0.9
            
            # Skip date/time columns: check if values match common date patterns
            is_date_like = False
            if not is_id_like:
                sample_values = df[col].dropna().head(5).astype(str)
                # Basic pattern check: contains dashes/slashes with numbers
                is_date_like = any('-' in str(v) or '/' in str(v) for v in sample_values)
            
            if not is_id_like and not is_date_like:
                # Add all valid categorical columns (no prioritization by name)
                meaningful_cols.append(col)
        
        # Show top 5 meaningful categorical columns
        for col in meaningful_cols[:5]:
            try:
                value_counts = df[col].value_counts().head(10)  # Show top 10 for better coverage
                # Only show if there are meaningful categories (not all unique)
                if len(value_counts) > 1 and value_counts.iloc[0] > 1:
                    preview_parts.append(f"📋 {col.upper()} - Top 10 Categories:")
                    for val, count in value_counts.items():
                        pct = (count / len(df)) * 100
                        preview_parts.append(f"   • {val}: {count:,} occurrences ({pct:.1f}%)")
                    preview_parts.append(f"")
            except:
                pass
        
        # GROUPED AGGREGATIONS - DYNAMIC DETECTION (no hardcoded keywords)
        # Detect potential grouping columns based on cardinality and data patterns
        grouping_cols = []
        for col in categorical_cols:
            try:
                unique_count = df[col].nunique()
                total_rows = len(df)
                
                # DYNAMIC HEURISTICS for grouping columns:
                # 1. Has reasonable cardinality (2-20 values OR 2-10% of dataset)
                # 2. Values repeat (not all unique like IDs)
                # 3. Not date/timestamp columns (handled separately)
                is_reasonable_cardinality = (2 <= unique_count <= 20) or (2 <= unique_count <= max(20, total_rows * 0.1))
                has_repetition = unique_count < total_rows * 0.9  # At least 10% of rows share values
                
                # Skip ID-like columns (high uniqueness)
                col_lower = col.lower()
                is_id_column = '_id' in col_lower or col_lower.endswith('id') or col_lower == 'id'
                
                if is_reasonable_cardinality and has_repetition and not is_id_column:
                    grouping_cols.append(col)
                    logging.info(f"✓ Detected grouping column: '{col}' ({unique_count} unique values, {unique_count/total_rows*100:.1f}% of dataset)")
                    
            except TypeError:
                # Column contains unhashable types, skip it
                pass
        
        # TOP-N PRE-CALCULATIONS - For ranking queries on large datasets
        # Disabled for now as it adds overhead and phi3:mini struggles with exact ordering
        # Re-enable when using larger models or for specific Top-N heavy workloads
        if False and len(df) > 500:
            # Detect ID/entity columns that could be used for Top-N queries
            id_cols = []
            entity_cols = []
            for col in df.columns:
                col_lower = col.lower()
                # Look for ID columns (customer_id, product_id, user_id, etc.)
                if '_id' in col_lower or col_lower.endswith('id'):
                    id_cols.append(col)
                # Look for entity name columns (customer, product, name, etc.)
                elif any(entity in col_lower for entity in ['customer', 'product', 'user', 'client', 'name', 'supplier']):
                    # Only include if it has reasonable uniqueness (not too many unique values)
                    try:
                        unique_count = df[col].nunique()
                        if 10 <= unique_count <= len(df):  # At least 10, but can be up to total rows
                            entity_cols.append(col)
                    except TypeError:
                        pass
            
            # Generate Top-N rankings if we have entity columns and numeric columns
            if (id_cols or entity_cols) and len(meaningful_numeric_cols) > 0:
                preview_parts.append(f"\n🏆 TOP-N RANKINGS (Pre-calculated from all {len(df):,} rows):")
                preview_parts.append(f"{'='*70}")
                preview_parts.append(f"⚠️  CRITICAL: Use these for 'top 5', 'top 10', 'highest', 'best' queries!")
                preview_parts.append(f"    DO NOT calculate from sample data - these are computed from FULL dataset!")
                
                # Combine ID and entity columns for grouping - prioritize customer/product
                ranking_cols = []
                for col in (id_cols + entity_cols):
                    col_lower = col.lower()
                    if 'customer' in col_lower:
                        ranking_cols.insert(0, col)  # Prioritize customer
                    elif 'product' in col_lower:
                        ranking_cols.insert(1 if len(ranking_cols) > 0 else 0, col)  # Second priority
                    else:
                        ranking_cols.append(col)
                
                ranking_cols = ranking_cols[:2]  # Limit to top 2 most important columns
                
                # Only calculate for the MOST important numeric column (usually first)
                for rank_col in ranking_cols:
                    for num_col in meaningful_numeric_cols[:1]:  # Only top 1 metric to save time
                        try:
                            # Calculate total by entity - this is the expensive operation
                            top_performers = df.groupby(rank_col)[num_col].sum().sort_values(ascending=False).head(10)
                            
                            if len(top_performers) > 0:
                                preview_parts.append(f"\n🏆 TOP-N RANKINGS (Pre-calculated from all {len(df):,} rows):")
                                preview_parts.append(f"⚠️  CRITICAL: Use these for 'top 5', 'top 10', 'highest', 'best' queries!")
                                preview_parts.append(f"    DO NOT calculate from sample data - these are computed from FULL dataset!")
                                preview_parts.append(f"\n🥇 Top 10 {rank_col.upper()} by {num_col.upper()}:")
                                for rank, (entity, value) in enumerate(top_performers.items(), 1):
                                    preview_parts.append(f"   {rank}. {entity}: {value:,.2f}" if isinstance(value, float) else f"   {rank}. {entity}: {value:,}")
                                preview_parts.append(f"")
                        except:
                            pass
                
                preview_parts.append(f"")
        
        # If we have grouping columns and numeric columns, show grouped stats
        if grouping_cols and len(numeric_cols) > 0:
            preview_parts.append(f"\n📈 GROUPED AGGREGATIONS (by category):")
            preview_parts.append(f"{'='*70}")
            preview_parts.append(f"⚠️  Use these for questions about 'by quarter', 'by region', 'by category', etc.")
            
            for group_col in grouping_cols[:5]:  # Show up to 5 grouping columns to cover most aggregation scenarios
                try:
                    # Prioritize important numeric columns
                    important_cols = []
                    for col in df.columns:
                        col_lower = col.lower()
                        if any(key in col_lower for key in ['revenue', 'profit', 'margin', 'income', 'expense', 'sales', 'cost', 'amount']):
                            if col in numeric_cols:
                                important_cols.append(col)
                    
                    # Add other numeric columns
                    for col in numeric_cols:
                        if col not in important_cols:
                            important_cols.append(col)
                    
                    # Show top 5 important numeric columns grouped
                    for num_col in important_cols[:5]:
                        try:
                            grouped = df.groupby(group_col)[num_col].agg(['sum', 'mean', 'count'])
                            preview_parts.append(f"\n📊 {num_col.upper()} by {group_col.upper()}:")
                            for group_val in grouped.index[:10]:  # Show top 10 groups
                                total = grouped.loc[group_val, 'sum']
                                avg = grouped.loc[group_val, 'mean']
                                count = grouped.loc[group_val, 'count']
                                preview_parts.append(f"  • {group_val}:")
                                preview_parts.append(f"    - Total: {total:,.2f}" if isinstance(total, float) else f"    - Total: {total:,}")
                                preview_parts.append(f"    - Average: {avg:,.2f}" if isinstance(avg, float) else f"    - Average: {avg:,}")
                                preview_parts.append(f"    - Count: {count:,} records")
                        except:
                            pass
                except:
                    pass
            
            # ADD QUICK RANKINGS for "highest/lowest by X" questions
            preview_parts.append(f"\n🏆 QUICK RANKINGS (for 'highest', 'lowest', 'best', 'worst' questions):")
            preview_parts.append(f"-" * 50)
            for group_col in grouping_cols[:2]:  # Top 2 grouping columns
                for num_col in ['revenue', 'sales', 'profit', 'amount', 'cost', 'spend', 'price']:
                    matching_cols = [c for c in numeric_cols if num_col in c.lower()]
                    for mc in matching_cols[:1]:
                        try:
                            grouped_totals = df.groupby(group_col)[mc].sum().sort_values(ascending=False)
                            highest_group = grouped_totals.index[0]
                            highest_val = grouped_totals.iloc[0]
                            lowest_group = grouped_totals.index[-1]
                            lowest_val = grouped_totals.iloc[-1]
                            preview_parts.append(f"• HIGHEST {mc.upper()} by {group_col.upper()}: {highest_group} = {highest_val:,.0f}")
                            preview_parts.append(f"• LOWEST {mc.upper()} by {group_col.upper()}: {lowest_group} = {lowest_val:,.0f}")
                        except:
                            pass
            preview_parts.append(f"")
        
        preview_parts.append(f"{'='*70}")
        
        # Schema preview
        preview_parts.append(f"\nColumn Types:")
        for col, info in list(schema.items())[:5]:
            preview_parts.append(f"- {col}: {info['type']} ({info['unique_values']} unique values)")
        
        # Sample data
        preview_parts.append(f"\nSample Data (first 3 rows):")
        sample_preview = df.head(3).to_string(max_cols=8, max_rows=3)
        preview_parts.append(sample_preview)
        
        # Important note for LLM
        if len(df) > 100:
            preview_parts.append(f"\n⚠️  CRITICAL REMINDER:")
            preview_parts.append(f"   • Sample shows only 3 rows, but ALL statistics above were calculated from {len(df):,} rows")
            preview_parts.append(f"   • For any aggregation question (sum, average, total, count), use the PRE-CALCULATED STATISTICS")
            preview_parts.append(f"   • DO NOT calculate from the 3-row sample - use the exact numbers provided above!")
        
        # Join and truncate
        preview_text = '\n'.join(preview_parts)
        
        if len(preview_text) > self.max_chars * 1.5:  # Allow more space for stats
            preview_text = preview_text[:int(self.max_chars * 1.5)] + "\n... (truncated)"
        
        return preview_text


# Convenience functions for direct use
def optimize_for_llm(filepath: str, max_rows: int = 100) -> Dict[str, Any]:
    """
    Quick optimization function
    
    Args:
        filepath: Path to data file
        max_rows: Maximum rows to sample
        
    Returns:
        Optimized data dict
    """
    optimizer = DataOptimizer(max_rows=max_rows)
    return optimizer.optimize_for_llm(filepath)


def flatten_nested_json(data: Union[Dict, List]) -> Union[Dict, List]:
    """
    Quick flatten function
    
    Args:
        data: Nested JSON data
        
    Returns:
        Flattened data
    """
    optimizer = DataOptimizer()
    return optimizer._flatten_nested_json(data)

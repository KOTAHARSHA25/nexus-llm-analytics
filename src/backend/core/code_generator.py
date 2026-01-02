
"""
Code Generator for LLM-based Data Analysis

Phase 2 Implementation: Generates and executes Python code from natural language queries.
Uses the existing EnhancedSandbox for safe execution.

Author: Research Team
Date: December 27, 2025
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GeneratedCode:
    """Container for generated code and metadata"""
    code: str
    is_valid: bool = True
    error_message: Optional[str] = None
    extraction_method: str = "python_block"


@dataclass 
class ExecutionResult:
    """Container for code execution results with full history context"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    code: Optional[str] = None
    execution_time_ms: float = 0.0
    
    # History tracking
    execution_id: Optional[str] = None
    generated_code: Optional[str] = None  # Original LLM output before cleaning
    query: Optional[str] = None
    model_used: Optional[str] = None
    attempt_count: int = 1
    retry_errors: Optional[List[str]] = None
    

class CodeGenerator:
    """
    Generates executable Python code from natural language queries.
    
    Workflow:
    1. Load prompt template
    2. Format prompt with data context
    3. Call LLM to generate code
    4. Extract code from response
    5. Validate code (syntax + security)
    6. Execute in sandbox
    7. Return results
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the code generator.
        
        Args:
            llm_client: LLM client for code generation (optional, will use default)
        """
        self.llm_client = llm_client
        # No more fixed template - we generate dynamic prompts
        
        # Lazy-load sandbox
        self._sandbox = None
        
        logger.info("CodeGenerator initialized")
    
    def _build_dynamic_prompt(self, query: str, df: pd.DataFrame) -> str:
        """
        Build a dynamic, context-aware prompt based on the user's query and data structure.
        This replaces the fixed template approach with intelligent prompt generation.
        """
        # Classify query intent first
        intent = self._classify_query_intent(query)
        
        # Extract comprehensive data structure
        data_structure = self._extract_data_structure(df)
        
        # Separate meaningful name columns from ID columns
        name_columns = []  # Human-readable names like track_name, artist_name
        id_columns = []    # Technical IDs like track_id, user_id
        
        for col in data_structure['identifier_columns']:
            col_lower = col.lower()
            # ID columns are technical identifiers, not useful for display
            if col_lower.endswith('_id') or col_lower == 'id' or 'uuid' in col_lower:
                id_columns.append(col)
            else:
                name_columns.append(col)
        
        # Get sample data with only meaningful columns (names + numeric)
        key_cols = (name_columns[:3] + data_structure['numeric_columns'][:2])
        if key_cols:
            sample_data = df[key_cols].head(3).to_string(index=False)
        else:
            sample_data = df.head(3).to_string(index=False)
        
        # Build column descriptions with their characteristics
        column_desc = []
        numeric_cols = []
        
        for col in df.columns:
            info = data_structure['columns'].get(col, {})
            dtype = info.get('dtype', str(df[col].dtype))
            samples = info.get('sample_values', [])
            sample_str = f" (e.g., {', '.join(str(s)[:30] for s in samples[:2])})" if samples else ""
            
            # Add column classification with clear guidance
            if col in name_columns:
                col_type = "[NAME - USE THIS for display]"
            elif col in id_columns:
                col_type = "[ID - do NOT use for display, not human-readable]"
            elif col in data_structure['numeric_columns']:
                min_val = info.get('min', '')
                max_val = info.get('max', '')
                col_type = f"[NUMERIC - range: {min_val} to {max_val}]"
                numeric_cols.append(col)
            elif col in data_structure['categorical_columns']:
                col_type = f"[CATEGORY - {info.get('unique_count', '?')} unique values]"
            elif col in data_structure['date_columns']:
                col_type = "[DATE]"
            else:
                col_type = f"[{dtype}]"
            
            column_desc.append(f"  - {col}: {col_type}{sample_str}")
        
        columns_text = "\n".join(column_desc)
        
        # Suggest relevant columns based on query analysis
        # IMPORTANT: Only suggest NAME columns, never ID columns
        query_lower = query.lower()
        suggested_cols = []
        
        # Add name columns (NOT id columns) for "top N", "list", "show" queries
        if any(word in query_lower for word in ['top', 'list', 'show', 'best', 'most', 'popular', 'highest', 'lowest']):
            suggested_cols.extend(name_columns[:2])  # Add NAME columns only
        
        # Find the metric column based on query keywords
        for col in numeric_cols:
            col_lower = col.lower()
            # Check if query mentions this column
            if any(word in query_lower for word in col_lower.replace('_', ' ').split()):
                suggested_cols.append(col)
                break
        
        # If no specific metric found, use common popularity/score patterns
        if not any(col in suggested_cols for col in numeric_cols):
            for col in numeric_cols:
                if any(word in col.lower() for word in ['popular', 'score', 'rating', 'count', 'sales', 'revenue', 'listen']):
                    suggested_cols.append(col)
                    break
        
        # Format suggested columns
        suggested_cols_str = ", ".join(f"'{c}'" for c in suggested_cols[:4]) if suggested_cols else "name and metric columns"
        
        # Add intent-specific instructions
        intent_guidance = ""
        if intent['type'] == 'ranking':
            limit = intent.get('limit', 10)
            order = 'nlargest' if intent['sort_order'] == 'desc' else 'nsmallest'
            intent_guidance = f"\nQUERY TYPE: Ranking - Use df.{order}({limit}, 'metric_column') and select NAME columns for display."
        elif intent['type'] == 'aggregation':
            agg = intent.get('aggregation', 'sum')
            intent_guidance = f"\nQUERY TYPE: Aggregation - Use .{agg}() function. Return a single value or grouped result."
        elif intent['type'] == 'comparison':
            intent_guidance = "\nQUERY TYPE: Comparison - Group by the categories being compared, then aggregate metrics."
        elif intent['type'] == 'lookup':
            intent_guidance = "\nQUERY TYPE: Lookup - Filter to find specific item, return the NAME not the ID."
        elif intent['type'] == 'trend':
            intent_guidance = "\nQUERY TYPE: Trend - Group by time period, aggregate metric, consider sorting by date."
        
        # Build DYNAMIC code examples using actual column names from the dataset
        # This ensures the prompt is 100% data-agnostic
        example_name_col = name_columns[0] if name_columns else "name_column"
        example_name_col2 = name_columns[1] if len(name_columns) > 1 else example_name_col
        example_metric_col = numeric_cols[0] if numeric_cols else "metric_column"
        
        # Build the dynamic prompt with clear instructions
        prompt = f"""You are a data analysis expert. Generate Python code to answer the user's question.

USER'S QUESTION: "{query}"

DATASET: {len(df)} rows, {len(df.columns)} columns

COLUMNS (note which are for display vs which are technical IDs):
{columns_text}

SAMPLE DATA:
{sample_data}

CRITICAL INSTRUCTIONS:
1. The DataFrame is loaded as `df`
2. Store the final answer in `result`
3. For listing/ranking queries:
   - Use NAME columns (marked with [NAME]) for display - these are human-readable
   - Do NOT include ID columns (marked with [ID]) - they contain codes like "abc123" that mean nothing to users
   - Include the metric column being ranked
{intent_guidance}

RECOMMENDED COLUMNS FOR THIS QUERY: [{suggested_cols_str}]

CODE PATTERNS (using columns from YOUR dataset):
```python
# For "top N" ranking queries:
result = df.nlargest(10, '{example_metric_col}')[['{example_name_col}', '{example_name_col2}', '{example_metric_col}']]

# For "what is the maximum/minimum X" - return just the value:
result = df['{example_metric_col}'].max()

# For "which item has highest X" - return the name:
result = df.loc[df['{example_metric_col}'].idxmax(), '{example_name_col}']

# For aggregations:
result = df.groupby('{example_name_col}')['{example_metric_col}'].sum().sort_values(ascending=False)
```

Generate ONLY a Python code block:
```python
result = ...
```"""
        
        return prompt
    
    def _classify_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify the user's query intent for better code generation.
        Returns intent type and extracted parameters.
        """
        query_lower = query.lower()
        
        intent = {
            'type': 'general',
            'limit': None,
            'sort_order': 'desc',
            'aggregation': None,
            'comparison': False
        }
        
        # Detect ranking queries: "top N", "bottom N", "best", "worst"
        top_match = re.search(r'\b(top|best|highest|most)\s*(\d+)?\b', query_lower)
        bottom_match = re.search(r'\b(bottom|worst|lowest|least)\s*(\d+)?\b', query_lower)
        
        if top_match:
            intent['type'] = 'ranking'
            intent['limit'] = int(top_match.group(2)) if top_match.group(2) else 10
            intent['sort_order'] = 'desc'
        elif bottom_match:
            intent['type'] = 'ranking'
            intent['limit'] = int(bottom_match.group(2)) if bottom_match.group(2) else 10
            intent['sort_order'] = 'asc'
        
        # Detect aggregation queries
        if any(word in query_lower for word in ['total', 'sum', 'average', 'avg', 'mean', 'count how many']):
            intent['type'] = 'aggregation'
            if 'average' in query_lower or 'avg' in query_lower or 'mean' in query_lower:
                intent['aggregation'] = 'mean'
            elif 'total' in query_lower or 'sum' in query_lower:
                intent['aggregation'] = 'sum'
            elif 'count' in query_lower:
                intent['aggregation'] = 'count'
        
        # Detect comparison queries
        if any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference between', 'compared to']):
            intent['type'] = 'comparison'
            intent['comparison'] = True
        
        # Detect lookup/filter queries
        if any(word in query_lower for word in ['what is the', 'which', 'find', 'get', 'show me the']):
            if intent['type'] == 'general':
                intent['type'] = 'lookup'
        
        # Detect trend queries
        if any(word in query_lower for word in ['trend', 'over time', 'by year', 'by month', 'growth', 'change']):
            intent['type'] = 'trend'
        
        return intent
    
    def _validate_result_human_readable(self, result: Any, df: pd.DataFrame) -> Tuple[Any, bool, str]:
        """
        Validate and transform result to ensure it's human-readable.
        Returns (transformed_result, was_modified, message).
        """
        if result is None:
            return None, False, ""
        
        # Get ID columns to check
        id_patterns = ['_id', 'uuid', 'guid']
        id_columns = [col for col in df.columns if any(p in col.lower() for p in id_patterns) or col.lower() == 'id']
        
        # If result is a DataFrame, check for ID-only columns
        if isinstance(result, pd.DataFrame):
            # Get name columns for replacement
            name_cols = [col for col in df.columns 
                        if col not in id_columns 
                        and df[col].dtype == 'object'
                        and col.lower().endswith('name') or 'name' in col.lower()]
            
            # Check if result only has ID columns (bad)
            result_cols = list(result.columns)
            has_only_ids = all(col in id_columns for col in result_cols if col in id_columns)
            has_no_names = not any(col in name_cols for col in result_cols)
            
            if has_only_ids and has_no_names and name_cols:
                # Try to add a name column
                for name_col in name_cols:
                    if name_col in df.columns and name_col not in result.columns:
                        # Can't easily add without index matching, just flag it
                        return result, False, f"Consider including '{name_col}' for readability"
            
            # Format numeric columns nicely
            for col in result.columns:
                if result[col].dtype in ['float64', 'float32']:
                    # Round to 2 decimals for display
                    if result[col].abs().max() > 1000:
                        result[col] = result[col].round(0).astype(int)
                    else:
                        result[col] = result[col].round(2)
            
            # Limit rows if too many
            if len(result) > 50:
                result = result.head(50)
                return result, True, "Result truncated to 50 rows"
        
        # If result is a Series with ID-like index, convert to DataFrame with names
        elif isinstance(result, pd.Series):
            if len(result) > 50:
                result = result.head(50)
                return result, True, "Result truncated to 50 items"
        
        return result, False, ""
    
    def _validate_code_columns(self, code: str, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Validate that code references existing columns.
        Returns (is_valid, error_message).
        """
        valid_columns = set(df.columns)
        
        # Find all potential column references in code
        # Pattern: df['column'] or df["column"] or ['col1', 'col2']
        quoted_refs = re.findall(r"['\"]([^'\"]+)['\"]", code)
        
        invalid_cols = []
        for ref in quoted_refs:
            # Skip if it's a known method or keyword
            if ref in ['result', 'df', 'ascending', 'descending', 'True', 'False', 'None']:
                continue
            # Skip if it looks like a format string
            if '{' in ref or '}' in ref:
                continue
            # Check if it could be a column reference
            if ref not in valid_columns and len(ref) < 50:  # Columns are usually short
                # Only flag if it looks like a column name (alphanumeric + underscore)
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', ref):
                    invalid_cols.append(ref)
        
        if invalid_cols:
            suggestions = []
            for invalid in invalid_cols[:3]:  # Limit suggestions
                # Find similar column names
                similar = [col for col in valid_columns if invalid.lower() in col.lower() or col.lower() in invalid.lower()]
                if similar:
                    suggestions.append(f"'{invalid}' -> maybe '{similar[0]}'?")
                else:
                    suggestions.append(f"'{invalid}' not found")
            
            return False, f"Invalid columns: {', '.join(suggestions)}. Valid: {', '.join(list(valid_columns)[:10])}"
        
        return True, None
    
    def _get_sandbox(self):
        """Lazy-load the sandbox"""
        if self._sandbox is None:
            from .sandbox import EnhancedSandbox
            self._sandbox = EnhancedSandbox(max_memory_mb=256, max_cpu_seconds=30)
        return self._sandbox
    
    def _get_llm_client(self):
        """Get or create LLM client"""
        if self.llm_client is None:
            from .llm_client import LLMClient
            self.llm_client = LLMClient()
        return self.llm_client
    
    def generate_code(self, 
                     query: str, 
                     df: pd.DataFrame,
                     model: str = "phi3:mini") -> GeneratedCode:
        """
        Generate Python code for the given query using dynamic prompt generation.
        
        This method:
        1. Analyzes the user's query
        2. Extracts comprehensive data structure
        3. Builds a context-aware prompt dynamically
        4. Generates code tailored to the specific question
        
        Args:
            query: Natural language question from the user
            df: DataFrame to analyze
            model: LLM model to use for generation
            
        Returns:
            GeneratedCode with the generated code or error
        """
        try:
            # Edge case: Empty or missing query
            if not query or not query.strip():
                return GeneratedCode(
                    code="",
                    is_valid=False,
                    error_message="Query cannot be empty"
                )
            
            # Edge case: Empty DataFrame
            if df is None or df.empty:
                return GeneratedCode(
                    code="",
                    is_valid=False,
                    error_message="DataFrame is empty or None - no data to analyze"
                )
            
            # Edge case: DataFrame with all null values
            if df.isna().all().all():
                return GeneratedCode(
                    code="",
                    is_valid=False,
                    error_message="DataFrame contains only null values"
                )
            
            # Build dynamic prompt based on user query and data structure
            # This is the key improvement - no fixed template, fully dynamic
            prompt = self._build_dynamic_prompt(query, df)
            
            # Extract data structure for logging
            data_structure = self._extract_data_structure(df)
            logger.info(f"Data structure: {len(data_structure['identifier_columns'])} identifiers, {len(data_structure['numeric_columns'])} numeric cols")
            
            # Call LLM with the dynamic prompt
            llm = self._get_llm_client()
            response = llm.generate(prompt, model=model)
            
            # Handle dict response from LLMClient
            if isinstance(response, dict):
                response_text = response.get('response', '')
                if response.get('error'):
                    return GeneratedCode(
                        code="",
                        is_valid=False,
                        error_message=response.get('error')
                    )
            else:
                response_text = str(response)
            
            # Extract code from response
            code = self._extract_code(response_text)
            
            if not code:
                return GeneratedCode(
                    code="",
                    is_valid=False,
                    error_message="No code block found in LLM response"
                )
            
            # Basic validation
            is_valid, error = self._validate_code_syntax(code)
            if not is_valid:
                return GeneratedCode(
                    code=code,
                    is_valid=False,
                    error_message=error
                )
            
            # Validate column references
            cols_valid, cols_error = self._validate_code_columns(code, df)
            if not cols_valid:
                logger.warning(f"Column validation warning: {cols_error}")
                # Don't fail, just log - LLM might be using valid but unrecognized patterns
            
            return GeneratedCode(code=code, is_valid=True)
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return GeneratedCode(
                code="",
                is_valid=False,
                error_message=str(e)
            )
    
    def execute_code(self, 
                    code: str, 
                    df: pd.DataFrame) -> ExecutionResult:
        """
        Execute generated code in the secure sandbox.
        
        Args:
            code: Python code to execute
            df: DataFrame to use as 'df' variable
            
        Returns:
            ExecutionResult with success status and result/error
        """
        import time
        start_time = time.time()
        
        try:
            sandbox = self._get_sandbox()
            
            # Execute with df available in globals
            result = sandbox.execute(
                code=code,
                data=df,
                extra_globals={'df': df}
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            if 'error' in result:
                return ExecutionResult(
                    success=False,
                    error=result['error'],
                    code=code,
                    execution_time_ms=execution_time
                )
            
            # Validate and format result for human readability
            raw_result = result.get('result')
            validated_result, was_modified, validation_msg = self._validate_result_human_readable(raw_result, df)
            
            if was_modified:
                logger.info(f"Result formatted: {validation_msg}")
            
            return ExecutionResult(
                success=True,
                result=validated_result,
                code=code,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Code execution failed: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
                code=code,
                execution_time_ms=execution_time
            )
    
    def generate_and_execute(self,
                            query: str,
                            df: pd.DataFrame,
                            model: str = "phi3:mini",
                            max_retries: int = 2,
                            data_file: Optional[str] = None,
                            save_history: bool = True) -> ExecutionResult:
        """
        Generate code and execute it, with retry logic and history tracking.
        
        Args:
            query: Natural language question
            df: DataFrame to analyze
            model: LLM model for generation
            max_retries: Number of retries on failure
            data_file: Name of the data file being analyzed (for history)
            save_history: Whether to save this execution to history
            
        Returns:
            ExecutionResult with the final result and history tracking
        """
        last_error = None
        retry_errors = []
        original_generated_code = None
        final_cleaned_code = None
        attempt_count = 0
        
        for attempt in range(max_retries + 1):
            attempt_count = attempt + 1
            
            # Generate code
            gen_result = self.generate_code(query, df, model)
            
            # Store original generated code from first attempt
            if original_generated_code is None and gen_result.code:
                original_generated_code = gen_result.code
            
            if not gen_result.is_valid:
                last_error = gen_result.error_message
                retry_errors.append(f"Gen attempt {attempt_count}: {last_error}")
                logger.warning(f"Code generation failed (attempt {attempt_count}): {last_error}")
                continue
            
            final_cleaned_code = gen_result.code
            
            # Execute code
            exec_result = self.execute_code(gen_result.code, df)
            
            if exec_result.success:
                logger.info(f"Code execution succeeded in {exec_result.execution_time_ms:.1f}ms")
                
                # Save to history
                execution_id = None
                if save_history:
                    execution_id = self._save_to_history(
                        query=query,
                        model=model,
                        generated_code=original_generated_code or gen_result.code,
                        cleaned_code=gen_result.code,
                        success=True,
                        execution_time_ms=exec_result.execution_time_ms,
                        result=exec_result.result,
                        data_file=data_file,
                        df=df,
                        attempt_count=attempt_count,
                        retry_errors=retry_errors
                    )
                
                # Return enriched result
                exec_result.execution_id = execution_id
                exec_result.generated_code = original_generated_code or gen_result.code
                exec_result.query = query
                exec_result.model_used = model
                exec_result.attempt_count = attempt_count
                exec_result.retry_errors = retry_errors if retry_errors else None
                
                return exec_result
            
            last_error = exec_result.error
            retry_errors.append(f"Exec attempt {attempt_count}: {last_error}")
            logger.warning(f"Code execution failed (attempt {attempt_count}): {last_error}")
        
        # All retries failed - still save to history
        execution_id = None
        if save_history:
            execution_id = self._save_to_history(
                query=query,
                model=model,
                generated_code=original_generated_code or "",
                cleaned_code=final_cleaned_code or "",
                success=False,
                execution_time_ms=0.0,
                result=None,
                error=last_error,
                data_file=data_file,
                df=df,
                attempt_count=attempt_count,
                retry_errors=retry_errors
            )
        
        return ExecutionResult(
            success=False,
            error=f"Failed after {max_retries + 1} attempts. Last error: {last_error}",
            execution_id=execution_id,
            generated_code=original_generated_code,
            code=final_cleaned_code,
            query=query,
            model_used=model,
            attempt_count=attempt_count,
            retry_errors=retry_errors
        )
    
    def _save_to_history(self,
                        query: str,
                        model: str,
                        generated_code: str,
                        cleaned_code: str,
                        success: bool,
                        execution_time_ms: float,
                        result: Any = None,
                        error: Optional[str] = None,
                        data_file: Optional[str] = None,
                        df: Optional[pd.DataFrame] = None,
                        attempt_count: int = 1,
                        retry_errors: Optional[List[str]] = None) -> Optional[str]:
        """Save execution to history and return execution_id"""
        try:
            from core.code_execution_history import get_execution_history
            
            history = get_execution_history()
            
            columns = list(df.columns) if df is not None else []
            row_count = len(df) if df is not None else 0
            
            execution_id = history.save_execution(
                query=query,
                model_used=model,
                generated_code=generated_code,
                cleaned_code=cleaned_code,
                success=success,
                execution_time_ms=execution_time_ms,
                result=result,
                error=error,
                data_file=data_file,
                columns=columns,
                row_count=row_count,
                attempt_count=attempt_count,
                retry_errors=retry_errors
            )
            
            return execution_id
            
        except Exception as e:
            logger.error(f"Failed to save execution to history: {e}")
            return None
    
    def replay_execution(self, 
                        execution_id: str, 
                        df: pd.DataFrame) -> ExecutionResult:
        """
        Replay a past execution using its stored code.
        
        Args:
            execution_id: ID of the execution to replay
            df: DataFrame to execute against (should match original)
            
        Returns:
            ExecutionResult from replaying the code
        """
        from core.code_execution_history import get_execution_history
        
        history = get_execution_history()
        replay_info = history.get_code_for_replay(execution_id)
        
        if not replay_info:
            return ExecutionResult(
                success=False,
                error=f"Execution {execution_id} not found in history"
            )
        
        # Execute the stored cleaned code
        result = self.execute_code(replay_info['cleaned_code'], df)
        result.execution_id = execution_id
        result.query = replay_info['query']
        result.generated_code = replay_info['original_code']
        result.model_used = replay_info['model_used']
        
        return result
    
    def get_execution_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent execution history for display.
        
        Returns:
            List of execution records with code and results
        """
        from core.code_execution_history import get_execution_history
        
        history = get_execution_history()
        records = history.get_recent_executions(limit=limit)
        
        return [
            {
                'execution_id': r.execution_id,
                'timestamp': r.timestamp,
                'query': r.query,
                'model_used': r.model_used,
                'success': r.success,
                'execution_time_ms': r.execution_time_ms,
                'result_preview': r.result_preview,
                'result_type': r.result_type,
                'error': r.error,
                'generated_code': r.generated_code,
                'cleaned_code': r.cleaned_code,
                'data_file': r.data_file,
                'columns': r.columns,
                'attempt_count': r.attempt_count
            }
            for r in records
        ]
    
    def _get_data_preview(self, df: pd.DataFrame, max_rows: int = 5) -> str:
        """Get a preview of the DataFrame for the prompt"""
        preview_df = df.head(max_rows)
        return preview_df.to_string(index=False)
    
    def _extract_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract comprehensive data structure information for LLM context.
        This helps the LLM understand what columns exist and how to use them.
        """
        structure = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': {},
            'identifier_columns': [],  # Columns that can identify a row (names, titles, etc.)
            'numeric_columns': [],     # Columns with numeric values
            'categorical_columns': [], # Columns with limited unique values
            'date_columns': []         # Date/time columns
        }
        
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'non_null_count': int(df[col].notna().sum()),
                'unique_count': int(df[col].nunique()),
                'sample_values': []
            }
            
            # Get sample values (up to 3 unique non-null values)
            non_null_values = df[col].dropna().unique()
            sample_count = min(3, len(non_null_values))
            col_info['sample_values'] = [str(v)[:50] for v in non_null_values[:sample_count]]
            
            # Classify column type
            dtype_str = str(df[col].dtype)
            
            if 'datetime' in dtype_str or 'date' in dtype_str:
                structure['date_columns'].append(col)
            elif dtype_str in ['int64', 'float64', 'int32', 'float32']:
                structure['numeric_columns'].append(col)
                # Add min/max for numeric columns
                col_info['min'] = float(df[col].min()) if df[col].notna().any() else None
                col_info['max'] = float(df[col].max()) if df[col].notna().any() else None
            elif dtype_str == 'object' or dtype_str == 'string':
                # Check if it's likely an identifier column
                uniqueness_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
                
                # Identifier columns have high uniqueness and text values
                # Also check column name patterns
                identifier_patterns = ['name', 'title', 'id', 'track', 'song', 'artist', 
                                       'product', 'item', 'user', 'customer', 'label']
                is_identifier = (
                    uniqueness_ratio > 0.5 or 
                    any(pattern in col.lower() for pattern in identifier_patterns)
                )
                
                if is_identifier:
                    structure['identifier_columns'].append(col)
                else:
                    structure['categorical_columns'].append(col)
            
            structure['columns'][col] = col_info
        
        return structure
    
    def _format_structure_for_prompt(self, structure: Dict[str, Any]) -> str:
        """Format the data structure as a clear string for the LLM prompt."""
        lines = []
        lines.append(f"Dataset has {structure['total_rows']} rows and {structure['total_columns']} columns.")
        lines.append("")
        
        # List columns by type with explicit guidance
        if structure['identifier_columns']:
            lines.append(f"IDENTIFIER COLUMNS (use these to identify/name records):")
            for col in structure['identifier_columns']:
                info = structure['columns'][col]
                samples = ", ".join(f'"{v}"' for v in info['sample_values'])
                lines.append(f"  - '{col}': {info['unique_count']} unique values, e.g., {samples}")
        
        if structure['numeric_columns']:
            lines.append(f"\nNUMERIC COLUMNS (use these for calculations, min, max, sum, avg):")
            for col in structure['numeric_columns']:
                info = structure['columns'][col]
                lines.append(f"  - '{col}': range [{info.get('min', 'N/A')} to {info.get('max', 'N/A')}]")
        
        if structure['categorical_columns']:
            lines.append(f"\nCATEGORICAL COLUMNS (use these for grouping/filtering):")
            for col in structure['categorical_columns']:
                info = structure['columns'][col]
                samples = ", ".join(f'"{v}"' for v in info['sample_values'])
                lines.append(f"  - '{col}': {info['unique_count']} categories, e.g., {samples}")
        
        if structure['date_columns']:
            lines.append(f"\nDATE COLUMNS:")
            for col in structure['date_columns']:
                lines.append(f"  - '{col}'")
        
        # Explicit warning
        lines.append("")
        lines.append("WARNING: ONLY USE THE COLUMN NAMES LISTED ABOVE. Do NOT invent or assume other column names exist.")
        
        # Add hint for top N queries
        if structure['identifier_columns'] and structure['numeric_columns']:
            lines.append("")
            lines.append(f"TIP FOR TOP N QUERIES: Select identifier columns ({', '.join(structure['identifier_columns'][:3])}) plus the numeric metric being ranked.")
        
        return "\n".join(lines)
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response"""
        # Try ```python ... ``` blocks first
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            code = matches[0].strip()
        else:
            # Try generic ``` ... ``` blocks
            pattern = r'```\s*(.*?)\s*```'
            matches = re.findall(pattern, response, re.DOTALL)
            
            if matches:
                code = matches[0].strip()
            elif 'result' in response and ('=' in response or 'df' in response):
                # No code blocks - try to use the whole response if it looks like code
                code = response.strip()
            else:
                return ""
        
        # Clean up the code - remove import statements and ellipsis
        code = self._clean_generated_code(code)
        return code
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean generated code to work in sandbox"""
        lines = code.split('\n')
        clean_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip import statements (pd and np already available)
            if stripped.startswith('import ') or stripped.startswith('from '):
                continue
            
            # Skip ellipsis lines (standalone or in patterns)
            if stripped == '...' or stripped == 'result = ...' or stripped == 'pass':
                continue
            
            # Skip lines that are just ellipsis in any context
            if '...' in stripped and not stripped.startswith('#'):
                # Check if the line is essentially just an ellipsis expression
                clean_stripped = stripped.replace('...', '').replace('=', '').strip()
                if not clean_stripped or clean_stripped in ('result', 'pass'):
                    continue
            
            # Skip print statements
            if stripped.startswith('print('):
                continue
                
            # Skip LLM placeholder comments like "# ... your code here"
            if stripped.startswith('#') and '...' in stripped:
                continue
            
            clean_lines.append(line)
        
        # Remove leading/trailing empty lines but preserve indentation
        result = '\n'.join(clean_lines).strip()
        
        # Final safety: if code still has bare ellipsis, remove them
        result = result.replace('\n...\n', '\n').replace('\n...', '').replace('...\n', '')
        
        return result
    
    def _validate_code_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate code syntax"""
        import ast
        
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"


# Singleton instance
_code_generator = None

def get_code_generator() -> CodeGenerator:
    """Get singleton CodeGenerator instance"""
    global _code_generator
    if _code_generator is None:
        _code_generator = CodeGenerator()
    return _code_generator

"""Code Generator for LLM-based Data Analysis — Nexus LLM Analytics
===================================================================

Phase 2 implementation: generates and executes Python code from natural
language queries using the existing :class:`EnhancedSandbox` for safe
execution.  Supports dynamic prompt building, model-size adaptation,
ML/statistical prompt specialisation, and DynamicPlanner integration.

Classes
-------
GeneratedCode
    Container for generated code and validation metadata.
ExecutionResult
    Container for sandbox execution results with history tracking.
CodeGenerator
    End-to-end code generation → validation → execution pipeline.

v2.0 Enterprise Additions
-------------------------
* :class:`CodeGenerationMetrics` — tracks generation/execution
  success rates, latencies, and retry statistics.
* :func:`get_code_generator` — already exists as singleton accessor
  (documented for completeness).
"""
from __future__ import annotations

import ast
import json
import re
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GeneratedCode:
    """Container for generated code and validation metadata.

    Attributes:
        code: The generated Python source code.
        is_valid: Whether the code passed syntax validation.
        error_message: Human-readable validation error (if any).
        extraction_method: Strategy used to extract code from LLM output.
    """
    code: str
    is_valid: bool = True
    error_message: Optional[str] = None
    extraction_method: str = "python_block"


@dataclass 
class ExecutionResult:
    """Container for code execution results with full history context.

    Attributes:
        success: Whether the sandbox execution completed without error.
        result: The value of the ``result`` variable after execution.
        error: Error message if execution failed.
        code: The cleaned code that was actually executed.
        execution_time_ms: Wall-clock execution duration.
        execution_id: Unique history identifier (set after save).
        generated_code: The original LLM output before cleaning.
        query: The originating natural-language question.
        model_used: LLM model identifier.
        attempt_count: Number of generation/execution attempts.
        retry_errors: Error messages from earlier failed attempts.
    """
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
    """Generates executable Python code from natural language queries.

    Workflow:
        1. Build a dynamic, context-aware prompt (adapts to model size).
        2. Call the configured LLM via circuit-breaker protection.
        3. Extract and clean code from the LLM response.
        4. Validate syntax and column references.
        5. Execute in the :class:`EnhancedSandbox`.
        6. Save result to execution history.

    Prompt Strategies:
        * **Standard** — full column metadata, intent classification,
          data-driven code examples.
        * **Simple** — reduced cognitive load for small models
          (tiny / mini / ≤3B params).
        * **ML** — specialised template for clustering, regression,
          PCA, statistical tests, and forecasting.

    Thread Safety:
        Not inherently thread-safe.  Wrap in external synchronisation
        if concurrent ``generate_and_execute`` calls are required.
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
        
        # Cache circuit breaker config
        self._cb_config = None
        
        logger.info("CodeGenerator initialized")
    
    def _load_circuit_breaker_config(self) -> Optional[Dict[str, Any]]:
        """Load circuit breaker configuration for code_generator from config file."""
        if self._cb_config is not None:
            return self._cb_config
        
        try:
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "cot_review_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    cb_config = config.get('circuit_breaker', {})
                    
                    if not cb_config.get('enabled', True):
                        self._cb_config = None
                        return None
                    
                    # Get code_generator specific settings
                    self._cb_config = cb_config.get('circuits', {}).get('code_generator', {
                        'failure_threshold': 2,
                        'recovery_timeout': 45,
                        'success_threshold': 2,
                        'timeout': 45,
                        'enabled': True
                    })
                    return self._cb_config
        except Exception as e:
            logger.warning("Failed to load circuit breaker config: %s", e)
        
        # Default config
        self._cb_config = {
            'failure_threshold': 2,
            'recovery_timeout': 45,
            'success_threshold': 2,
            'timeout': 45,
            'enabled': True
        }
        return self._cb_config
    
    def _build_dynamic_prompt(self, query: str, df: pd.DataFrame, model: str = None, analysis_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a dynamic, context-aware prompt based on the user's query and data structure.
        Supports both standard analytics and advanced ML/statistical queries.
        Adapts prompt complexity based on model size for optimal accuracy.
        
        Args:
            query: User's natural language question
            df: DataFrame to analyze
            model: LLM model name (e.g., 'phi3:mini', 'llama3.1:8b')
            analysis_context: Optional strategy from DynamicPlanner (dict with 'strategy' and 'steps')
        """
        # Inject DynamicPlanner strategy at the top if available (with robust validation)
        planner_guidance = ""
        if analysis_context:
            try:
                # Validate analysis_context structure
                if not isinstance(analysis_context, dict):
                    logger.warning("Invalid analysis_context type: %s", type(analysis_context))
                else:
                    strategy = analysis_context.get('strategy', '')
                    if strategy and isinstance(strategy, str):
                        # Sanitize strategy (max 1000 chars, no special control chars)
                        strategy = str(strategy).strip()[:1000]
                        strategy = ''.join(char for char in strategy if char.isprintable() or char == '\n')
                        
                        if strategy:
                            planner_guidance = f"\n\nANALYSIS STRATEGY (follow this approach):\n{strategy}\n"
                            logger.info("DynamicPlanner strategy injected into code prompt: %s...", strategy[:80])
                            
                            # Validate and add steps
                            steps = analysis_context.get('steps', [])
                            if steps and isinstance(steps, list):
                                validated_steps = []
                                for i, step in enumerate(steps[:10]):  # Max 10 steps
                                    if isinstance(step, str):
                                        step_clean = str(step).strip()[:300]  # Max 300 chars per step
                                        step_clean = ''.join(char for char in step_clean if char.isprintable() or char == '\n')
                                        if step_clean:
                                            validated_steps.append(f"{i+1}. {step_clean}")
                                
                                if validated_steps:
                                    steps_text = "\n".join(validated_steps)
                                    planner_guidance += f"\nSTEPS:\n{steps_text}\n"
                                    logger.debug("Injected %d validated steps", len(validated_steps))
            except Exception as e:
                logger.warning("Failed to inject DynamicPlanner strategy: %s", e)
                planner_guidance = ""  # Fail gracefully without breaking code generation
        # Detect if query requires ML/statistical capabilities
        # Use broad patterns that work across ALL domains (genomics, IoT, finance, etc.)
        ml_keywords = ['cluster', 'classification', 'classify', 'predict', 'regression', 'pca',
                       'random forest', 'decision tree', 'logistic', 'machine learning', 'ml',
                       't-test', 'anova', 'chi-square', 'statistical test', 'significance',
                       'forecast', 'arima', 'time series', 'seasonal', 'trend analysis',
                       'anomaly', 'outlier', 'z-score', 'standard deviation test',
                       'feature importance', 'dimensionality reduction', 'train test split',
                       'normalize', 'survival analysis', 'enrichment', 'decompose',
                       'k-nearest', 'svm', 'neural', 'cross validation', 'bootstrap',
                       'bayesian', 'monte carlo', 'principal component', 'eigenvalue',
                       'fft', 'frequency analysis', 'spectral', 'kaplan-meier',
                       'upregulated', 'downregulated', 'differential expression']
        
        query_lower = query.lower()
        is_ml_query = any(keyword in query_lower for keyword in ml_keywords)
        
        # ML queries ALWAYS get detailed ML prompts regardless of model size
        if is_ml_query:
            logger.info("ML query detected - using ML prompt regardless of model size")
            return self._build_ml_prompt(query, df)
        
        # For non-ML queries, detect if this is a small model that needs simplified prompts
        # Small models: tiny, mini, or 1-3 billion parameters
        model_name = (model or "").lower()
        
        # Check for small model indicators
        is_tiny = 'tiny' in model_name
        is_mini = 'mini' in model_name
        # Match patterns like ':1b', ':2b', ':3b', ':1.5b', ':0.5b' but NOT ':13b', ':70b', ':35b'
        # The pattern ensures the first digit after separator is 0-3
        small_param_match = re.search(r'[:_\-\s]([0-3](?:\.\d+)?)\s*b\b', model_name)
        is_small_params = small_param_match is not None
        
        is_small_model = is_tiny or is_mini or is_small_params
        
        # If small model, use simplified prompt template
        if is_small_model:
            logger.info("Using simplified prompt for small model: %s", model)
            return self._build_simple_prompt(query, df)
        
        # Otherwise use standard analytics prompt
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
        
        # If no specific metric found, use the first numeric column as a sensible default
        # This is domain-agnostic: works for any dataset (genomics, IoT, finance, logs, etc.)
        if not any(col in suggested_cols for col in numeric_cols):
            if numeric_cols:
                suggested_cols.append(numeric_cols[0])
        
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

USER'S QUESTION: "{query}"{planner_guidance}

DATASET: {len(df)} rows, {len(df.columns)} columns

COLUMNS (note which are for display vs which are technical IDs):
{columns_text}

SAMPLE DATA:
{sample_data}

CRITICAL INSTRUCTIONS:
1. The DataFrame is loaded as `df`
2. Store the final answer in `result`
3. DO NOT generate code to plot charts (e.g., do not use matplotlib, seaborn, or plt.show()).
   - The system will AUTOMATICALY generate interactive Plotly charts if 'result' is a DataFrame.
   - Just filter/aggregate the data into a DataFrame and assign it to `result`.
4. For listing/ranking queries:
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
    
    def _build_simple_prompt(self, query: str, df: pd.DataFrame) -> str:
        """
        Build a simplified prompt for small models (tiny, mini, 1b-3b).
        Reduces cognitive load with concise instructions and essential info only.
        
        Args:
            query: User's natural language question
            df: DataFrame to analyze
        
        Returns:
            Simplified prompt optimized for small models
        """
        try:
            # Extract essential data structure
            data_structure = self._extract_data_structure(df)
            
            # Get all columns as a simple comma-separated list
            all_columns = list(df.columns)
            columns_list = ", ".join(f"'{col}'" for col in all_columns)
            
            # Read simple template
            template_path = Path(__file__).parent.parent / "prompts" / "code_generation_prompt_simple.txt"
            
            if template_path.exists():
                with open(template_path, 'r', encoding='utf-8') as f:
                    template = f.read()
                
                # Simple substitution - just the essentials
                prompt = template.format(
                    columns=columns_list,
                    query=query
                )
            else:
                # Fallback if template missing
                logger.warning("Simple prompt template not found, using inline fallback")
                prompt = f"""Generate Python code to answer: "{query}"

Columns: {columns_list}

Rules:
1. Use ONLY these columns
2. Store answer in `result`
3. No print statements
4. NO plotting code (matplotlib/seaborn)

Output code only:
```python
result = ...
```"""
            
            return prompt
            
        except Exception as e:
            logger.error("Error building simple prompt: %s", e, exc_info=True)
            # Minimal fallback
            return f"""Generate code for: "{query}"
Columns: {', '.join(df.columns)}
Store result in `result` variable.
```python
result = ...
```"""
    
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
    
    def _build_ml_prompt(self, query: str, df: pd.DataFrame) -> str:
        """
        Build specialized prompt for ML/statistical queries.
        Includes examples of clustering, regression, classification, etc.
        """
        # Get data structure
        columns_str = ", ".join(df.columns.tolist())
        dtypes_str = df.dtypes.to_string()
        sample_data = df.head(5).to_string(max_rows=5, max_cols=10)
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Load ML prompt template
        prompt_file = Path(__file__).parent.parent / 'prompts' / 'ml_code_generation_prompt.txt'
        
        if prompt_file.exists():
            with open(prompt_file, 'r', encoding='utf-8') as f:
                template = f.read()
            
            # Fill in template
            prompt = template.format(
                data_preview=sample_data,
                columns=columns_str,
                dtypes=dtypes_str,
                query=query
            )
        else:
            # Fallback inline ML prompt
            prompt = f"""You are an expert data scientist. Generate Python code for: "{query}"

DATA: {len(df)} rows, {len(df.columns)} columns
COLUMNS: {columns_str}
NUMERIC: {', '.join(numeric_cols[:10]) if numeric_cols else 'None'}
CATEGORICAL: {', '.join(categorical_cols[:10]) if categorical_cols else 'None'}

SAMPLE DATA:
{sample_data}

AVAILABLE (pre-loaded, do NOT import):
- KMeans, RandomForestClassifier, LogisticRegression, LinearRegression
- PCA, StandardScaler, train_test_split, accuracy_score
- stats, ttest_ind, f_oneway, pearsonr, ARIMA

RULES:
1. Store result in `result` variable
2. Handle missing data: df = df.dropna()
3. Use simple models: max_iter=100, max_depth=5, n_estimators=50
4. Return dict for ML results: {{'accuracy': ..., 'feature_importance': ...}}
5. DO NOT PLOT. Do not use matplotlib/seaborn. Return data structures only.

EXAMPLE (K-means):
```python
X = df[['col1', 'col2']].dropna()
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clean = df.dropna(subset=['col1', 'col2'])
df_clean['cluster'] = kmeans.fit_predict(X)
result = df_clean.groupby('cluster')[['col1', 'col2']].mean()
```

Generate code:
```python
result = ...
```"""
        
        return prompt
    
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
            from backend.core.security.sandbox import EnhancedSandbox
            self._sandbox = EnhancedSandbox(max_memory_mb=256, max_cpu_seconds=30)
        return self._sandbox
    
    def _get_llm_client(self):
        """Get or create LLM client"""
        if self.llm_client is None:
            from backend.core.llm_client import LLMClient
            self.llm_client = LLMClient()
        return self.llm_client
    
    def generate_code(self, 
                     query: str, 
                     df: pd.DataFrame,
                     model: str = "phi3:mini",
                     analysis_context: Optional[Dict[str, Any]] = None,
                     error_feedback: Optional[Dict[str, str]] = None) -> GeneratedCode:
        """
        Generate Python code for the given query using dynamic prompt generation.
        
        This method:
        1. Analyzes the user's query
        2. Extracts comprehensive data structure
        3. Builds a context-aware prompt dynamically
        4. Generates code tailored to the specific question
        5. If error_feedback is provided, includes previous failed code and
           error message so the LLM can produce a corrected version
           (Patent: iterative verification feedback loop)
        
        Args:
            query: Natural language question from the user
            df: DataFrame to analyze
            model: LLM model to use for generation
            analysis_context: Optional strategy from DynamicPlanner (dict with 'strategy' and 'steps')
            error_feedback: Optional dict with 'failed_code' and 'error' from a
                           previous failed attempt, enabling iterative correction
            
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
            # Adapts prompt complexity based on model size
            # Includes DynamicPlanner strategy if available
            prompt = self._build_dynamic_prompt(query, df, model, analysis_context=analysis_context)
            
            # PATENT: Iterative Verification Feedback Loop
            # If a previous attempt failed, inject the error context so the LLM
            # can learn from the failure and generate corrected code instead of
            # regenerating blindly. This implements the patent's requirement that
            # "the Verifier triggers an automated feedback loop to send feedback
            # back through the generation of code phases."
            if error_feedback and isinstance(error_feedback, dict):
                failed_code = error_feedback.get('failed_code', '')
                error_msg = error_feedback.get('error', '')
                attempt_num = error_feedback.get('attempt', 0)
                if failed_code and error_msg:
                    correction_prompt = f"""\n\n⚠️ CORRECTION REQUIRED (Attempt {attempt_num}):\nYour previous code FAILED with this error:\n```\n{error_msg}\n```\n\nThe failing code was:\n```python\n{failed_code}\n```\n\nFix the error above. Generate corrected Python code that avoids this issue.\nCommon fixes: check column names exist, handle NaN values, use correct data types.\nStore the result in `result` variable.\n"""
                    prompt = prompt + correction_prompt
                    logger.info("Injected error feedback for iterative correction (attempt %d)", attempt_num)
            
            # Extract data structure for logging
            data_structure = self._extract_data_structure(df)
            logger.info("Data structure: %d identifiers, %d numeric cols", len(data_structure['identifier_columns']), len(data_structure['numeric_columns']))
            
            # FIX 12 (ENTERPRISE): Circuit Breaker Protection for Code Generation LLM Calls
            try:
                # Try to use circuit breaker if Phase 1 available
                try:
                    try:
                        from backend.infra.circuit_breaker import get_circuit_breaker, CircuitBreakerConfig
                    except ImportError:
                        # Circuit breaker not available, continue without it
                        logger.debug("Circuit breaker not available for code generator")
                        raise
                    # phase1_integration is available if circuit_breaker imported OK
                    PHASE1_AVAILABLE = True
                    
                    if PHASE1_AVAILABLE:
                        # Load circuit breaker configuration
                        cb_config = self._load_circuit_breaker_config()
                        
                        if cb_config and cb_config.get('enabled', True):
                            config = CircuitBreakerConfig(
                                failure_threshold=cb_config.get('failure_threshold', 2),
                                recovery_timeout=cb_config.get('recovery_timeout', 45.0),
                                success_threshold=cb_config.get('success_threshold', 2),
                                timeout=cb_config.get('timeout', 45.0)
                            )
                            circuit = get_circuit_breaker("code_generator", config)
                            
                            # Wrap LLM call
                            def llm_call():
                                llm = self._get_llm_client()
                                response = llm.generate(prompt, model=model)
                                if isinstance(response, dict):
                                    if response.get('error'):
                                        return {"success": False, "error": response.get('error')}
                                    return {"success": True, "response": response.get('response', '')}
                                return {"success": True, "response": str(response)}
                            
                            result = circuit.call(llm_call)
                            
                            if result.get("fallback_used"):
                                logger.warning("Circuit breaker fallback for code_generator")
                                return GeneratedCode(
                                    code="",
                                    is_valid=False,
                                    error_message="Code generation service temporarily unavailable"
                                )
                            
                            if not result.get("success"):
                                return GeneratedCode(
                                    code="",
                                    is_valid=False,
                                    error_message=result.get('error', 'Unknown error')
                                )
                            
                            response_text = result.get("response", "")
                        else:
                            # Circuit breaker disabled, call directly
                            llm = self._get_llm_client()
                            response = llm.generate(prompt, model=model)
                            response_text = response.get('response', '') if isinstance(response, dict) else str(response)
                    else:
                        # Phase 1 not available, call directly
                        llm = self._get_llm_client()
                        response = llm.generate(prompt, model=model)
                        response_text = response.get('response', '') if isinstance(response, dict) else str(response)
                        
                except ImportError:
                    # Circuit breaker not available, proceed without protection
                    llm = self._get_llm_client()
                    response = llm.generate(prompt, model=model)
                    response_text = response.get('response', '') if isinstance(response, dict) else str(response)
            
            except Exception as llm_error:
                logger.error("LLM call failed: %s", llm_error, exc_info=True)
                return GeneratedCode(
                    code="",
                    is_valid=False,
                    error_message=f"Code generation failed: {str(llm_error)}"
                )
            
            # Extract code from response
            code = self._extract_code(response_text)
            
            if not code:
                return GeneratedCode(
                    code="",
                    is_valid=False,
                    error_message="No code block found in LLM response"
                )
            
            # Basic validation (with auto-repair for small model errors)
            is_valid, error = self._validate_code_syntax(code)
            if not is_valid:
                # Attempt auto-repair for common small-model syntax errors
                repaired = self._attempt_code_repair(code)
                if repaired and repaired != code:
                    is_valid2, error2 = self._validate_code_syntax(repaired)
                    if is_valid2:
                        logger.info("Auto-repaired code syntax (was: %s)", error)
                        code = repaired
                        is_valid, error = True, None
                
                if not is_valid:
                    return GeneratedCode(
                        code=code,
                        is_valid=False,
                        error_message=error
                    )
            
            # Validate column references
            cols_valid, cols_error = self._validate_code_columns(code, df)
            if not cols_valid:
                logger.warning("Column validation warning: %s", cols_error)
                # Don't fail, just log - LLM might be using valid but unrecognized patterns
            
            return GeneratedCode(code=code, is_valid=True)
            
        except Exception as e:
            logger.error("Code generation failed: %s", e, exc_info=True)
            return GeneratedCode(
                code="",
                is_valid=False,
                error_message=str(e)
            )

    def _review_generated_code(self, code: str, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Pre-execution code review — Patent Claim 2.

        Reviews automatically generated analytical code **before** it is
        executed in the sandbox.  Performs deterministic static checks that
        catch common LLM code-generation errors without needing an extra
        LLM call (keeps latency low).

        Checks performed:
            1. Code must assign to ``result`` (contract with sandbox).
            2. Dangerous/disallowed operations (file I/O, network, exec/eval).
            3. Column references validated against the actual DataFrame.
            4. Overly-long code rejected (LLMs sometimes generate essays).
            5. Infinite-loop risk detection (unbounded ``while True``).

        Args:
            code: The generated Python source to review.
            query: The user's original question (for contextual logging).
            df: The DataFrame the code will operate on.

        Returns:
            dict with ``approved`` (bool) and ``reason`` (str) keys.
        """
        try:
            # 1. Must assign to 'result' variable
            if 'result' not in code and 'result =' not in code:
                return {'approved': False, 'reason': "Code does not assign to 'result' variable"}

            # 2. Block dangerous operations (defence-in-depth with sandbox)
            dangerous_patterns = [
                ('open(', 'File I/O operation detected'),
                ('os.system', 'System command execution detected'),
                ('subprocess', 'Subprocess execution detected'),
                ('requests.', 'Network request detected'),
                ('urllib', 'Network access detected'),
                ('eval(', 'eval() call detected'),
                ('exec(', 'exec() call detected'),
                ('__import__', 'Dynamic import detected'),
                ('shutil', 'File manipulation detected'),
                ('pickle', 'Deserialization risk detected'),
            ]
            for pattern, reason in dangerous_patterns:
                if pattern in code:
                    return {'approved': False, 'reason': reason}

            # 3. Validate column references against actual DataFrame
            # Extract string literals that look like column references
            import re as _re
            string_refs = _re.findall(r"""['"]([^'"]{1,80})['"]""", code)
            df_columns_lower = {col.lower(): col for col in df.columns}
            bad_cols = []
            for ref in string_refs:
                ref_lower = ref.lower()
                # Skip common non-column strings
                if ref_lower in ('result', 'index', 'python', 'pandas', 'numpy',
                                 'true', 'false', 'none', 'all', 'any', 'mean',
                                 'sum', 'count', 'min', 'max', 'std', 'var',
                                 'asc', 'desc', 'left', 'right', 'inner', 'outer',
                                 'bar', 'line', 'scatter', 'pie', 'hist', 'box',
                                 'coerce', 'ignore', 'raise', 'records', 'columns',
                                 'first', 'last', 'number', 'object', 'category',
                                 'datetime64', 'float64', 'int64', 'string',''):
                    continue
                # Skip format strings and long text
                if len(ref) > 60 or ' ' in ref and len(ref) > 20:
                    continue
                # Check if this looks like an attempted column reference
                # (used with df[...] patterns)
                if ref_lower not in df_columns_lower and ref in string_refs:
                    # Only flag if it looks like a column access pattern in context
                    col_access_pattern = f"['{ref}']" in code or f'["{ref}"]' in code
                    if col_access_pattern:
                        bad_cols.append(ref)

            if len(bad_cols) >= 2:
                return {
                    'approved': False,
                    'reason': f"References non-existent columns: {', '.join(bad_cols[:3])}. Available: {', '.join(list(df.columns)[:10])}"
                }

            # 4. Reject unreasonably long code (LLM hallucination indicator)
            if len(code) > 5000:
                return {'approved': False, 'reason': f"Generated code too long ({len(code)} chars) — likely hallucinated"}

            # 5. Infinite loop risk
            if 'while True' in code and 'break' not in code:
                return {'approved': False, 'reason': "Potential infinite loop detected (while True without break)"}

            return {'approved': True, 'reason': 'Code review passed'}

        except Exception as e:
            logger.warning("Code review encountered error (approving with caution): %s", e)
            # On review failure, approve but log — don't block the pipeline
            return {'approved': True, 'reason': f'Review error (skipped): {e}'}

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
                logger.info("Result formatted: %s", validation_msg)
            
            return ExecutionResult(
                success=True,
                result=validated_result,
                code=code,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error("Code execution failed: %s", e, exc_info=True)
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
                            save_history: bool = True,
                            analysis_context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Generate code and execute it, with retry logic and history tracking.
        
        Args:
            query: Natural language question
            df: DataFrame to analyze
            model: LLM model for generation
            max_retries: Number of retries on failure
            data_file: Name of the data file being analyzed (for history)
            save_history: Whether to save this execution to history
            analysis_context: Optional analysis strategy from DynamicPlanner
                             Dict with 'strategy' (summary) and 'steps' (list)
            
        Returns:
            ExecutionResult with the final result and history tracking
        """
        last_error = None
        retry_errors = []
        original_generated_code = None
        final_cleaned_code = None
        attempt_count = 0
        
        # Track previous errors for iterative feedback loop (Patent: Verifier feedback)
        previous_error_feedback = None
        
        for attempt in range(max_retries + 1):
            attempt_count = attempt + 1
            
            # Generate code with error feedback from previous attempt if available
            # Patent: "feedback loop to send feedback back through the generation
            # of code phases so the system can revise... generate a correct code
            # version without the need for human intervention"
            gen_result = self.generate_code(
                query, df, model,
                analysis_context=analysis_context,
                error_feedback=previous_error_feedback
            )
            
            # Store original generated code from first attempt
            if original_generated_code is None and gen_result.code:
                original_generated_code = gen_result.code
            
            if not gen_result.is_valid:
                last_error = gen_result.error_message
                retry_errors.append(f"Gen attempt {attempt_count}: {last_error}")
                logger.warning("Code generation failed (attempt %d): %s", attempt_count, last_error)
                # Build error feedback for next attempt
                previous_error_feedback = {
                    'failed_code': gen_result.code or '',
                    'error': last_error,
                    'attempt': attempt_count
                }
                continue
            
            final_cleaned_code = gen_result.code
            
            # PATENT CLAIM 2: Pre-execution code review
            # "agent responsible for confirming and reviewing all automatically
            # generated analytical code before running it"
            review_result = self._review_generated_code(gen_result.code, query, df)
            if not review_result['approved']:
                logger.warning("Code review rejected (attempt %d): %s", attempt_count, review_result['reason'])
                retry_errors.append(f"Review attempt {attempt_count}: {review_result['reason']}")
                previous_error_feedback = {
                    'failed_code': gen_result.code,
                    'error': f"Code review rejected: {review_result['reason']}",
                    'attempt': attempt_count
                }
                last_error = review_result['reason']
                continue
            
            # Execute code in secure sandbox
            exec_result = self.execute_code(gen_result.code, df)
            
            if exec_result.success:
                logger.info("Code execution succeeded in %.1fms", exec_result.execution_time_ms)
                
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
            logger.warning("Code execution failed (attempt %d): %s", attempt_count, last_error)
            
            # Build error feedback for next iteration (iterative correction loop)
            previous_error_feedback = {
                'failed_code': gen_result.code,
                'error': last_error,
                'attempt': attempt_count
            }
        
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
            from backend.core.code_execution_history import get_execution_history
            
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
            logger.error("Failed to save execution to history: %s", e, exc_info=True)
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
        from backend.core.code_execution_history import get_execution_history
        
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
    
    # get_execution_history() removed — dead wrapper, use code_execution_history.py directly
    
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
            # Handle nested/unhashable types gracefully
            try:
                unique_count = int(df[col].nunique())
            except (TypeError, AttributeError):
                # For unhashable types (dicts, lists), count total non-null as approximation
                unique_count = int(df[col].notna().sum())
            
            col_info = {
                'dtype': str(df[col].dtype),
                'non_null_count': int(df[col].notna().sum()),
                'unique_count': unique_count,
                'sample_values': []
            }
            
            # Get sample values (up to 3 unique non-null values)
            try:
                non_null_values = df[col].dropna().unique()
                sample_count = min(3, len(non_null_values))
                col_info['sample_values'] = [str(v)[:50] for v in non_null_values[:sample_count]]
            except (TypeError, AttributeError):
                # For unhashable types, just take first 3 values
                sample_values = df[col].dropna().head(3).tolist()
                col_info['sample_values'] = [str(v)[:50] for v in sample_values]
            
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
                # Check if it's likely an identifier column (use safe unique_count)
                uniqueness_ratio = unique_count / len(df) if len(df) > 0 else 0
                
                # Identifier columns: high uniqueness OR column name ends with common
                # identifier suffixes (domain-agnostic: works for any dataset)
                col_lower = col.lower()
                is_identifier = (
                    uniqueness_ratio > 0.5 or 
                    col_lower.endswith('_name') or
                    col_lower.endswith('_id') or
                    col_lower in ('name', 'title', 'id', 'label', 'key', 'identifier') or
                    col_lower.endswith('_title') or
                    col_lower.endswith('_label') or
                    col_lower.endswith('_key')
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
    
    def _attempt_code_repair(self, code: str) -> Optional[str]:
        """Attempt to auto-repair common syntax errors from small LLMs.
        
        Handles:
        - Unbalanced brackets/parentheses
        - Trailing text after last complete statement
        - Missing closing brackets at end of code
        """
        try:
            lines = code.strip().split('\n')
            
            # Strategy 1: Try truncating to last syntactically valid line
            for end_idx in range(len(lines), 0, -1):
                candidate = '\n'.join(lines[:end_idx]).strip()
                if not candidate:
                    continue
                try:
                    ast.parse(candidate)
                    if 'result' in candidate:
                        return candidate
                except SyntaxError:
                    continue
            
            # Strategy 2: Balance brackets/parens at the end
            repaired = code.rstrip()
            open_parens = repaired.count('(') - repaired.count(')')
            open_brackets = repaired.count('[') - repaired.count(']')
            open_braces = repaired.count('{') - repaired.count('}')
            
            if open_parens > 0:
                repaired += ')' * open_parens
            if open_brackets > 0:
                repaired += ']' * open_brackets
            if open_braces > 0:
                repaired += '}' * open_braces
            
            # Remove trailing comma before closing bracket
            repaired = re.sub(r',\s*\)', ')', repaired)
            repaired = re.sub(r',\s*\]', ']', repaired)
            repaired = re.sub(r',\s*\}', '}', repaired)
            
            try:
                ast.parse(repaired)
                return repaired
            except SyntaxError:
                pass
            
            return None
        except Exception:
            return None

    def _validate_code_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate code syntax"""
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


# =====================================================================
# v2.0 Enterprise Additions — appended; all v1.x code is unchanged
# =====================================================================

import threading
from collections import Counter


@dataclass
class CodeGenerationMetrics:
    """Tracks code generation and execution success rates.

    Attributes:
        total_generations: Number of ``generate_code`` calls.
        successful_generations: Generations that passed validation.
        total_executions: Number of ``execute_code`` calls.
        successful_executions: Executions that returned a result.
        total_retries: Cumulative retry count across all calls.
        total_generation_ms: Cumulative generation latency.
        total_execution_ms: Cumulative execution latency.
        model_usage: Counter mapping model name → call count.

    v2.0 Enterprise Addition.
    """

    total_generations: int = 0
    successful_generations: int = 0
    total_executions: int = 0
    successful_executions: int = 0
    total_retries: int = 0
    total_generation_ms: float = 0.0
    total_execution_ms: float = 0.0
    model_usage: Counter = field(default_factory=Counter)

    # ------------------------------------------------------------------
    def record_generation(
        self, *, success: bool, latency_ms: float = 0.0, model: str = ""
    ) -> None:
        """Record a code-generation attempt."""
        self.total_generations += 1
        if success:
            self.successful_generations += 1
        self.total_generation_ms += latency_ms
        if model:
            self.model_usage[model] += 1

    def record_execution(
        self, *, success: bool, latency_ms: float = 0.0
    ) -> None:
        """Record a code-execution attempt."""
        self.total_executions += 1
        if success:
            self.successful_executions += 1
        self.total_execution_ms += latency_ms

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot."""
        return {
            "total_generations": self.total_generations,
            "generation_success_rate": round(
                self.successful_generations / self.total_generations, 4
            )
            if self.total_generations
            else 0.0,
            "total_executions": self.total_executions,
            "execution_success_rate": round(
                self.successful_executions / self.total_executions, 4
            )
            if self.total_executions
            else 0.0,
            "total_retries": self.total_retries,
            "avg_generation_ms": round(
                self.total_generation_ms / self.total_generations, 2
            )
            if self.total_generations
            else 0.0,
            "avg_execution_ms": round(
                self.total_execution_ms / self.total_executions, 2
            )
            if self.total_executions
            else 0.0,
            "model_usage": dict(self.model_usage),
        }

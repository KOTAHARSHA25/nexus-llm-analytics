from __future__ import annotations

"""Enterprise Sandbox Execution Environment.

Production-ready secure code execution with RestrictedPython, enterprise
security controls, and comprehensive observability.

.. versionadded:: 2.0.0
   Added :class:`ExecutionQuota`, :class:`SandboxCircuitBreaker`,
   :class:`ComplianceLogger`, and policy-based execution.

Backward Compatibility
----------------------
All v1.x public APIs (:class:`EnhancedSandbox`, :class:`Sandbox`) remain
fully compatible.  New enterprise classes are additive and opt-in.
"""

import copy
import hashlib
import logging
import re
import threading
import time
import contextlib
import io
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# Enhanced secure code execution sandbox using RestrictedPython with security guards
from RestrictedPython import compile_restricted_exec
from .security_guards import (
    SecurityGuards, ResourceManager, CodeValidator,
    AuditLogger, RateLimiter, ThreatDetector, ThreatLevel, SecurityPolicy,
    get_audit_logger, get_rate_limiter, get_threat_detector,
)

logger = logging.getLogger(__name__)

__all__ = [
    # v1.x (backward compatible)
    "EnhancedSandbox",
    "Sandbox",
    # v2.0 Enterprise additions
    "ExecutionQuota",
    "SandboxCircuitBreaker",
    "CircuitState",
    "ComplianceLogger",
    "EnterpriseSandbox",
    "ExecutionMetrics",
]


class EnhancedSandbox:
    """
    Production-ready secure code execution environment with comprehensive security measures.
    
    Features:
    - RestrictedPython compilation with custom security guards
    - Resource limits (memory, CPU, execution time)
    - AST-based code validation
    - Pattern-based security scanning
    - Comprehensive logging and audit trail
    - Data isolation with deep copying
    """
    
    def __init__(self, max_memory_mb: int = 512, max_cpu_seconds: int = 120) -> None:
        """
        Initialize the enhanced sandbox with security configurations.
        Increased defaults for ML workloads.
        
        Args:
            max_memory_mb: Maximum memory usage in MB (default 512 for ML)
            max_cpu_seconds: Maximum CPU execution time in seconds (default 120 for ML training)
        """
        self.max_memory_mb = max_memory_mb
        self.max_cpu_seconds = max_cpu_seconds
        
        # Create secure builtins using our security guards
        self.safe_builtins = SecurityGuards.create_safe_builtins()
        
        # Initialize safe module imports
        self.safe_modules = self._initialize_safe_modules()
        
        logger.info(
            "Sandbox initialized: max_memory_mb=%s, max_cpu_seconds=%s, safe_modules=%s",
            max_memory_mb, max_cpu_seconds, list(self.safe_modules.keys()),
        )
    
    def _initialize_safe_modules(self) -> Dict[str, Any]:
        """Build the allow-listed module namespace for sandbox execution.

        Imports data-analysis, ML, statistical, and visualisation libraries
        wrapped in restricted proxy classes that block all file I/O.

        Returns:
            Mapping of short import names (e.g. ``'pd'``, ``'np'``) to
            their restricted module proxies.
        """
        safe_modules = {}
        
        try:
            # Core data analysis
            import pandas as pd
            import numpy as np
            import json
            import math
            import datetime
            
            # High-performance data processing (if available)
            try:
                import polars as pl
                polars_available = True
            except ImportError:
                pl = None
                polars_available = False
            
            # Machine Learning
            from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
            from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            from sklearn.decomposition import PCA, TruncatedSVD
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, silhouette_score
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
            from sklearn.naive_bayes import GaussianNB
            from sklearn.svm import SVC, SVR
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
            
            # Statistical Analysis
            from scipy import stats
            from scipy.stats import pearsonr, spearmanr, chi2_contingency, ttest_ind, f_oneway, mannwhitneyu
            import statsmodels.api as sm
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            from statsmodels.tsa.seasonal import seasonal_decompose
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            from statsmodels.formula.api import ols
            
            logger.info("ML libraries loaded: sklearn=%s, scipy=%s, statsmodels=%s", True, True, True)
            
            # Create restricted pandas - only allow DataFrame operations, no file I/O
            class RestrictedPandas:
                """Proxy exposing pandas DataFrame/Series operations while blocking file I/O."""
                DataFrame = pd.DataFrame
                Series = pd.Series
                concat = pd.concat
                merge = pd.merge
                pivot_table = pd.pivot_table
                cut = pd.cut
                qcut = pd.qcut
                crosstab = pd.crosstab
                get_dummies = pd.get_dummies
                # Note: melt needs special handling as it's called as pd.melt(frame, ...)
                # We'll handle it through __getattr__
                wide_to_long = pd.wide_to_long
                
                # Safe transformation methods - these operate on data, not files
                @staticmethod
                def _safe_apply(df_or_series, func, *args, **kwargs):
                    """Safe apply that only allows simple operations"""
                    return df_or_series.apply(func, *args, **kwargs)
                
                @staticmethod
                def _safe_map(series, func_or_dict):
                    """Safe map for Series"""
                    return series.map(func_or_dict)
                
                @staticmethod
                def _safe_transform(df_or_series, func, *args, **kwargs):
                    """Safe transform operation"""
                    return df_or_series.transform(func, *args, **kwargs)
                
                @staticmethod
                def _safe_agg(df_or_series, func, *args, **kwargs):
                    """Safe aggregation"""
                    return df_or_series.agg(func, *args, **kwargs)
                
                # Explicitly block dangerous functions
                def __getattr__(self, name):
                    dangerous_funcs = {
                        'read_csv', 'read_json', 'read_excel', 'read_sql', 'read_html',
                        'read_xml', 'read_pickle', 'read_parquet', 'read_feather',
                        'to_pickle', 'read_clipboard', 'read_fwf', 'read_table',
                        'to_csv', 'to_json', 'to_excel', 'to_sql', 'to_html'
                    }
                    if name in dangerous_funcs:
                        raise AttributeError(f"Access to pandas.{name} is not allowed for security reasons")
                    
                    # Allow safe methods that were missed
                    safe_methods = {
                        # Missing/conversion methods
                        'isna', 'notna', 'isnull', 'notnull', 'fillna', 'dropna',
                        # Indexing methods  
                        'sort_values', 'sort_index', 'reset_index', 'set_index',
                        # Type conversion methods
                        'to_datetime', 'to_numeric', 'to_string', 'to_dict', 'to_list', 'to_numpy',
                        # Reshaping methods
                        'melt', 'pivot', 'stack', 'unstack', 'transpose',
                        # String operations
                        'Series', 'Categorical',
                        # Date/time utilities
                        'Timedelta', 'Timestamp', 'DatetimeIndex', 'TimedeltaIndex',
                        # Options
                        'set_option', 'get_option', 'option_context'
                    }
                    if name in safe_methods and hasattr(pd, name):
                        return getattr(pd, name)
                    
                    raise AttributeError(f"'{name}' is not available in restricted pandas")
            
            # Create restricted numpy - only mathematical functions
            class RestrictedNumpy:
                """Proxy exposing numpy math and array operations while blocking file I/O."""
                array = np.array
                zeros = np.zeros
                ones = np.ones
                arange = np.arange
                linspace = np.linspace
                mean = np.mean
                std = np.std
                sum = np.sum
                min = np.min
                max = np.max
                abs = np.abs
                sqrt = np.sqrt
                sin = np.sin
                cos = np.cos
                tan = np.tan
                log = np.log
                exp = np.exp
                
                def __getattr__(self, name):
                    dangerous_funcs = {
                        'load', 'save', 'loadtxt', 'savetxt', 'fromfile', 'tofile'
                    }
                    if name in dangerous_funcs:
                        raise AttributeError(f"Access to numpy.{name} is not allowed for security reasons")
                    # Allow other mathematical functions
                    if hasattr(np, name):
                        attr = getattr(np, name)
                        if callable(attr):
                            return attr
                    raise AttributeError(f"'{name}' is not available in restricted numpy")
            
            # Create restricted JSON - no file operations
            class RestrictedJSON:
                """Proxy exposing json.loads/dumps while blocking file-based load/dump."""
                loads = json.loads
                dumps = json.dumps
                
                def __getattr__(self, name):
                    if name in ['load', 'dump']:
                        raise AttributeError(f"File operations json.{name} not allowed for security reasons")
                    raise AttributeError(f"'{name}' is not available in restricted json")
            
            # Polars restricted class (if available)
            if polars_available:
                class RestrictedPolars:
                    """Restricted Polars - safe in-memory operations only"""
                    DataFrame = pl.DataFrame
                    col = pl.col
                    lit = pl.lit
                    concat = pl.concat
                    
                    def __getattr__(self, name):
                        # Block file I/O
                        dangerous_funcs = {
                            'read_csv', 'read_json', 'read_excel', 'read_parquet',
                            'scan_csv', 'scan_parquet', 'scan_ipc',
                        }
                        if name in dangerous_funcs:
                            raise AttributeError(f"Access to polars.{name} is not allowed for security reasons")
                        
                        # Allow safe methods
                        if hasattr(pl, name):
                            return getattr(pl, name)
                        
                        raise AttributeError(f"'{name}' is not available in restricted polars")
            
            # Only expose safe, restricted versions
            safe_modules.update({
                # Core data manipulation
                'pd': RestrictedPandas(),
                # Direct pandas access for apply/map (needed by generated code)
                'pandas': RestrictedPandas(),
                'np': RestrictedNumpy(),
                'json': RestrictedJSON(),
                'math': math,
                'datetime': datetime,
                're': re,
                
                # Machine Learning - Clustering
                'KMeans': KMeans,
                'DBSCAN': DBSCAN,
                'AgglomerativeClustering': AgglomerativeClustering,
                
                # Machine Learning - Classification
                'RandomForestClassifier': RandomForestClassifier,
                'GradientBoostingClassifier': GradientBoostingClassifier,
                'LogisticRegression': LogisticRegression,
                'DecisionTreeClassifier': DecisionTreeClassifier,
                'GaussianNB': GaussianNB,
                'SVC': SVC,
                'KNeighborsClassifier': KNeighborsClassifier,
                
                # Machine Learning - Regression
                'RandomForestRegressor': RandomForestRegressor,
                'GradientBoostingRegressor': GradientBoostingRegressor,
                'LinearRegression': LinearRegression,
                'Ridge': Ridge,
                'Lasso': Lasso,
                'ElasticNet': ElasticNet,
                'DecisionTreeRegressor': DecisionTreeRegressor,
                'SVR': SVR,
                'KNeighborsRegressor': KNeighborsRegressor,
                
                # Dimensionality Reduction
                'PCA': PCA,
                'TruncatedSVD': TruncatedSVD,
                
                # Model Selection & Metrics
                'train_test_split': train_test_split,
                'cross_val_score': cross_val_score,
                'accuracy_score': accuracy_score,
                'precision_score': precision_score,
                'recall_score': recall_score,
                'f1_score': f1_score,
                'mean_squared_error': mean_squared_error,
                'r2_score': r2_score,
                'silhouette_score': silhouette_score,
                
                # Preprocessing
                'StandardScaler': StandardScaler,
                'MinMaxScaler': MinMaxScaler,
                'LabelEncoder': LabelEncoder,
                
                # Statistical Analysis
                'stats': stats,
                'pearsonr': pearsonr,
                'spearmanr': spearmanr,
                'chi2_contingency': chi2_contingency,
                'ttest_ind': ttest_ind,
                'f_oneway': f_oneway,
                'mannwhitneyu': mannwhitneyu,
                'pairwise_tukeyhsd': pairwise_tukeyhsd,
                'ols': ols,
                
                # Time Series
                'sm': sm,
                'ARIMA': ARIMA,
                'ExponentialSmoothing': ExponentialSmoothing,
                'seasonal_decompose': seasonal_decompose,
            })
            
            # Add polars if available
            if polars_available:
                safe_modules['pl'] = RestrictedPolars()
                safe_modules['polars'] = RestrictedPolars()
                logger.info("Polars loaded")
            
            # Try to add visualization libraries (if installed)
            try:
                import plotly.express as px
                import plotly.graph_objects as go
                safe_modules.update({
                    'px': px,
                    'go': go,
                })
                logger.info("Plotly loaded")
            except ImportError:
                logger.info("Plotly not available")
            
            # Add matplotlib with restricted pyplot (block file saving)
            try:
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend (safer)
                import matplotlib.pyplot as plt
                import types
                
                # Create a shallow copy of the plt module to avoid mutating the global module
                safe_plt = types.ModuleType('matplotlib.pyplot')
                for attr in dir(plt):
                    if not attr.startswith('__'):
                        setattr(safe_plt, attr, getattr(plt, attr))
                
                # Override dangerous methods on the copy only
                def block_savefig(*args, **kwargs):
                    raise AttributeError("matplotlib.pyplot.savefig() is not allowed for security reasons")
                
                def block_save(*args, **kwargs):
                    raise AttributeError("matplotlib.pyplot.save() is not allowed for security reasons")
                
                safe_plt.savefig = block_savefig
                safe_plt.save = block_save
                
                safe_modules.update({
                    'plt': safe_plt,  # Expose safe copy, not the global module
                    'matplotlib': matplotlib,
                })
                
                logger.info("Matplotlib loaded")
            except ImportError:
                logger.info("Matplotlib not available")
            
            # Add seaborn (statistical visualization)
            try:
                import seaborn as sns
                safe_modules['sns'] = sns
                logger.info("Seaborn loaded")
            except ImportError:
                logger.info("Seaborn not available")
            
            logger.info("Safe modules loaded: %s", list(safe_modules.keys()))
            
        except ImportError as e:
            logger.warning("Module import failed: %s", e)
        
        # ML specific imports
        try:
            import sklearn.feature_selection
            import sklearn.pipeline
            import sklearn.impute
            import sklearn.metrics
            import scipy.optimize
            
            safe_modules['sklearn.feature_selection'] = self._create_restricted_proxy(sklearn.feature_selection)
            safe_modules['sklearn.pipeline'] = self._create_restricted_proxy(sklearn.pipeline)
            safe_modules['sklearn.impute'] = self._create_restricted_proxy(sklearn.impute)
            safe_modules['sklearn.metrics'] = self._create_restricted_proxy(sklearn.metrics)
            safe_modules['scipy.optimize'] = self._create_restricted_proxy(scipy.optimize)
        except ImportError:
            pass

        return safe_modules
    
    def _create_restricted_proxy(self, module: Any) -> Any:
        """Create a restricted proxy for a module to block dangerous access."""
        class RestrictedModuleProxy:
            def __repr__(self):
                return f"<RestrictedProxy for {module.__name__}>"
            
            def __getattr__(self, name):
                # Block private attributes
                if name.startswith("_"):
                    raise AttributeError(f"Access to private attribute '{name}' is not allowed in sandbox")
                
                # Block dangerous names
                if name in ['load', 'dump', 'save', 'open', 'file', 'exec', 'eval', 'pickle', 'joblib']:
                    raise AttributeError(f"Access to '{name}' is not allowed in sandbox")
                
                # Check if attribute exists
                if hasattr(module, name):
                    return getattr(module, name)
                
                raise AttributeError(f"'{name}' not found in {module.__name__}")
                
        return RestrictedModuleProxy()
    
    def validate_code(self, code: str) -> tuple[bool, str]:
        """
        Comprehensive code validation before execution.
        
        Args:
            code: Python code string to validate
            
        Returns:
            tuple: (is_valid, error_message)
        """
        # Step 1: AST validation
        ast_valid, ast_msg = CodeValidator.validate_ast(code)
        if not ast_valid:
            return False, f"AST validation failed: {ast_msg}"
        
        # Step 2: Pattern validation
        pattern_valid, pattern_msg = CodeValidator.validate_code_patterns(code)
        if not pattern_valid:
            return False, f"Pattern validation failed: {pattern_msg}"
        
        # Step 3: Length and complexity checks
        if len(code) > 10000:  # 10KB limit
            return False, "Code too long (max 10KB)"
        
        if code.count('\n') > 500:  # 500 lines limit
            return False, "Too many lines (max 500)"
        
        return True, "Code validation passed"
    
    def _analyze_code_safety(self, code: str) -> bool:
        """
        Perform deep code analysis for security threats.
        
        Args:
            code: Python code to analyze
            
        Returns:
            True if code appears safe, False otherwise
        """
        # Check for dangerous patterns that might bypass RestrictedPython
        dangerous_patterns = [
            # Direct bytecode manipulation
            r'compile\s*\(',
            r'eval\s*\(',
            r'exec\s*\(',
            
            # Import system bypasses
            r'__import__\s*\(',
            r'importlib',
            r'__builtins__',
            r'globals\s*\(\)',
            r'locals\s*\(\)',
            
            # Attribute access bypasses
            r'getattr\s*\(',
            r'setattr\s*\(',
            r'hasattr\s*\(',
            r'delattr\s*\(',
            
            # Class and function manipulation
            r'\.__class__',
            r'\.__bases__',
            r'\.__subclasses__',
            r'\.__mro__',
            
            # File system access
            r'open\s*\(',
            r'file\s*\(',
            
            # Network access
            r'socket\.',
            r'urllib',
            r'requests',
            
            # Process execution
            r'subprocess',
            r'os\.system',
            r'os\.popen',
            
            # Code generation - be more specific about lambda
            r'lambda\s*:[^,\)]*(?:__|\beval\b|\bexec\b|\bcompile\b)',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                logger.warning(
                    "Dangerous pattern detected: pattern=%s, snippet=%s",
                    pattern, code[:100],
                )
                return False
        
        return True
    
    def _create_restricted_globals(self) -> Dict[str, Any]:
        """Create a highly restricted global namespace.

        Combines safe builtins, restricted modules, and RestrictedPython
        guard functions (``_print_``, ``_getattr_``, ``_getitem_``,
        ``_write_``) into a single globals dict for ``exec()``.

        Returns:
            Globals mapping suitable for sandboxed ``exec()``.
        """
        globals_dict = {
            '__builtins__': self.safe_builtins,
            '__name__': '__main__',
            '__doc__': None,
        }
        
        # Add only the safe modules (already restricted)
        globals_dict.update(self.safe_modules)
        
        # Add RestrictedPython guards for print and getattr
        from RestrictedPython.PrintCollector import PrintCollector
        globals_dict['_print_'] = PrintCollector
        globals_dict['_getattr_'] = SecurityGuards.safer_getattr
        globals_dict['_getitem_'] = SecurityGuards.safe_getitem
        globals_dict['_write_'] = SecurityGuards.guarded_write
        globals_dict['__import__'] = self.safe_builtins['__import__']
        
        return globals_dict
    
    def _is_safe_type(self, value: Any) -> bool:
        """Check if a value is of a safe type for the sandbox.

        Recursively inspects containers up to a depth/size limit.

        Args:
            value: The Python object to check.

        Returns:
            ``True`` if the value (and its contents) are safe primitives,
            pandas DataFrames/Series, or numpy arrays.
        """
        safe_types = (
            int, float, str, bool, list, dict, tuple, set, frozenset,
            type(None)
        )
        
        # Check the type directly
        if type(value) in safe_types:
            return True
        
        # Allow pandas DataFrames and Series (core data types for analysis)
        try:
            import pandas as pd
            import numpy as np
            if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
                return True
        except ImportError:
            pass
        
        # For containers, check contents recursively (but limit depth)
        if isinstance(value, (list, tuple)):
            return all(self._is_safe_type(item) for item in value[:100])  # Limit to 100 items
        elif isinstance(value, dict):
            if len(value) > 100:  # Limit dict size
                return False
            return all(
                isinstance(k, (str, int)) and self._is_safe_type(v) 
                for k, v in value.items()
            )
        elif isinstance(value, set):
            return len(value) <= 100 and all(self._is_safe_type(item) for item in value)
        
        # Anything else is potentially unsafe
        return False
    
    def execute(self, code: str, data: Any = None, extra_globals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute code in the secure sandbox environment.
        
        Args:
            code: Python code string to execute
            data: Data object (will be deep copied for safety)
            extra_globals: Additional global variables (optional)
            
        Returns:
            Dictionary with 'result' on success or 'error' on failure
        """
        execution_id = f"exec_{int(time.time() * 1000)}"
        start_time = time.time()
        
        logger.info(
            "Sandbox execution start: id=%s, code_length=%s, has_data=%s",
            execution_id, len(code), data is not None,
        )
        
        try:
            # Step 1: Validate code
            is_valid, validation_msg = self.validate_code(code)
            if not is_valid:
                logger.warning(
                    "Code validation failed: id=%s, reason=%s",
                    execution_id, validation_msg,
                )
                return {"error": f"Code validation failed: {validation_msg}"}
            
            # Step 1.5: Deep security analysis
            if not self._analyze_code_safety(code):
                logger.error(
                    "Security analysis failed: id=%s, snippet=%s",
                    execution_id, code[:200],
                )
                return {"error": "Code contains potentially dangerous security patterns"}
            
            # Step 2: Prepare highly restricted execution environment
            globals_dict = self._create_restricted_globals()
            
            # Add extra globals if provided (with strict validation)
            if extra_globals:
                for key, value in extra_globals.items():
                    # Much stricter validation
                    if (key.startswith('_') or 
                        key in SecurityGuards.DANGEROUS_ATTRIBUTES or
                        key.startswith('__') or
                        key in ['globals', 'locals', 'vars', 'dir', 'eval', 'exec', 'compile']):
                        logger.warning("Skipping dangerous global variable: %s", key)
                        continue
                    
                    # Type validation - only allow safe types
                    if not self._is_safe_type(value):
                        logger.warning("Skipping unsafe type for global variable %s: %s", key, type(value))
                        continue
                        
                    globals_dict[key] = value
            
            # Deep copy data for complete isolation
            if data is not None:
                try:
                    # Import pandas for isinstance check
                    import pandas as pd
                    
                    data_copy = copy.deepcopy(data)
                    # Convert DataFrames to RestrictedDataFrame for security
                    if isinstance(data_copy, pd.DataFrame):
                        # Monkey-patch the dangerous methods to block them
                        def block_method(method_name):
                            def blocked(*args, **kwargs):
                                raise AttributeError(f"DataFrame.{method_name}() is not allowed for security reasons")
                            return blocked
                        
                        for method in ['to_csv', 'to_json', 'to_excel', 'to_sql', 'to_html',
                                      'to_pickle', 'to_parquet', 'to_feather', 'to_hdf']:
                            setattr(data_copy, method, block_method(method))
                    
                    globals_dict['data'] = data_copy
                    logger.info(
                        "Data copied: id=%s, type=%s",
                        execution_id, type(data_copy).__name__,
                    )
                except Exception as e:
                    logger.error(
                        "Data copy failed: id=%s, error=%s",
                        execution_id, e, exc_info=True,
                    )
                    return {"error": f"Failed to copy data: {str(e)}"}
            
            # Step 3: Compile code with RestrictedPython and strict policy
            try:
                # Use compile_restricted_exec with enhanced security policy
                compiled_result = compile_restricted_exec(code, '<sandbox>')
                
                # Check for compilation errors
                if compiled_result.errors:
                    error_msg = "; ".join(compiled_result.errors)
                    logger.error(
                        "Compilation errors: id=%s, errors=%s",
                        execution_id, compiled_result.errors,
                    )
                    return {"error": f"Compilation errors: {error_msg}"}
                
                if compiled_result.warnings:
                    logger.warning(
                        "Compilation warnings: id=%s, warnings=%s",
                        execution_id, compiled_result.warnings,
                    )
                
                byte_code = compiled_result.code
                if byte_code is None:
                    return {"error": "Code compilation failed - potentially unsafe code detected"}
                    
            except SyntaxError as e:
                logger.error(
                    "Compilation failed: id=%s, error=%s",
                    execution_id, e, exc_info=True,
                )
                return {"error": f"Syntax error: {str(e)}"}
            except Exception as e:
                logger.error(
                    "Compilation error: id=%s, error=%s",
                    execution_id, e, exc_info=True,
                )
                return {"error": f"Compilation error: {str(e)}"}
            
            # Step 4: Execute with enhanced resource limits and monitoring
            local_vars = {}
            
            try:
                # Apply resource limits (use constructor values directly, not hard-capped)
                stdout_capture = io.StringIO()
                with ResourceManager.limit_resources(
                    max_memory_mb=self.max_memory_mb,
                    max_cpu_seconds=self.max_cpu_seconds
                ), contextlib.redirect_stdout(stdout_capture):
                    # Execute compiled bytecode with enhanced monitoring
                    exec(byte_code, globals_dict, local_vars)
                
                execution_time = time.time() - start_time
                
                logger.info(
                    "Sandbox execution success: id=%s, time=%s, vars=%s",
                    execution_id, execution_time, list(local_vars.keys()),
                )
                
                # Clean up potentially dangerous variables from result
                cleaned_vars = {}
                for key, value in local_vars.items():
                    if not key.startswith('_') and key not in SecurityGuards.DANGEROUS_ATTRIBUTES:
                        cleaned_vars[key] = value
                
                return {
                    "success": True,
                    "result": cleaned_vars,
                    "std_out": stdout_capture.getvalue(),
                    "execution_time": execution_time,
                    "execution_id": execution_id
                }
                
            except TimeoutError as e:
                logger.error(
                    "Execution timeout: id=%s, max_seconds=%s",
                    execution_id, self.max_cpu_seconds, exc_info=True,
                )
                return {"error": f"Execution timeout: {str(e)}"}
                
            except MemoryError as e:
                logger.error(
                    "Memory exceeded: id=%s, max_mb=%s",
                    execution_id, self.max_memory_mb, exc_info=True,
                )
                return {"error": f"Memory limit exceeded: {str(e)}"}
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                logger.error(
                    "Execution error: id=%s, time=%s, error=%s",
                    execution_id, execution_time, e, exc_info=True,
                )
                
                # Return user-friendly error without exposing system details
                return {"error": f"Execution error: {str(e)}"}
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "Sandbox critical error: id=%s, time=%s, error=%s",
                execution_id, execution_time, e, exc_info=True,
            )
            return {"error": f"Sandbox error: {str(e)}"}

# Maintain backward compatibility
class Sandbox(EnhancedSandbox):
    """Backward-compatible alias for :class:`EnhancedSandbox`.

    .. deprecated:: 2.0.0
        Use :class:`EnhancedSandbox` or :class:`EnterpriseSandbox` directly.
    """

    def __init__(self) -> None:
        super().__init__(max_memory_mb=256, max_cpu_seconds=30)
        logger.info("Using enhanced sandbox with backward compatibility")


# =============================================================================
# ENTERPRISE: EXECUTION QUOTA MANAGEMENT
# =============================================================================

@dataclass
class ExecutionMetrics:
    """Real-time metrics for a single sandbox execution.

    Attributes:
        execution_id: Unique identifier.
        code_hash: SHA-256 hash of the executed code.
        start_time: Unix timestamp of execution start.
        end_time: Unix timestamp of execution end.
        duration_seconds: Wall-clock duration.
        memory_peak_mb: Peak memory usage (estimated).
        code_length: Source code length in characters.
        line_count: Number of lines in the source code.
        success: Whether execution completed without errors.
        error_type: Classification of the error (if any).
        threat_level: Highest threat level detected during scans.
        variables_produced: Number of variables in the local namespace.
    """
    execution_id: str = ""
    code_hash: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0
    memory_peak_mb: float = 0.0
    code_length: int = 0
    line_count: int = 0
    success: bool = False
    error_type: str = ""
    threat_level: str = "none"
    variables_produced: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "execution_id": self.execution_id,
            "code_hash": self.code_hash,
            "duration_seconds": round(self.duration_seconds, 4),
            "code_length": self.code_length,
            "line_count": self.line_count,
            "success": self.success,
            "error_type": self.error_type,
            "threat_level": self.threat_level,
            "variables_produced": self.variables_produced,
        }


class ExecutionQuota:
    """Per-identity execution quota manager.

    Tracks daily and hourly execution counts, cumulative CPU time,
    and code volume to enforce configurable limits.

    Args:
        max_daily_executions: Maximum executions per identity per day.
        max_hourly_executions: Maximum executions per identity per hour.
        max_daily_cpu_seconds: Maximum cumulative CPU seconds per day.
        max_code_volume_kb: Maximum cumulative code volume per day.
    """

    def __init__(
        self,
        max_daily_executions: int = 1000,
        max_hourly_executions: int = 200,
        max_daily_cpu_seconds: int = 3600,
        max_code_volume_kb: int = 10_000,
    ) -> None:
        self._max_daily = max_daily_executions
        self._max_hourly = max_hourly_executions
        self._max_cpu = max_daily_cpu_seconds
        self._max_volume = max_code_volume_kb * 1024
        self._lock = threading.Lock()
        self._daily: Dict[str, Dict] = defaultdict(lambda: {
            "count": 0, "cpu_seconds": 0.0, "code_bytes": 0, "reset_at": 0.0
        })
        self._hourly: Dict[str, Dict] = defaultdict(lambda: {
            "count": 0, "reset_at": 0.0
        })
        logger.info(
            "ExecutionQuota initialized: daily=%d, hourly=%d, cpu=%ds",
            max_daily_executions, max_hourly_executions, max_daily_cpu_seconds,
        )

    def check_and_consume(
        self, identity: str, code_length: int = 0, cpu_seconds: float = 0.0
    ) -> tuple:
        """Check quota and consume one execution unit.

        Args:
            identity: The identity (user ID, API key, etc.).
            code_length: Length of code being executed.
            cpu_seconds: Estimated CPU time.

        Returns:
            ``(allowed, reason)`` tuple.
        """
        now = time.time()
        with self._lock:
            daily = self._daily[identity]
            hourly = self._hourly[identity]

            # Reset daily counters at midnight
            day_start = now - (now % 86400)
            if daily["reset_at"] < day_start:
                daily["count"] = 0
                daily["cpu_seconds"] = 0.0
                daily["code_bytes"] = 0
                daily["reset_at"] = now

            # Reset hourly counters
            hour_start = now - (now % 3600)
            if hourly["reset_at"] < hour_start:
                hourly["count"] = 0
                hourly["reset_at"] = now

            if daily["count"] >= self._max_daily:
                return False, f"Daily execution limit reached ({self._max_daily})"
            if hourly["count"] >= self._max_hourly:
                return False, f"Hourly execution limit reached ({self._max_hourly})"
            if daily["cpu_seconds"] + cpu_seconds > self._max_cpu:
                return False, f"Daily CPU quota exhausted ({self._max_cpu}s)"
            if daily["code_bytes"] + code_length > self._max_volume:
                return False, "Daily code volume quota exhausted"

            daily["count"] += 1
            daily["cpu_seconds"] += cpu_seconds
            daily["code_bytes"] += code_length
            hourly["count"] += 1
            return True, "Quota available"

    def get_usage(self, identity: str) -> Dict[str, Any]:
        """Get current quota usage for an identity.

        Args:
            identity: The identity to query.

        Returns:
            Dict with daily/hourly counts and remaining quotas.
        """
        with self._lock:
            daily = self._daily.get(identity, {})
            hourly = self._hourly.get(identity, {})
            return {
                "daily_used": daily.get("count", 0),
                "daily_limit": self._max_daily,
                "daily_remaining": self._max_daily - daily.get("count", 0),
                "hourly_used": hourly.get("count", 0),
                "hourly_limit": self._max_hourly,
                "cpu_used_seconds": round(daily.get("cpu_seconds", 0.0), 2),
                "cpu_limit_seconds": self._max_cpu,
            }


# =============================================================================
# ENTERPRISE: CIRCUIT BREAKER PATTERN
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states following the standard pattern."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failures exceeded threshold, blocking calls
    HALF_OPEN = "half_open"  # Testing if service has recovered


class SandboxCircuitBreaker:
    """Circuit breaker for sandbox execution to prevent cascade failures.

    Monitors execution failure rates and temporarily stops accepting
    new executions when failures exceed a configurable threshold.
    After a recovery timeout the breaker enters half-open state and
    allows a single probe execution to test recovery.

    Args:
        failure_threshold: Number of consecutive failures to trip open.
        recovery_timeout_seconds: Seconds to wait before probing recovery.
        half_open_max_calls: Maximum probe calls in half-open state.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 60.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self._threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_seconds
        self._half_open_max = half_open_max_calls
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._total_trips = 0
        logger.info(
            "SandboxCircuitBreaker initialized: threshold=%d, recovery=%ds",
            failure_threshold, recovery_timeout_seconds,
        )

    @property
    def state(self) -> CircuitState:
        """Current circuit state, auto-transitioning OPEN → HALF_OPEN on timeout."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time > self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("Circuit breaker → HALF_OPEN (recovery timeout elapsed)")
            return self._state

    def allow_request(self) -> bool:
        """Check whether a new execution is permitted.

        Returns:
            ``True`` if the request may proceed.
        """
        current = self.state
        if current == CircuitState.CLOSED:
            return True
        if current == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self._half_open_max:
                    self._half_open_calls += 1
                    return True
                return False
        return False  # OPEN

    def record_success(self) -> None:
        """Record a successful execution."""
        with self._lock:
            self._failure_count = 0
            self._success_count += 1
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                logger.info("Circuit breaker → CLOSED (recovery confirmed)")

    def record_failure(self) -> None:
        """Record a failed execution."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._total_trips += 1
                logger.warning("Circuit breaker → OPEN (half-open probe failed)")
            elif self._failure_count >= self._threshold:
                self._state = CircuitState.OPEN
                self._total_trips += 1
                logger.warning(
                    "Circuit breaker → OPEN (failure_count=%d >= threshold=%d)",
                    self._failure_count, self._threshold,
                )

    def get_statistics(self) -> Dict[str, Any]:
        """Return circuit breaker statistics."""
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "total_trips": self._total_trips,
                "recovery_timeout_seconds": self._recovery_timeout,
            }


# =============================================================================
# ENTERPRISE: COMPLIANCE LOGGER
# =============================================================================

class ComplianceLogger:
    """SOC2/HIPAA-grade compliance logging for sandbox operations.

    Wraps :class:`AuditLogger` with pre-formatted compliance events,
    code hashing, and retention policy enforcement.

    Args:
        audit_logger: Underlying audit logger instance.
    """

    def __init__(self, audit_logger: Optional[AuditLogger] = None) -> None:
        self._audit = audit_logger or get_audit_logger()

    @staticmethod
    def hash_code(code: str) -> str:
        """Compute a SHA-256 hash of source code for audit records.

        Args:
            code: Raw source code string.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    def log_execution_start(
        self, execution_id: str, code_hash: str, identity: str = "system",
    ) -> None:
        """Log the start of a sandboxed execution."""
        self._audit.log_execution(
            actor=identity,
            action="execution_start",
            resource=code_hash,
            outcome="pending",
            execution_id=execution_id,
        )

    def log_execution_end(
        self, execution_id: str, code_hash: str, success: bool,
        duration: float, identity: str = "system", error: str = "",
    ) -> None:
        """Log the completion of a sandboxed execution."""
        self._audit.log_execution(
            actor=identity,
            action="execution_end",
            resource=code_hash,
            outcome="success" if success else "failure",
            execution_id=execution_id,
            duration_seconds=round(duration, 4),
            error=error,
        )

    def log_policy_violation(
        self, execution_id: str, policy_name: str, violation: str,
    ) -> None:
        """Log a security policy violation."""
        self._audit.log_access_denied(
            actor="sandbox",
            resource=f"policy:{policy_name}",
            reason=violation,
            execution_id=execution_id,
        )


# =============================================================================
# ENTERPRISE: FULL-FEATURED SANDBOX
# =============================================================================

class EnterpriseSandbox(EnhancedSandbox):
    """Enterprise-grade sandbox with integrated security controls.

    Extends :class:`EnhancedSandbox` with:

    * **Circuit breaker** — prevents cascade failures.
    * **Execution quotas** — per-identity daily/hourly limits.
    * **Rate limiting** — sliding-window DoS protection.
    * **Threat detection** — multi-layer code scanning before execution.
    * **Compliance logging** — SOC2/HIPAA audit trail.
    * **Security policies** — configurable execution constraints.
    * **Execution metrics** — detailed per-execution telemetry.

    Fully backward-compatible with :class:`EnhancedSandbox`.

    Args:
        max_memory_mb: Maximum memory in MB.
        max_cpu_seconds: Maximum CPU wall-clock seconds.
        policy: Optional :class:`SecurityPolicy` to enforce.
        enable_circuit_breaker: Activate the circuit breaker.
        enable_quotas: Activate execution quotas.
        enable_rate_limiting: Activate rate limiting.
        enable_threat_detection: Activate pre-execution threat scanning.
        enable_compliance_logging: Activate compliance audit logging.
    """

    def __init__(
        self,
        max_memory_mb: int = 512,
        max_cpu_seconds: int = 120,
        policy: Optional[SecurityPolicy] = None,
        enable_circuit_breaker: bool = True,
        enable_quotas: bool = True,
        enable_rate_limiting: bool = True,
        enable_threat_detection: bool = True,
        enable_compliance_logging: bool = True,
    ) -> None:
        super().__init__(max_memory_mb=max_memory_mb, max_cpu_seconds=max_cpu_seconds)

        self.policy = policy or SecurityPolicy.production()
        self._circuit_breaker = SandboxCircuitBreaker() if enable_circuit_breaker else None
        self._quota = ExecutionQuota() if enable_quotas else None
        self._rate_limiter = get_rate_limiter() if enable_rate_limiting else None
        self._threat_detector = get_threat_detector() if enable_threat_detection else None
        self._compliance = ComplianceLogger() if enable_compliance_logging else None

        self._lock = threading.Lock()
        self._execution_history: List[ExecutionMetrics] = []
        self._total_executions = 0
        self._total_successes = 0
        self._total_failures = 0
        self._total_blocked = 0

        logger.info(
            "EnterpriseSandbox initialized: policy=%s, circuit_breaker=%s, "
            "quotas=%s, rate_limit=%s, threat_detect=%s, compliance=%s",
            self.policy.name, enable_circuit_breaker, enable_quotas,
            enable_rate_limiting, enable_threat_detection, enable_compliance_logging,
        )

    def execute(
        self,
        code: str,
        data: Any = None,
        extra_globals: Optional[Dict[str, Any]] = None,
        identity: str = "system",
    ) -> Dict[str, Any]:
        """Execute code with full enterprise security controls.

        Overrides :meth:`EnhancedSandbox.execute` to add:
        - Circuit breaker check
        - Rate limit check
        - Quota check
        - Policy validation
        - Threat scanning
        - Compliance logging
        - Execution metrics

        Args:
            code: Python source code to execute.
            data: Data object (deep-copied for isolation).
            extra_globals: Additional global variables.
            identity: Identity of the executor (for quotas/rate limits).

        Returns:
            Dict with ``result`` on success or ``error`` on failure.
        """
        execution_id = f"exec_{int(time.time() * 1000)}"
        code_hash = ComplianceLogger.hash_code(code)
        start_time = time.time()

        metrics = ExecutionMetrics(
            execution_id=execution_id,
            code_hash=code_hash,
            start_time=start_time,
            code_length=len(code),
            line_count=code.count("\n") + 1,
        )

        # --- Pre-execution checks ---

        # 1. Circuit breaker
        if self._circuit_breaker and not self._circuit_breaker.allow_request():
            self._total_blocked += 1
            metrics.error_type = "circuit_breaker_open"
            self._record_metrics(metrics)
            return {"error": "Sandbox circuit breaker is OPEN — too many recent failures. Retry later."}

        # 2. Rate limiting
        if self._rate_limiter:
            try:
                self._rate_limiter.check(identity)
            except Exception as e:
                self._total_blocked += 1
                metrics.error_type = "rate_limit_exceeded"
                self._record_metrics(metrics)
                return {"error": str(e)}

        # 3. Execution quota
        if self._quota:
            allowed, reason = self._quota.check_and_consume(
                identity, code_length=len(code), cpu_seconds=self.max_cpu_seconds
            )
            if not allowed:
                self._total_blocked += 1
                metrics.error_type = "quota_exceeded"
                self._record_metrics(metrics)
                return {"error": f"Execution quota exceeded: {reason}"}

        # 4. Policy validation
        policy_ok, policy_msg = self.policy.validate_code(code)
        if not policy_ok:
            self._total_blocked += 1
            metrics.error_type = "policy_violation"
            self._record_metrics(metrics)
            if self._compliance:
                self._compliance.log_policy_violation(execution_id, self.policy.name, policy_msg)
            return {"error": f"Policy violation: {policy_msg}"}

        # 5. Threat detection
        if self._threat_detector:
            threats = self._threat_detector.scan(code, source=identity)
            critical = [t for t in threats if t.level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL)]
            if critical:
                self._total_blocked += 1
                metrics.error_type = "threat_detected"
                metrics.threat_level = critical[0].level.value
                self._record_metrics(metrics)
                descs = "; ".join(t.description for t in critical[:3])
                return {"error": f"Security threat detected: {descs}"}

        # 6. Compliance logging (start)
        if self._compliance:
            self._compliance.log_execution_start(execution_id, code_hash, identity)

        # --- Execute via parent ---
        result = super().execute(code, data, extra_globals)

        # --- Post-execution bookkeeping ---
        end_time = time.time()
        metrics.end_time = end_time
        metrics.duration_seconds = end_time - start_time
        metrics.success = "error" not in result
        metrics.error_type = "" if metrics.success else "execution_error"
        metrics.variables_produced = len(result.get("result", {})) if metrics.success else 0

        if metrics.success:
            self._total_successes += 1
            if self._circuit_breaker:
                self._circuit_breaker.record_success()
        else:
            self._total_failures += 1
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()

        self._total_executions += 1
        self._record_metrics(metrics)

        # Compliance logging (end)
        if self._compliance:
            self._compliance.log_execution_end(
                execution_id, code_hash, metrics.success,
                metrics.duration_seconds, identity,
                error=result.get("error", ""),
            )

        return result

    def _record_metrics(self, metrics: ExecutionMetrics) -> None:
        with self._lock:
            self._execution_history.append(metrics)
            # Keep last 1000 executions
            if len(self._execution_history) > 1000:
                self._execution_history = self._execution_history[-1000:]

    def get_execution_metrics(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent execution metrics.

        Args:
            limit: Maximum number of recent metrics to return.

        Returns:
            List of serialized :class:`ExecutionMetrics` dicts.
        """
        with self._lock:
            return [m.to_dict() for m in self._execution_history[-limit:]]

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive sandbox statistics.

        Returns:
            Dict with execution counts, success rate, circuit breaker
            state, quota usage summary, and policy information.
        """
        stats: Dict[str, Any] = {
            "total_executions": self._total_executions,
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
            "total_blocked": self._total_blocked,
            "success_rate": round(
                self._total_successes / max(self._total_executions, 1) * 100, 2
            ),
            "policy": self.policy.name,
        }
        if self._circuit_breaker:
            stats["circuit_breaker"] = self._circuit_breaker.get_statistics()
        if self._threat_detector:
            stats["threat_detection"] = self._threat_detector.get_statistics()
        return stats

    # Backward compatibility: make 3-arg execute calls work transparently
    # (identity defaults to "system" so old code doesn't break)

import logging
import copy
import time
import traceback
from typing import Any, Dict, Optional

# Enhanced secure code execution sandbox using RestrictedPython with security guards
from RestrictedPython import compile_restricted_exec
from RestrictedPython.Guards import safe_builtins, safe_globals
from .security_guards import SecurityGuards, ResourceManager, CodeValidator

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
    
    def __init__(self, max_memory_mb: int = 256, max_cpu_seconds: int = 30):
        """
        Initialize the enhanced sandbox with security configurations.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            max_cpu_seconds: Maximum CPU execution time in seconds
        """
        self.max_memory_mb = max_memory_mb
        self.max_cpu_seconds = max_cpu_seconds
        
        # Create secure builtins using our security guards
        self.safe_builtins = SecurityGuards.create_safe_builtins()
        
        # Initialize safe module imports
        self.safe_modules = self._initialize_safe_modules()
        
        logging.info({
            "event": "sandbox_initialized",
            "max_memory_mb": max_memory_mb,
            "max_cpu_seconds": max_cpu_seconds,
            "safe_modules": list(self.safe_modules.keys())
        })
    
    def _initialize_safe_modules(self) -> Dict[str, Any]:
        """Initialize safely restricted modules for sandbox use"""
        safe_modules = {}
        
        try:
            # Create restricted pandas with only safe functions
            import pandas as pd
            import numpy as np
            import json
            import math
            import datetime
            import re
            
            # Create restricted pandas - only allow DataFrame operations, no file I/O
            class RestrictedPandas:
                DataFrame = pd.DataFrame
                Series = pd.Series
                concat = pd.concat
                merge = pd.merge
                pivot_table = pd.pivot_table
                cut = pd.cut
                qcut = pd.qcut
                crosstab = pd.crosstab
                get_dummies = pd.get_dummies
                
                # Explicitly block dangerous functions
                def __getattr__(self, name):
                    dangerous_funcs = {
                        'read_csv', 'read_json', 'read_excel', 'read_sql', 'read_html',
                        'read_xml', 'read_pickle', 'read_parquet', 'read_feather',
                        'to_pickle', 'read_clipboard', 'read_fwf', 'read_table'
                    }
                    if name in dangerous_funcs:
                        raise AttributeError(f"Access to pandas.{name} is not allowed for security reasons")
                    raise AttributeError(f"'{name}' is not available in restricted pandas")
            
            # Create restricted numpy - only mathematical functions
            class RestrictedNumpy:
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
                loads = json.loads
                dumps = json.dumps
                
                def __getattr__(self, name):
                    if name in ['load', 'dump']:
                        raise AttributeError(f"File operations json.{name} not allowed for security reasons")
                    raise AttributeError(f"'{name}' is not available in restricted json")
            
            # Only expose safe, restricted versions
            safe_modules.update({
                'pd': RestrictedPandas(),
                'np': RestrictedNumpy(),
                'json': RestrictedJSON(),
                'math': math,  # Math module is generally safe
                'datetime': datetime,  # Datetime is safe
                're': re,  # Regex is safe but could be used for ReDoS - TODO: add pattern validation
            })
            
            logging.info({"event": "safe_modules_loaded", "modules": list(safe_modules.keys())})
            
        except ImportError as e:
            logging.warning({"event": "module_import_failed", "error": str(e)})
        
        return safe_modules
    
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
        
        import re
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                logging.warning({
                    "event": "dangerous_pattern_detected",
                    "pattern": pattern,
                    "code_snippet": code[:100]  # Log first 100 chars for debugging
                })
                return False
        
        return True
    
    def _create_restricted_globals(self) -> Dict[str, Any]:
        """Create a highly restricted global namespace."""
        globals_dict = {
            '__builtins__': self.safe_builtins,
            '__name__': '__main__',
            '__doc__': None,
        }
        
        # Add only the safe modules (already restricted)
        globals_dict.update(self.safe_modules)
        
        return globals_dict
    
    def _is_safe_type(self, value: Any) -> bool:
        """Check if a value is of a safe type for the sandbox."""
        safe_types = (
            int, float, str, bool, list, dict, tuple, set, frozenset,
            type(None)
        )
        
        # Check the type directly
        if type(value) in safe_types:
            return True
        
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
        
        logging.info({
            "event": "sandbox_execution_start",
            "execution_id": execution_id,
            "code_length": len(code),
            "has_data": data is not None
        })
        
        try:
            # Step 1: Validate code
            is_valid, validation_msg = self.validate_code(code)
            if not is_valid:
                logging.warning({
                    "event": "code_validation_failed",
                    "execution_id": execution_id,
                    "reason": validation_msg
                })
                return {"error": f"Code validation failed: {validation_msg}"}
            
            # Step 1.5: Deep security analysis
            if not self._analyze_code_safety(code):
                logging.error({
                    "event": "security_analysis_failed",
                    "execution_id": execution_id,
                    "code_snippet": code[:200]
                })
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
                        logging.warning(f"Skipping dangerous global variable: {key}")
                        continue
                    
                    # Type validation - only allow safe types
                    if not self._is_safe_type(value):
                        logging.warning(f"Skipping unsafe type for global variable {key}: {type(value)}")
                        continue
                        
                    globals_dict[key] = value
            
            # Deep copy data for complete isolation
            if data is not None:
                try:
                    globals_dict['data'] = copy.deepcopy(data)
                    logging.info({
                        "event": "data_copied",
                        "execution_id": execution_id,
                        "data_type": type(data).__name__
                    })
                except Exception as e:
                    logging.error({
                        "event": "data_copy_failed",
                        "execution_id": execution_id,
                        "error": str(e)
                    })
                    return {"error": f"Failed to copy data: {str(e)}"}
            
            # Step 3: Compile code with RestrictedPython and strict policy
            try:
                # Use compile_restricted_exec with enhanced security policy
                compiled_result = compile_restricted_exec(code, '<sandbox>')
                
                # Check for compilation errors
                if compiled_result.errors:
                    error_msg = "; ".join(compiled_result.errors)
                    logging.error({
                        "event": "compilation_errors",
                        "execution_id": execution_id,
                        "errors": compiled_result.errors
                    })
                    return {"error": f"Compilation errors: {error_msg}"}
                
                if compiled_result.warnings:
                    logging.warning({
                        "event": "compilation_warnings",
                        "execution_id": execution_id,
                        "warnings": compiled_result.warnings
                    })
                
                byte_code = compiled_result.code
                if byte_code is None:
                    return {"error": "Code compilation failed - potentially unsafe code detected"}
                    
            except SyntaxError as e:
                logging.error({
                    "event": "compilation_failed",
                    "execution_id": execution_id,
                    "error": str(e)
                })
                return {"error": f"Syntax error: {str(e)}"}
            except Exception as e:
                logging.error({
                    "event": "compilation_error",
                    "execution_id": execution_id,
                    "error": str(e)
                })
                return {"error": f"Compilation error: {str(e)}"}
            
            # Step 4: Execute with enhanced resource limits and monitoring
            local_vars = {}
            
            try:
                # Apply stricter resource limits for enhanced security
                with ResourceManager.limit_resources(
                    max_memory_mb=min(self.max_memory_mb, 256),  # Cap at 256MB max
                    max_cpu_seconds=min(self.max_cpu_seconds, 10)  # Cap at 10 seconds max
                ):
                    # Execute compiled bytecode with enhanced monitoring
                    exec(byte_code, globals_dict, local_vars)
                
                execution_time = time.time() - start_time
                
                logging.info({
                    "event": "sandbox_execution_success",
                    "execution_id": execution_id,
                    "execution_time": execution_time,
                    "local_vars": list(local_vars.keys())
                })
                
                # Clean up potentially dangerous variables from result
                cleaned_vars = {}
                for key, value in local_vars.items():
                    if not key.startswith('_') and key not in SecurityGuards.DANGEROUS_ATTRIBUTES:
                        cleaned_vars[key] = value
                
                return {
                    "result": cleaned_vars,
                    "execution_time": execution_time,
                    "execution_id": execution_id
                }
                
            except TimeoutError as e:
                logging.error({
                    "event": "execution_timeout",
                    "execution_id": execution_id,
                    "max_seconds": self.max_cpu_seconds
                })
                return {"error": f"Execution timeout: {str(e)}"}
                
            except MemoryError as e:
                logging.error({
                    "event": "memory_exceeded",
                    "execution_id": execution_id,
                    "max_memory_mb": self.max_memory_mb
                })
                return {"error": f"Memory limit exceeded: {str(e)}"}
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_trace = traceback.format_exc()
                
                logging.error({
                    "event": "execution_error",
                    "execution_id": execution_id,
                    "execution_time": execution_time,
                    "error": str(e),
                    "traceback": error_trace
                })
                
                # Return user-friendly error without exposing system details
                return {"error": f"Execution error: {str(e)}"}
                
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error({
                "event": "sandbox_critical_error",
                "execution_id": execution_id,
                "execution_time": execution_time,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            return {"error": f"Sandbox error: {str(e)}"}

# Maintain backward compatibility
class Sandbox(EnhancedSandbox):
    """Backward compatibility wrapper for the enhanced sandbox"""
    
    def __init__(self):
        super().__init__(max_memory_mb=256, max_cpu_seconds=30)
        logging.info("Using enhanced sandbox with backward compatibility")

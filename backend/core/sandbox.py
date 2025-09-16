import logging
import copy
import time
import traceback
from typing import Any, Dict, Optional

# Enhanced secure code execution sandbox using RestrictedPython with security guards
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins
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
        """Initialize safely imported modules for sandbox use"""
        safe_modules = {}
        
        try:
            import pandas as pd
            import polars as pl
            import numpy as np
            import matplotlib.pyplot as plt
            import plotly.express as px
            import plotly.graph_objects as go
            import json
            import math
            import datetime
            import re
            
            # Only expose safe, read-only versions
            safe_modules.update({
                'pd': pd,
                'pl': pl,
                'np': np,
                'plt': plt,
                'px': px,
                'go': go,
                'json': json,
                'math': math,
                'datetime': datetime,
                're': re
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
            
            # Step 2: Prepare execution environment
            globals_dict = {'__builtins__': self.safe_builtins}
            
            # Add safe modules
            globals_dict.update(self.safe_modules)
            
            # Add extra globals if provided (with validation)
            if extra_globals:
                for key, value in extra_globals.items():
                    if key.startswith('_') or key in SecurityGuards.DANGEROUS_ATTRIBUTES:
                        logging.warning(f"Skipping dangerous global variable: {key}")
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
            
            # Step 3: Compile code with RestrictedPython
            try:
                byte_code = compile_restricted(code, '<sandbox>', 'exec')
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
            
            # Step 4: Execute with resource limits
            local_vars = {}
            
            try:
                with ResourceManager.limit_resources(
                    max_memory_mb=self.max_memory_mb,
                    max_cpu_seconds=self.max_cpu_seconds
                ):
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

# Enhanced Security Guards for RestrictedPython
# Custom guards to prevent security vulnerabilities and resource abuse

import ast
import sys  
import signal
from contextlib import contextmanager

# Handle Windows compatibility for resource module
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    # Windows doesn't have resource module
    HAS_RESOURCE = False
    resource = None

class SecurityGuards:
    """Enhanced security guards for RestrictedPython execution"""
    
    # Dangerous modules and functions that should never be accessible
    DANGEROUS_MODULES = {
        'os', 'sys', 'subprocess', 'importlib', 'imp', 'marshal', 
        'pickle', 'shelve', 'dill', 'joblib', 'cloudpickle',
        'ctypes', 'multiprocessing', 'threading', 'asyncio',
        'socket', 'urllib', 'requests', 'http', 'ftplib',
        'smtplib', 'telnetlib', 'xmlrpc', 'sqlite3', 'dbm'
    }
    
    DANGEROUS_BUILTINS = {
        'eval', 'exec', 'compile', '__import__', 'open', 'file',
        'input', 'raw_input', 'reload', 'vars', 'globals', 'locals',
        'dir', 'getattr', 'setattr', 'delattr', 'hasattr'
    }
    
    DANGEROUS_ATTRIBUTES = {
        '__class__', '__bases__', '__subclasses__', '__mro__',
        '__globals__', '__code__', '__func__', '__self__',
        'func_globals', 'func_code', 'im_class', 'im_func', 'im_self'
    }
    
    @staticmethod
    def safer_getattr(obj, name, default=None, getattr=getattr):
        """Guarded getattr to prevent access to dangerous attributes"""
        if name in SecurityGuards.DANGEROUS_ATTRIBUTES:
            raise AttributeError(f"Access to '{name}' is not allowed")
        
        if name.startswith('_'):
            raise AttributeError(f"Access to private attribute '{name}' is not allowed")
        
        return getattr(obj, name, default)
    
    @staticmethod
    def safer_setattr(obj, name, value):
        """Guarded setattr to prevent setting dangerous attributes"""
        if name in SecurityGuards.DANGEROUS_ATTRIBUTES or name.startswith('_'):
            raise AttributeError(f"Setting attribute '{name}' is not allowed")
        
        return setattr(obj, name, value)
    
    @staticmethod
    def safer_delattr(obj, name):
        """Guarded delattr to prevent deletion of system attributes"""
        if name in SecurityGuards.DANGEROUS_ATTRIBUTES or name.startswith('_'):
            raise AttributeError(f"Deleting attribute '{name}' is not allowed")
        
        return delattr(obj, name)
    
    @staticmethod
    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        """Controlled import function that only allows safe modules"""
        
        # Check if the base module is in dangerous list
        base_module = name.split('.')[0]
        if base_module in SecurityGuards.DANGEROUS_MODULES:
            raise ImportError(f"Import of '{name}' is not allowed")
        
        # Only allow specific safe modules
        allowed_modules = {
            'pandas', 'numpy', 'matplotlib', 'plotly', 'seaborn',
            'scipy', 'sklearn', 'statsmodels', 'math', 'datetime',
            'json', 'csv', 'collections', 'itertools', 'functools',
            're', 'string', 'textwrap', 'unicodedata'
        }
        
        if base_module not in allowed_modules:
            raise ImportError(f"Import of '{name}' is not allowed. Allowed modules: {allowed_modules}")
        
        return __import__(name, globals, locals, fromlist, level)
    
    @staticmethod
    def guarded_write(obj):
        """Prevent all write operations"""
        raise AttributeError("Write operations are not allowed")
    
    @staticmethod
    def create_safe_builtins():
        """Create a safe builtins dictionary"""
        from RestrictedPython.Guards import safe_builtins
        import operator
        
        safe_dict = dict(safe_builtins)
        
        # Add safe operations
        safe_dict.update({
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sorted': sorted,
            'reversed': reversed,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'list': list,
            'tuple': tuple,
            'dict': dict,
            'set': set,
            '_getitem_': operator.getitem,
            '_getiter_': iter,
            '_getattr_': SecurityGuards.safer_getattr,
            '_write_': SecurityGuards.guarded_write,
            '__import__': SecurityGuards.guarded_import,
        })
        
        # Remove dangerous builtins
        for dangerous in SecurityGuards.DANGEROUS_BUILTINS:
            safe_dict.pop(dangerous, None)
        
        return safe_dict

class ResourceManager:
    """Manage computational resources during code execution"""
    
    @staticmethod
    @contextmanager
    def limit_resources(max_memory_mb=256, max_cpu_seconds=30):
        """Context manager to limit memory and CPU usage - Windows compatible"""
        import threading
        import time
        
        # For Windows, we'll use a simple timeout approach
        def timeout_handler():
            time.sleep(max_cpu_seconds)
            raise TimeoutError(f"Code execution exceeded {max_cpu_seconds} seconds")
        
        # Set resource limits (Unix-like systems only)
        old_limits = {}
        timeout_thread = None
        
        try:
            # Unix-like resource limits
            if HAS_RESOURCE and resource:
                if hasattr(resource, 'RLIMIT_AS'):
                    # Memory limit
                    old_limits['memory'] = resource.getrlimit(resource.RLIMIT_AS)
                    resource.setrlimit(resource.RLIMIT_AS, (max_memory_mb * 1024 * 1024, -1))
                
                if hasattr(resource, 'RLIMIT_CPU'):
                    # CPU time limit
                    old_limits['cpu'] = resource.getrlimit(resource.RLIMIT_CPU)
                    resource.setrlimit(resource.RLIMIT_CPU, (max_cpu_seconds, max_cpu_seconds + 5))
                
                # Set timeout signal (Unix only)
                if hasattr(signal, 'SIGALRM'):
                    def signal_timeout_handler(signum, frame):
                        raise TimeoutError(f"Code execution exceeded {max_cpu_seconds} seconds")
                    old_handler = signal.signal(signal.SIGALRM, signal_timeout_handler)
                    signal.alarm(max_cpu_seconds)
            else:
                # Windows fallback - use threading timer
                timeout_thread = threading.Timer(max_cpu_seconds, timeout_handler)
                timeout_thread.daemon = True
                timeout_thread.start()
            
            yield
            
        except Exception as e:
            raise e
        finally:
            # Cleanup
            try:
                if timeout_thread and timeout_thread.is_alive():
                    timeout_thread.cancel()
                
                if HAS_RESOURCE and resource:
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)
                        if 'old_handler' in locals():
                            signal.signal(signal.SIGALRM, old_handler)
                    
                    for limit_type, old_limit in old_limits.items():
                        if limit_type == 'memory' and hasattr(resource, 'RLIMIT_AS'):
                            resource.setrlimit(resource.RLIMIT_AS, old_limit)
                        elif limit_type == 'cpu' and hasattr(resource, 'RLIMIT_CPU'):
                            resource.setrlimit(resource.RLIMIT_CPU, old_limit)
            except Exception:
                pass  # Ignore errors when restoring limits

class CodeValidator:
    """Validate code for security issues before execution"""
    
    @staticmethod
    def validate_ast(code: str) -> tuple[bool, str]:
        """Validate code using AST analysis"""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in SecurityGuards.DANGEROUS_BUILTINS:
                        return False, f"Dangerous function call: {node.func.id}"
            
            # Check for dangerous attribute access
            if isinstance(node, ast.Attribute):
                if node.attr in SecurityGuards.DANGEROUS_ATTRIBUTES:
                    return False, f"Dangerous attribute access: {node.attr}"
            
            # Check for import statements
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base_module = alias.name.split('.')[0]
                    if base_module in SecurityGuards.DANGEROUS_MODULES:
                        return False, f"Dangerous import: {alias.name}"
            
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    base_module = node.module.split('.')[0]
                    if base_module in SecurityGuards.DANGEROUS_MODULES:
                        return False, f"Dangerous import from: {node.module}"
        
        return True, "Code validation passed"
    
    @staticmethod
    def validate_code_patterns(code: str) -> tuple[bool, str]:
        """Check for dangerous string patterns in code"""
        dangerous_patterns = [
            'exec(', 'eval(', '__import__', 'open(', 'file(',
            'os.system', 'subprocess', 'socket.', 'urllib.',
            'pickle.', 'marshal.', 'ctypes.', 'globals()',
            'locals()', 'vars()', 'dir()', '__class__',
            '__bases__', '__subclasses__', '__mro__'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                return False, f"Dangerous pattern detected: {pattern}"
        
        return True, "Pattern validation passed"
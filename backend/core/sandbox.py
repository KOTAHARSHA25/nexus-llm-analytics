import logging

# Secure code execution sandbox using RestrictedPython
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins
import operator
import copy

class Sandbox:
    """Secure code execution environment using RestrictedPython. Only allows safe imports and runs on a copy of the data."""
    def __init__(self):
        # Allow only a minimal set of builtins
        self.builtins = dict(safe_builtins)
        self.builtins.update({
            'print': print,
            'len': len,
            'range': range,
            '_getitem_': operator.getitem,
            '_getiter_': iter,
        })

    def execute(self, code, data=None, extra_globals=None):
        """
        Execute code in a restricted environment. Data is always copied before use.
        :param code: The code string to execute
        :param data: The data object (e.g., DataFrame) to operate on (will be copied)
        :param extra_globals: Any extra globals to expose (optional)
        :return: dict with 'result' or 'error'
        """
        import_types = {'pandas', 'numpy', 'matplotlib'}
        # Prepare globals
        globals_dict = {'__builtins__': self.builtins}
        if extra_globals:
            globals_dict.update(extra_globals)
        # Only allow safe imports
        allowed_imports = {}
        try:
            import pandas as pd
            import numpy as np
            import matplotlib
            allowed_imports['pd'] = pd
            allowed_imports['np'] = np
            allowed_imports['matplotlib'] = matplotlib
        except ImportError:
            pass
        globals_dict.update(allowed_imports)
        # Always work on a copy of the data
        if data is not None:
            globals_dict['data'] = copy.deepcopy(data)
        # Compile and execute code
        try:
            byte_code = compile_restricted(code, '<string>', 'exec')
            local_vars = {}
            exec(byte_code, globals_dict, local_vars)
            logging.info({"event": "sandbox_exec_success", "code": code, "local_vars": list(local_vars.keys())})
            # Return only the local_vars, not the globals
            return {'result': local_vars}
        except Exception as e:
            logging.error({"event": "sandbox_exec_error", "code": code, "error": str(e)})
            return {'error': str(e)}

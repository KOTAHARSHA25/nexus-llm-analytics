# Secure code execution sandbox using RestrictedPython

class Sandbox:
    """Simple, temporary code execution environment (not secure)."""
    def __init__(self):
        self.safe_builtins = {"__builtins__": {"print": print, "len": len, "range": range}}

    def execute(self, code, local_vars=None):
        # WARNING: This is not secure! For demo/testing only.
        if local_vars is None:
            local_vars = {}
        try:
            exec(code, self.safe_builtins, local_vars)
            return {"result": local_vars}
        except Exception as e:
            return {"error": str(e)}

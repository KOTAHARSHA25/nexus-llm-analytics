import logging
# CrewAI Review & Correction Agent: Reviews and corrects generated code


import ast

class ReviewAgent:
    """Agent for reviewing and correcting generated code/results."""
    def __init__(self):
        pass

    def review(self, code):
        """
        Review the provided code for syntax errors and basic security risks.
        Returns a dict with 'status', 'explanation', and 'corrected_code' if needed.
        """
        # Check for syntax errors
        try:
            ast.parse(code)
        except SyntaxError as e:
            logging.error({"event": "review_syntax_error", "code": code, "error": str(e)})
            return {
                "status": "error",
                "explanation": f"Syntax error: {e}",
                "corrected_code": None
            }
        # Basic security check: block dangerous builtins
        dangerous = ["exec", "eval", "open", "os.system", "subprocess", "import os", "import sys"]
        for bad in dangerous:
            if bad in code:
                logging.warning({"event": "review_security_risk", "code": code, "risk": bad})
                return {
                    "status": "error",
                    "explanation": f"Potentially unsafe code detected: '{bad}' is not allowed.",
                    "corrected_code": None
                }
        # If no issues, return success
        logging.info({"event": "review_passed", "code": code})
        return {
            "status": "ok",
            "explanation": "Code is syntactically correct and passes basic security checks.",
            "corrected_code": code
        }

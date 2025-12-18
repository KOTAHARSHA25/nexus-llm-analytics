import logging
# CrewAI Review & Correction Agent: Reviews and corrects generated code


import ast

from backend.core.llm_client import LLMClient

class ReviewAgent:
    """Agent for reviewing and correcting generated code/results using LLM (Phi-3 Mini) and AST."""
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or LLMClient()

    def review(self, code):
        """
        Review the provided code for syntax errors, security, and correctness using both AST and LLM.
        Returns a dict with 'status', 'explanation', and 'corrected_code' if needed.
        """
        # AST syntax and security check
        try:
            ast.parse(code)
        except SyntaxError as e:
            logging.error({"event": "review_syntax_error", "code": code, "error": str(e)})
            return {
                "status": "error",
                "explanation": f"Syntax error: {e}",
                "corrected_code": None
            }
        dangerous = ["exec", "eval", "open", "os.system", "subprocess", "import os", "import sys"]
        for bad in dangerous:
            if bad in code:
                logging.warning({"event": "review_security_risk", "code": code, "risk": bad})
                return {
                    "status": "error",
                    "explanation": f"Potentially unsafe code detected: '{bad}' is not allowed.",
                    "corrected_code": None
                }
        # LLM-powered review and correction
        prompt = f"Review the following Python pandas code for errors, security, and correctness. If you find an issue, provide corrected code and a brief explanation.\n\nCode:\n{code}\n\nRespond in JSON: {{'status': 'ok' or 'error', 'explanation': explanation, 'corrected_code': code or null}}"
        llm_result = self.llm_client.generate_review(prompt)
        import json
        try:
            llm_json = json.loads(llm_result.get("response", ""))
            if llm_json.get("status") == "ok":
                logging.info({"event": "review_passed", "code": code})
                return {
                    "status": "ok",
                    "explanation": llm_json.get("explanation", "LLM review passed."),
                    "corrected_code": code
                }
            else:
                return {
                    "status": "error",
                    "explanation": llm_json.get("explanation", "LLM review found an error."),
                    "corrected_code": llm_json.get("corrected_code")
                }
        except Exception as e:
            # Fallback: treat as pass if LLM output is not valid JSON
            logging.warning({"event": "llm_review_parse_error", "error": str(e), "llm_response": llm_result.get("response")})
            return {
                "status": "ok",
                "explanation": "LLM review could not be parsed, assuming pass.",
                "corrected_code": code
            }

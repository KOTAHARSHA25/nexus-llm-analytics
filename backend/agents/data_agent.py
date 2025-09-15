import logging
# CrewAI Data Agent: Generates and executes code for structured data analysis

import pandas as pd
import os
from backend.agents.review_agent import ReviewAgent
from backend.core.sandbox import Sandbox
from backend.core.utils import log_data_version

class DataAgent:
    """Responsible for data manipulation and analysis."""
    def __init__(self, review_agent=None, sandbox=None):
        self.data = None
        self.filename = None
        self.last_code = None
        self.last_explanation = None
        self.review_agent = review_agent or ReviewAgent()
        self.sandbox = sandbox or Sandbox()

    def load_file(self, filename):
        logging.info({"event": "load_file", "filename": filename})
        log_data_version("load_file", filename, details={"action": "load_file_start"})
        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', filename)
        try:
            if filename.endswith('.csv'):
                self.data = pd.read_csv(filepath)
                self.last_code = f"pd.read_csv('{filename}')"
                self.last_explanation = f"Load CSV file '{filename}' into a pandas DataFrame."
            elif filename.endswith('.json'):
                self.data = pd.read_json(filepath)
                self.last_code = f"pd.read_json('{filename}')"
                self.last_explanation = f"Load JSON file '{filename}' into a pandas DataFrame."
            else:
                return {"error": "Unsupported file type"}
            self.filename = filename
            logging.info({"event": "file_loaded", "filename": filename, "columns": list(self.data.columns)})
            log_data_version("file_loaded", filename, details={"columns": list(self.data.columns)})
            return {"success": True}
        except Exception as e:
            logging.error({"event": "file_load_error", "filename": filename, "error": str(e)})
            log_data_version("file_load_error", filename, details={"error": str(e)})
            return {"error": str(e)}

    def summarize(self):
        if self.data is None:
            return {"error": "No data loaded."}
        code = "result = data.head(5)"
        explanation = "Show the first 5 rows of the DataFrame."
        review = self.review_agent.review(code)
        logging.info({"event": "code_review", "query": "summarize", "code": code, "review_status": review["status"]})
        if review["status"] != "ok":
            logging.warning({"event": "code_review_failed", "query": "summarize", "explanation": review["explanation"]})
            return {"error": review["explanation"]}
        exec_result = self.sandbox.execute(review["corrected_code"], data=self.data)
        logging.info({"event": "code_execution", "query": "summarize", "result_keys": list(exec_result.keys())})
        if "error" in exec_result:
            logging.error({"event": "code_execution_error", "query": "summarize", "error": exec_result["error"]})
            return {"error": exec_result["error"]}
        result = exec_result["result"].get("result")
        log_data_version("summarize", self.filename, details={"preview": result.to_dict(orient="records") if hasattr(result, 'to_dict') else str(result)})
        return {
            "filename": self.filename,
            "columns": list(self.data.columns),
            "row_count": int(len(self.data)),
            "preview": result.to_dict(orient="records") if hasattr(result, 'to_dict') else str(result),
            "code": code,
            "explanation": explanation
        }

    def describe(self):
        if self.data is None:
            return {"error": "No data loaded."}
        code = "result = data.describe(include='all').fillna('')"
        explanation = "Show summary statistics for all columns."
        review = self.review_agent.review(code)
        logging.info({"event": "code_review", "query": "describe", "code": code, "review_status": review["status"]})
        if review["status"] != "ok":
            logging.warning({"event": "code_review_failed", "query": "describe", "explanation": review["explanation"]})
            return {"error": review["explanation"]}
        exec_result = self.sandbox.execute(review["corrected_code"], data=self.data)
        logging.info({"event": "code_execution", "query": "describe", "result_keys": list(exec_result.keys())})
        if "error" in exec_result:
            logging.error({"event": "code_execution_error", "query": "describe", "error": exec_result["error"]})
            return {"error": exec_result["error"]}
        result = exec_result["result"].get("result")
        log_data_version("describe", self.filename, details={"describe": result.to_dict() if hasattr(result, 'to_dict') else str(result)})
        return {"describe": result.to_dict() if hasattr(result, 'to_dict') else str(result), "code": code, "explanation": explanation}

    def value_counts(self, column):
        if self.data is None:
            return {"error": "No data loaded."}
        if column not in self.data.columns:
            return {"error": f"Column '{column}' not found."}
        code = f"result = data['{column}'].value_counts()"
        explanation = f"Show value counts for column '{column}'."
        review = self.review_agent.review(code)
        logging.info({"event": "code_review", "query": "value_counts", "code": code, "review_status": review["status"]})
        if review["status"] != "ok":
            logging.warning({"event": "code_review_failed", "query": "value_counts", "explanation": review["explanation"]})
            return {"error": review["explanation"]}
        exec_result = self.sandbox.execute(review["corrected_code"], data=self.data)
        logging.info({"event": "code_execution", "query": "value_counts", "result_keys": list(exec_result.keys())})
        if "error" in exec_result:
            logging.error({"event": "code_execution_error", "query": "value_counts", "error": exec_result["error"]})
            return {"error": exec_result["error"]}
        result = exec_result["result"].get("result")
        log_data_version("value_counts", self.filename, details={"column": column, "value_counts": result.to_dict() if hasattr(result, 'to_dict') else str(result)})
        return {"value_counts": result.to_dict() if hasattr(result, 'to_dict') else str(result), "code": code, "explanation": explanation}

    def filter(self, column, value):
        if self.data is None:
            return {"error": "No data loaded."}
        if column not in self.data.columns:
            return {"error": f"Column '{column}' not found."}
        code = f"result = data[data['{column}'] == filter_value]"
        explanation = f"Filter rows where column '{column}' equals '{value}'."
        review = self.review_agent.review(code)
        logging.info({"event": "code_review", "query": "filter", "code": code, "review_status": review["status"]})
        if review["status"] != "ok":
            logging.warning({"event": "code_review_failed", "query": "filter", "explanation": review["explanation"]})
            return {"error": review["explanation"]}
        exec_result = self.sandbox.execute(review["corrected_code"], data=self.data, extra_globals={'filter_value': value})
        logging.info({"event": "code_execution", "query": "filter", "result_keys": list(exec_result.keys())})
        if "error" in exec_result:
            logging.error({"event": "code_execution_error", "query": "filter", "error": exec_result["error"]})
            return {"error": exec_result["error"]}
        result = exec_result["result"].get("result")
        log_data_version("filter", self.filename, details={"column": column, "value": value, "filtered_count": int(len(result))})
        return {
            "filtered_count": int(len(result)),
            "preview": result.head(5).to_dict(orient="records") if hasattr(result, 'head') else str(result),
            "code": code,
            "explanation": explanation
        }

import logging
# CrewAI Data Agent: Generates and executes code for structured data analysis

import pandas as pd
import polars as pl
import os
from backend.agents.review_agent import ReviewAgent
from backend.core.sandbox import Sandbox
from backend.core.utils import log_data_version
from backend.core.llm_client import LLMClient

class DataAgent:
    """Responsible for data manipulation and analysis."""
    def __init__(self, review_agent=None, sandbox=None, llm_client=None):
        self.data = None
        self.filename = None
        self.last_code = None
        self.last_explanation = None
        self.review_agent = review_agent or ReviewAgent()
        self.sandbox = sandbox or Sandbox()
        self.llm_client = llm_client or LLMClient()

    def load_file(self, filename):
        logging.info({"event": "load_file", "filename": filename})
        log_data_version("load_file", filename, details={"action": "load_file_start"})
        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', filename)
        try:
            if filename.endswith('.csv'):
                # OPTIMIZATION: Smart loading based on file size for performance
                file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                
                if file_size_mb > 10:  # Large files (>10MB)
                    logging.info(f"Loading large file ({file_size_mb:.1f}MB) with optimizations")
                    # Use polars for initial loading (faster for large files)
                    polars_df = pl.read_csv(filepath)
                    # Convert to pandas only if needed for compatibility
                    self.data = polars_df.to_pandas()
                    # Keep polars reference for performance operations
                    self._polars_data = polars_df
                else:
                    # Standard pandas loading for smaller files
                    self.data = pd.read_csv(filepath)
                    # Create polars version for potential performance operations
                    self._polars_data = pl.from_pandas(self.data)
                try:
                    self.data_pl = pl.read_csv(filepath)
                except Exception:
                    self.data_pl = None
                self.last_code = f"pd.read_csv('{filename}')  # Also available as polars: pl.read_csv('{filename}')"
                self.last_explanation = f"Load CSV file '{filename}' into pandas DataFrame (with polars backup for performance)."
            elif filename.endswith('.json'):
                self.data = pd.read_json(filepath)
                # Store polars version for high-performance operations
                try:
                    self.data_pl = pl.read_json(filepath)
                except Exception:
                    self.data_pl = None
                self.last_code = f"pd.read_json('{filename}')  # Also available as polars: pl.read_json('{filename}')"
                self.last_explanation = f"Load JSON file '{filename}' into pandas DataFrame (with polars backup for performance)."
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
        # Use LLM to generate code and explanation from user query
        user_query = "Show the first 5 rows of the DataFrame."
        max_attempts = 3
        attempt = 0
        last_error = None
        code = None
        explanation = None
        while attempt < max_attempts:
            prompt = f"Given the following user query, write Python pandas code to answer it. Also provide a short explanation.\n\nQuery: {user_query}\n\nRespond in JSON: {{'code': code, 'explanation': explanation}}"
            llm_result = self.llm_client.generate_primary(prompt)
            import json
            try:
                llm_json = json.loads(llm_result.get("response", ""))
                code = llm_json.get("code")
                explanation = llm_json.get("explanation")
            except Exception as e:
                code = "result = data.head(5)"
                explanation = "Show the first 5 rows of the DataFrame."
            review = self.review_agent.review(code)
            logging.info({"event": "code_review", "query": "summarize", "code": code, "review_status": review["status"]})
            if review["status"] == "ok":
                break
            # If error, try again with LLM correction
            last_error = review["explanation"]
            if review.get("corrected_code"):
                code = review["corrected_code"]
            else:
                # Ask LLM to fix the code
                fix_prompt = f"The following code has an error: {last_error}\n\nCode:\n{code}\n\nPlease correct it and explain. Respond in JSON: {{'code': code, 'explanation': explanation}}"
                fix_result = self.llm_client.generate_primary(fix_prompt)
                try:
                    fix_json = json.loads(fix_result.get("response", ""))
                    code = fix_json.get("code")
                    explanation = fix_json.get("explanation")
                except Exception:
                    logging.debug("Operation failed (non-critical) - continuing")
            attempt += 1
        if review["status"] != "ok":
            return {"error": f"Code generation failed after {max_attempts} attempts: {last_error}"}
        exec_result = self.sandbox.execute(code, data=self.data)
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
        prompt = "Write Python pandas code to show summary statistics for all columns in a DataFrame called 'data'. Explain what the code does."
        llm_result = self.llm_client.generate_primary(prompt)
        code = "result = data.describe(include='all').fillna('')"
        explanation = llm_result.get("response") or "Show summary statistics for all columns."
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
    
    def _optimize_large_dataset_operation(self, operation_type: str, **kwargs):
        """OPTIMIZATION: Use vectorized operations for large datasets"""
        if self.data is None or not hasattr(self, '_polars_data'):
            return None
            
        # Check if dataset is large enough to benefit from optimization
        if len(self.data) < 1000:
            return None  # Use standard pandas operations for small data
            
        logging.info(f"🚀 Using optimized processing for large dataset ({len(self.data)} rows)")
        
        try:
            if operation_type == "describe":
                # Use polars for faster statistical operations
                polars_stats = self._polars_data.describe()
                return polars_stats.to_pandas()
                
            elif operation_type == "value_counts" and "column" in kwargs:
                column = kwargs["column"]
                if column in self._polars_data.columns:
                    # Polars value_counts is much faster for large datasets
                    result = self._polars_data.select(pl.col(column).value_counts()).to_pandas()
                    return result[column].iloc[0] if len(result) > 0 else None
                    
            elif operation_type == "group_by" and "column" in kwargs:
                column = kwargs["column"]
                if column in self._polars_data.columns:
                    # Use polars for efficient grouping
                    result = self._polars_data.group_by(column).agg([
                        pl.count().alias("count"),
                        pl.col("*").exclude(column).mean().name.suffix("_mean")
                    ])
                    return result.to_pandas()
                    
        except Exception as e:
            logging.warning(f"⚠️ Optimized operation failed, falling back to pandas: {e}")
            return None
        
        return None

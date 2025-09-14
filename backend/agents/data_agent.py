# CrewAI Data Agent: Generates and executes code for structured data analysis

import pandas as pd
import os

class DataAgent:
    """Responsible for data manipulation and analysis."""
    def __init__(self):
        self.data = None
        self.filename = None
        self.last_code = None
        self.last_explanation = None

    def load_file(self, filename):
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
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}

    def summarize(self):
        if self.data is None:
            return {"error": "No data loaded."}
        code = f"df.head(5)"
        explanation = "Show the first 5 rows of the DataFrame."
        return {
            "filename": self.filename,
            "columns": list(self.data.columns),
            "row_count": int(len(self.data)),
            "preview": self.data.head(5).to_dict(orient="records"),
            "code": code,
            "explanation": explanation
        }

    def describe(self):
        if self.data is None:
            return {"error": "No data loaded."}
        code = f"df.describe(include='all')"
        explanation = "Show summary statistics for all columns."
        try:
            desc = self.data.describe(include='all').fillna('').to_dict()
            return {"describe": desc, "code": code, "explanation": explanation}
        except Exception as e:
            return {"error": str(e)}

    def value_counts(self, column):
        if self.data is None:
            return {"error": "No data loaded."}
        if column not in self.data.columns:
            return {"error": f"Column '{column}' not found."}
        code = f"df['{column}'].value_counts()"
        explanation = f"Show value counts for column '{column}'."
        try:
            counts = self.data[column].value_counts().to_dict()
            return {"value_counts": counts, "code": code, "explanation": explanation}
        except Exception as e:
            return {"error": str(e)}

    def filter(self, column, value):
        if self.data is None:
            return {"error": "No data loaded."}
        if column not in self.data.columns:
            return {"error": f"Column '{column}' not found."}
        code = f"df[df['{column}'] == '{value}']"
        explanation = f"Filter rows where column '{column}' equals '{value}'."
        try:
            filtered = self.data[self.data[column] == value]
            return {
                "filtered_count": int(len(filtered)),
                "preview": filtered.head(5).to_dict(orient="records"),
                "code": code,
                "explanation": explanation
            }
        except Exception as e:
            return {"error": str(e)}

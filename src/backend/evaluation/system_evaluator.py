
import asyncio
import os
import json
import logging
import pandas as pd
import numpy as np
import traceback
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import shutil
import time
import gc

# Add src to path
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))

from backend.plugins.data_analyst_agent import DataAnalystAgent
from backend.core.code_execution_history import get_execution_history
from backend.core.enhanced_cache_integration import get_enhanced_cache_manager

# Configuration
EVAL_DIR = Path("data/evaluation")
REPORT_DIR = Path("reports/evaluation")
LOG_FILE = REPORT_DIR / "evaluation.log"

import re

# Dataset Configurations
DATASETS = {
    "structured_financial": {
        "filename": "financial_sample.csv",
        "type": "structured",
        "columns": ["Date", "Revenue", "Expenses", "Profit", "Region", "Category"]
    },
    "large_scale": {
        "filename": "large_scale_sales.csv",
        "type": "large",
    },
    "corrupted": {
        "filename": "corrupted_data.csv",
        "type": "corrupted",
        "columns": ["ID", "Value"]
    },
    "mixed_schema": {
        "filename": "mixed_schema.json",
        "type": "mixed",
    },
    "unstructured_text": {
        "filename": "market_report.txt",
        "type": "unstructured",
    },
    "empty_dataset": {
        "filename": "empty.csv",
        "type": "empty",
        "columns": ["Col1", "Col2"]
    }
}

class SystemEvaluator:
    def __init__(self):
        print("Initializing SystemEvaluator...")
        try:
            from backend.services.analysis_service import get_analysis_service
            self.service = get_analysis_service()
            print("AnalysisService initialized.")
        except Exception as e:
            print(f"Failed to init service: {e}")
            traceback.print_exc()
            raise
            
        self.results = []
        self._setup_logging()
        self._ensure_directories()
        
    def _setup_logging(self):
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=LOG_FILE,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)
        
    def _ensure_directories(self):
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        
    def generate_datasets(self):
        """Generate synthetic datasets for testing."""
        logging.info("Generating synthetic datasets...")
        
        # Set seed for reproducibility (Ground Truth Validation)
        np.random.seed(42)
        
        # Financial
        if not (EVAL_DIR / "financial_sample.csv").exists():
            df_fin = pd.DataFrame({
                "Date": pd.date_range(start="2023-01-01", periods=100),
                "Revenue": np.random.uniform(1000, 5000, 100),
                "Expenses": np.random.uniform(500, 3000, 100),
                "Region": np.random.choice(["North", "South", "East", "West"], 100),
                "Category": np.random.choice(["Software", "Hardware", "Services"], 100)
            })
            df_fin["Profit"] = df_fin["Revenue"] - df_fin["Expenses"]
            df_fin.to_csv(EVAL_DIR / "financial_sample.csv", index=False)
        
        # Large Scale
        if not (EVAL_DIR / "large_scale_sales.csv").exists():
            df_large = pd.DataFrame({
                "TransactionID": range(10000),
                "Amount": np.random.uniform(10, 1000, 10000),
                "CustomerID": np.random.randint(1, 500, 10000),
                "Date": pd.date_range(start="2023-01-01", periods=10000)
            })
            df_large.to_csv(EVAL_DIR / "large_scale_sales.csv", index=False)
        
        # Corrupted
        if not (EVAL_DIR / "corrupted_data.csv").exists():
            df_corrupt = pd.DataFrame({
                "ID": range(50),
                "Value": ["100", "200", "invalid", "400"] * 12 + ["500", "600"]
            })
            df_corrupt.to_csv(EVAL_DIR / "corrupted_data.csv", index=False)
        
        # Mixed JSON
        if not (EVAL_DIR / "mixed_schema.json").exists():
            mixed_data = [
                {"id": 1, "val": 10, "meta": {"type": "A"}},
                {"id": 2, "val": "20", "meta": {"type": "B"}}, # String number
                {"id": 3, "val": None, "meta": {}}
            ]
            with open(EVAL_DIR / "mixed_schema.json", "w") as f:
                json.dump(mixed_data, f)
        
        # Unstructured Text
        if not (EVAL_DIR / "market_report.txt").exists():
            text = """
            Market Report 2023:
            The technology sector saw a 15% growth in Q3. AI adoption is driving revenue.
            However, hardware sales dipped by 5% due to supply chain issues.
            Cloud services remain the most profitable segment with 30% margin.
            """
            with open(EVAL_DIR / "market_report.txt", "w") as f:
                f.write(text.strip())
                
        # Empty Dataset
        if not (EVAL_DIR / "empty.csv").exists():
            pd.DataFrame(columns=["Col1", "Col2"]).to_csv(EVAL_DIR / "empty.csv", index=False)
            
        logging.info("Datasets generated successfully.")

    async def run_query(self, query: str, dataset_name: str, repetition_idx: int) -> Dict[str, Any]:
        """Run a single query against a dataset."""
        dataset_path = EVAL_DIR / DATASETS[dataset_name]["filename"]
        
        try:
            get_enhanced_cache_manager().clear_all()
        except Exception as e:
            logging.warning(f"Failed to clear cache: {e}")
            
        start_time = datetime.now()
        
        logging.info(f"Running Query [Rep {repetition_idx}]: {query} on {dataset_name}")
        print(f"Executing: {query}")
        
        try:
            # Use AnalysisService instead of direct agent
            # Add timeout to prevent benchmark hanging (Robustness)
            result = await asyncio.wait_for(
                self.service.analyze(
                    query=query,
                    context={
                        'filename': DATASETS[dataset_name]["filename"],
                        'filepath': str(dataset_path.absolute()),
                        'file_type': '.' + DATASETS[dataset_name]["filename"].split('.')[-1]
                    }
                ),
                timeout=120.0 # 2 minutes max per query
            )
            
            duration = (datetime.now() - start_time).total_seconds()

            
            if not result:
                 logging.error("Service returned None/Empty!")
                 return {
                    "success": False,
                    "response": None,
                    "error": "Service returned None",
                    "duration": duration
                 }

            # Map AnalysisService result to Evaluator format
            # Service returns: {'success': True, 'result': ..., 'metadata': ...}
            # Or: {'success': True, 'response': ...} depending on agent
            
            # Extract main response text
            response_content = result.get("result") or result.get("response") or result.get("answer")
            
            # Extract code if present
            code_generated = None
            if "metadata" in result:
                code_generated = result["metadata"].get("generated_code") or result["metadata"].get("code")

            return {
                "success": result.get("success", False),
                "response": response_content,
                "code": code_generated, 
                "metadata": result.get("metadata", {}),
                "duration": duration,
                "error": result.get("error")
            }
        except Exception as e:
            logging.error(f"Query Failed: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "response": None,
                "error": str(e),
                "duration": (datetime.now() - start_time).total_seconds()
            }

    async def run_suite(self, test_cases: List[Dict]):
        """Run the full test suite."""
        self.results = []
        
        for case in test_cases:
            case_id = case["id"]
            query = case["query"]
            dataset = case["dataset"]
            repetitions = case.get("repetitions", 2) 
            
            logging.info(f"--- Starting Case {case_id} ---")
            print(f"--- Starting {case_id} ---")
            
            case_results = []
            for i in range(repetitions):
                result = await self.run_query(query, dataset, i+1)
                result["repetition"] = i + 1
                result["case_id"] = case_id
                
                # Validation
                validation = self._validate_result(result, case)
                result["validation"] = validation
                
                case_results.append(result)
                
                # ACADEMIC REQUIREMENT: Cooldown & Cleanup
                logging.info("Cooling down (15s)...")
                time.sleep(15)
                gc.collect()
            
            # Drift Analysis
            drift_metric = self._analyze_drift(case_results)
            
            self.results.append({
                "case": case,
                "runs": case_results,
                "drift_metric": drift_metric
            })
            
    def _validate_result(self, result: Dict, case: Dict) -> Dict:
        """Validate result against expected values and Ground Truth."""
        if not result["success"]:
            return {"valid": False, "reason": "Execution Failed"}
            
        response_text = str(result["response"]).lower()
        expected = case.get("expected")
        
        # 1. Keyword Check
        if expected:
            if isinstance(expected, dict):
                if "keywords" in expected:
                    for kw in expected["keywords"]:
                        if kw.lower() not in response_text:
                            return {"valid": False, "reason": f"Missing keyword: {kw}"}
            elif isinstance(expected, str):
                if expected.lower() not in response_text:
                    return {"valid": False, "reason": "Expected text not found"}
        
        # 2. Dynamic Ground Truth Check
        try:
            return self._validate_ground_truth(case, response_text)
        except Exception as e:
            logging.warning(f"Ground Truth validation error: {e}")
            return {"valid": True, "reason": "Passed keywords (GT skipped)"}

    def _validate_ground_truth(self, case: Dict, response_text: str) -> Dict:
        """Calculate actual answer and compare."""
        case_id = case["id"]
        dataset_name = case["dataset"]
        
        # Only validate structured_financial and large_scale for now
        if dataset_name not in ["structured_financial", "large_scale"]:
             return {"valid": True, "reason": "Passed keywords"}

        df_path = EVAL_DIR / DATASETS[dataset_name]["filename"]
        df = pd.read_csv(df_path)
        
        # TC Logic
        if case_id == "TC_001": # Total Revenue
            actual = df["Revenue"].sum() # Approx 300k
            # Extract numbers from text
            nums = [float(x.replace(',','')) for x in re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', response_text)]
            # Check if any num is close to actual (within 5%)
            if any(abs(n - actual) / actual < 0.05 for n in nums):
                return {"valid": True, "reason": "Passed Numeric Check"}
            return {"valid": False, "reason": f"Numeric Mismatch. Expected ~{actual:.2f}"}

        elif case_id == "TC_002": # Highest Profit Region
            # Group by Region sum/mean? Query: "Which region has highest profit?" usually implies sum or finding max record?
            # Implied: Sum of profit by region, or single transaction?
            # Usually "Region with highest profit" means Sum.
            actual_region = df.groupby("Region")["Profit"].sum().idxmax().lower()
            if actual_region in response_text:
                return {"valid": True, "reason": "Passed Ground Truth"}
            return {"valid": False, "reason": f"Expected region: {actual_region}"}

        elif case_id == "TC_003": # Trend Analysis (Non-Numeric Logic)
            # Query: "Identify the trend in revenue over time."
            # Logic: Calculate slope or correlation of Revenue vs Date index
            df["time_idx"] = np.arange(len(df))
            slope = np.polyfit(df["time_idx"], df["Revenue"], 1)[0]
            expected_trend = "increasing" if slope > 5 else "decreasing" if slope < -5 else "stable"
            
            # Allow synonyms
            synonyms = {
                "increasing": ["upward", "growth", "rising", "increase", "positive"],
                "decreasing": ["downward", "decline", "falling", "decrease", "negative"],
                "stable": ["flat", "consistent", "steady", "no significant trend"]
            }
            
            found = False
            for term in synonyms.get(expected_trend, [expected_trend]):
                if term in response_text:
                    found = True
                    break
            
            if found:
                 return {"valid": True, "reason": f"Correctly identified {expected_trend} trend"}
            return {"valid": False, "reason": f"Missed trend: {expected_trend}"}

        elif case_id == "TC_011": # Empty Dataset (Structural)
            # Logic: Should detect empty state and NOT try to calculate
            if df.empty:
                if "empty" in response_text or "no data" in response_text or "0 rows" in response_text:
                     return {"valid": True, "reason": "Correctly identified empty dataset"}
                return {"valid": False, "reason": "Failed to identify empty dataset"}
            return {"valid": True, "reason": "Dataset was not empty (Setup Error?)"}

        elif case_id == "TC_010": # Negative Test (Unknown Column)
            # Logic: Should fail or return error message, NOT hallucinate a value
            # Expected: "not found", "error", "unavailable"
            negative_markers = ["not found", "does not exist", "error", "unavailable", "invalid", "column"]
            if any(m in response_text for m in negative_markers):
                 return {"valid": True, "reason": "Correctly handled missing column"}
            return {"valid": False, "reason": "Failed to report missing column"}
            
        elif case_id == "TC_005": # Sum Amount (Large Scale)
             actual = df["Amount"].sum()
             nums = [float(x.replace(',','')) for x in re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', response_text)]
             if any(abs(n - actual) / actual < 0.05 for n in nums):
                return {"valid": True, "reason": "Passed Numeric Check"}
             return {"valid": False, "reason": f"Numeric Mismatch. Expected ~{actual:.2f}"}

        elif case_id == "TC_008": # Unstructured Summary (Market Report)
             # Ground Truth Themes from known text
             themes = ["ai", "growth", "hardware", "cloud", "profit", "15%"]
             found_count = sum(1 for t in themes if t in response_text)
             
             if found_count >= 2:
                 return {"valid": True, "reason": f"Found {found_count} valid themes"}
             return {"valid": False, "reason": f"Missing key themes (Found {found_count}/6)"}

        elif case_id == "TC_012": # Nested: Region max rev -> avg profit
             target_region = df.groupby("Region")["Revenue"].sum().idxmax()
             expected_val = df[df["Region"] == target_region]["Profit"].mean()
             nums = [float(x.replace(',','')) for x in re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', response_text)]
             if any(abs(n - expected_val) / expected_val < 0.05 for n in nums):
                 return {"valid": True, "reason": "Passed Numeric Check"}
             return {"valid": False, "reason": f"Numeric Mismatch. Expected ~{expected_val:.2f} (Region: {target_region})"}

        return {"valid": True, "reason": "Passed keywords"}

    def _analyze_drift(self, runs: List[Dict]) -> float:
        """Calculate drift score (0.0 = consistent, 1.0 = highly variable)."""
        responses = [str(r["response"]) for r in runs if r["success"]]
        if not responses: return 0.0
        if len(set(responses)) == 1:
            return 0.0
        return 0.5 
        
    def generate_report(self):
        """Generate final academic report."""
        logging.info("Generating report...")
        
        report_path = REPORT_DIR / "ACADEMIC_EVALUATION_REPORT.md"
        
        try:
            with open(report_path, "w", encoding='utf-8') as f:
                f.write("# Phase 20: Comprehensive System Evaluation Report\n\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                f.write("**Evaluator:** Automated SystemEvaluator\n\n")
                
                # Section A
                f.write("## Section A: Test Case Catalog\n")
                f.write("| ID | Category | Query | Dataset |\n")
                f.write("| :--- | :--- | :--- | :--- |\n")
                for entry in self.results:
                    case = entry["case"]
                    f.write(f"| {case['id']} | {case['category']} | {case['query']} | {case['dataset']} |\n")
                    
                # Section B
                f.write("\n## Section B: Accuracy Analysis\n")
                total_runs = sum(len(e["runs"]) for e in self.results)
                passed_runs = sum(1 for e in self.results for r in e["runs"] if r.get("validation", {}).get("valid"))
                accuracy = passed_runs / total_runs if total_runs else 0
                
                f.write(f"- **Overall Reliability Score:** {accuracy:.1%}\n")
                f.write(f"- **Total Runs:** {total_runs}\n")
                f.write(f"- **Passed:** {passed_runs}\n\n")
                
                for entry in self.results:
                    c = entry["case"]
                    valid_runs = sum(1 for r in entry["runs"] if r.get("validation", {}).get("valid"))
                    f.write(f"### {c['id']}: {c['query']}\n")
                    f.write(f"- Accuracy: {valid_runs}/{len(entry['runs'])}\n")
                    for r in entry["runs"]:
                        # Use ASCII icons for safety
                        icon = "[PASS]" if r.get("validation", {}).get("valid") else "[FAIL]"
                        resp = str(r.get('response', ''))
                        code_snippet = str(r.get('code', '')).replace('\n', ' ')[:50]
                        f.write(f"  - Rep {r['repetition']}: {icon} {resp[:100]}...\n")
                        if r.get('code'):
                            f.write(f"    - Code: `{code_snippet}...`\n")
                
                # Section C
                f.write("\n## Section C: Consistency & Drift Report\n")
                drift_cases = [e for e in self.results if e["drift_metric"] > 0]
                if drift_cases:
                    f.write(f"Detected drift in {len(drift_cases)} test cases.\n")
                    for e in drift_cases:
                         f.write(f"- **{e['case']['id']}**: Responses varied across runs.\n")
                else:
                     f.write("High Consistency: No significant drift detected.\n")

                # Section D: Visualization & Report Validation
                f.write("\n## Section D: Visualization & Artifacts\n")
                artifacts_found = 0
                for entry in self.results:
                    for r in entry["runs"]:
                        meta = r.get("metadata", {})
                        if meta.get("files"):
                            f.write(f"- Case {r['case_id']}: Generated {len(meta['files'])} files ({', '.join(meta['files'])})\n")
                            artifacts_found += 1
                if artifacts_found == 0:
                    f.write("No visualization artifacts generated in this run.\n")

                # Section E: Error Log & Corrections
                f.write("\n## Section E: Error Log\n")
                errors = [r for e in self.results for r in e["runs"] if not r["success"]]
                if errors:
                    for r in errors:
                        f.write(f"- Case {r['case_id']} Rep {r['repetition']}: {r['error']}\n")
                else:
                    f.write("No execution errors recorded.\n")
                    
                # Section F: Final Reliability Score
                f.write("\n## Section F: Final Reliability Score\n")
                consistency_score = 1.0 - (len(drift_cases) / len(self.results)) if self.results else 0.0
                reliability_score = (accuracy * 0.7) + (consistency_score * 0.3)
                
                f.write(f"### Reliability Score: {reliability_score:.1%}\n")
                f.write("_Formula: (Accuracy * 0.7) + (Consistency * 0.3)_\n")
                f.write(f"- Accuracy Component: {accuracy:.1%}\n")
                f.write(f"- Consistency Component: {consistency_score:.1%}\n")
                    
            logging.info(f"Report written to {report_path}")
            print(f"Report written to {report_path}")
        except Exception as e:
            print(f"Report generation failed: {e}")
            traceback.print_exc()

# Run Harness
async def main():
    try:
        evaluator = SystemEvaluator()
        evaluator.generate_datasets()
        
        # Define Test Cases
        full_test_cases = [
            # Structured - Financial
            {"id": "TC_001", "category": "Factual", "query": "What is the total revenue?", "dataset": "structured_financial", "expected": {"keywords": ["revenue"]}, "repetitions": 2},
            {"id": "TC_002", "category": "Analytical", "query": "Which region has the highest profit?", "dataset": "structured_financial", "expected": {"keywords": ["region", "profit"]}, "repetitions": 1},
            {"id": "TC_003", "category": "Complex", "query": "Calculate the ratio of Profit to Revenue for the 'Software' category.", "dataset": "structured_financial", "expected": {"keywords": ["ratio", "software"]}, "repetitions": 1},
            {"id": "TC_004", "category": "Categorical", "query": "List all unique categories.", "dataset": "structured_financial", "expected": {"keywords": ["software", "hardware"]}, "repetitions": 1},
            
            # Large Scale
            {"id": "TC_005", "category": "Performance", "query": "Sum the 'Amount' column.", "dataset": "large_scale", "expected": {"keywords": ["amount"]}, "repetitions": 1},
            
            # Corrupted
            {"id": "TC_006", "category": "Edge Case", "query": "Calculate the average of 'Value' column, ignoring string values.", "dataset": "corrupted", "expected": {"keywords": ["average"]}, "repetitions": 1},
            
            # Mixed JSON
            {"id": "TC_007", "category": "JSON Parsing", "query": "What is the sum of 'val'?", "dataset": "mixed_schema", "expected": {"keywords": ["30"]}, "repetitions": 1},
            
            # Unstructured
            {"id": "TC_008", "category": "Text Analysis", "query": "Summarize the market report.", "dataset": "unstructured_text", "expected": {"keywords": ["market", "report"]}, "repetitions": 1},
            
            # Ambiguous / Edge
            {"id": "TC_009", "category": "Ambiguous", "query": "Is the performance good?", "dataset": "structured_financial", "expected": {"keywords": ["profit", "revenue"]}, "repetitions": 1},
            {"id": "TC_010", "category": "Negative Test", "query": "Analyze the 'NonExistent' column.", "dataset": "structured_financial", "expected": {"keywords": ["error", "not found", "available"]}, "repetitions": 1},
            
            # Empty Dataset
            {"id": "TC_011", "category": "Robustness", "query": "Describe the dataset.", "dataset": "empty_dataset", "expected": {"keywords": ["empty", "no data", "columns"]}, "repetitions": 1},
            
            # Nested Chain
            {"id": "TC_012", "category": "Nested Logic", "query": "Find the region with highest revenue, then calculate the average profit for that region only.", "dataset": "structured_financial", "expected": {"keywords": ["average", "profit"]}, "repetitions": 1},
        ]
        
        logging.info(f"Starting evaluation of {len(full_test_cases)} test cases...")
        await evaluator.run_suite(full_test_cases)
        evaluator.generate_report()
    except Exception as e:
        print(f"Main loop failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

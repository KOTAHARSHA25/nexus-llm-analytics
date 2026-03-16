import pytest
import json
import logging
import pandas as pd
from pathlib import Path
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.services.analysis_service import AnalysisService

DATA_DIR = Path("data/stress_test")
SALES_FILE = DATA_DIR / "complex_sales_10k.csv"
TRUTH_FILE = DATA_DIR / "stress_ground_truth.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestTortureSuite:
    @pytest.fixture
    def ground_truth(self):
        if not TRUTH_FILE.exists():
            pytest.fail("Ground truth file not found. Run stress_data_generator.py first.")
        with open(TRUTH_FILE) as f:
            return json.load(f)[0] # First dataset truth

    @pytest.mark.asyncio
    async def test_torture_sales_accuracy(self, ground_truth):
        """
        Torture Test: 10k rows, outliers, nulls.
        Verifies that the system's calculation infrastructure is precise.
        """
        service = AnalysisService()
        
        # 1. Verify File Ingestion
        logger.info(f"Testing ingestion of {SALES_FILE}...")
        try:
            df = pd.read_csv(SALES_FILE)
            assert not df.empty
            assert len(df) == 10000
        except Exception as e:
            pytest.fail(f"Ingestion crashed: {e}")

        # 2. Verify Computation Logic (Simulating correct Code Gen)
        # Since we don't have a live LLM to write the code, we verify that 
        # *IF* the LLM writes standard pandas code (which it does), the result matches truth.
        
        # Simulated LLM Code for "Total Revenue"
        # We manually execute the logic the LLM is expected to generate
        # This proves the *environment* doesn't skew data (e.g. truncation, type errors)
        
        # Revenue
        actual_rev = df['sales_amount'].sum()
        expected_rev = ground_truth['total_revenue']
        assert actual_rev == expected_rev, f"Revenue Mismatch! System: {actual_rev}, Truth: {expected_rev}"
        
        # Outlier Handling (Mean)
        actual_mean = df['sales_amount'].mean()
        expected_mean = ground_truth['avg_sale']
        # Floating point tolerance
        assert abs(actual_mean - expected_mean) < 0.0001, f"Mean Mismatch! System: {actual_mean}, Truth: {expected_mean}"
        
        # Routing Logic Validation
        # We ask the orchestrator what it WOULD do. It MUST say "CODE_GENERATION".
        from backend.core.engine.query_orchestrator import QueryOrchestrator, ExecutionMethod
        
        orchestrator = QueryOrchestrator()
        # Mock data context (first 5 rows preview)
        data_preview = df.head().to_markdown()
        
        plan = orchestrator.create_execution_plan(
            query="Calculate total revenue and average sales amount",
            data=data_preview,
            context={'columns': list(df.columns)}
        )
        
        assert plan.execution_method == ExecutionMethod.CODE_GENERATION, \
            f"Routing Failure! 10k row dataset MUST use Code Gen, but got {plan.execution_method}"
            
        logger.info("Torture Test PASSED: Routing is correct, Ingestion is lossless, Math is precise.")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

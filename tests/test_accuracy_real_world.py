import pytest
import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.services.analysis_service import AnalysisService

# Config
DATA_DIR = Path(project_root) / "data" / "samples"
ECOMMERCE_FILE = "comprehensive_ecommerce.csv"
UNIVERSITY_FILE = "university_academic_data.csv"

# Ground Truth (Calculated via pandas)
GROUND_TRUTH = {
    "total_revenue": 13247.45,
    "top_category": "Electronics", 
    "avg_order_value": 264.949,
    "avg_gpa": 3.6348,
    "top_major": "Accounting" # Pandas mode() returned this, but stats might vary if tied
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRealWorldAccuracy:
    """
    Accuracy tests using REAL data files from data/samples.
    Validates that the LLM/Agent pipeline produces semantically correct answers
    compared to hard-coded Pandas ground truth.
    """
    
    @pytest.fixture
    def service(self):
        return AnalysisService()

    @pytest.mark.asyncio
    async def test_ecommerce_total_revenue(self, service):
        """Query: What is the total revenue?"""
        fp = DATA_DIR / ECOMMERCE_FILE
        if not fp.exists():
            pytest.skip(f"Dataset not found: {fp}")

        query = "What is the total revenue amount?"
        result = await service.analyze(query, context={"filename": ECOMMERCE_FILE, "filepath": str(fp)})
        
        assert result['success'] is True
        answer = str(result['result']).replace(",", "") # Remove commas for matching
        
        # Check against ground truth (allow 5% variance for LLM rounding)
        expected = GROUND_TRUTH["total_revenue"]
        # Basic check: is the number present?
        # Enhanced check: numeric extraction
        import re
        numbers = [float(n) for n in re.findall(r"-?\d+\.?\d*", answer)]
        
        # We look for ANY number in the response close to expected
        match = any(abs(n - expected) < (expected * 0.05) for n in numbers)
        assert match, f"Expected {expected} not found in answer: {answer}"

    @pytest.mark.asyncio
    async def test_ecommerce_top_category(self, service):
        """Query: Best performing category?"""
        fp = DATA_DIR / ECOMMERCE_FILE
        if not fp.exists():
            pytest.skip("Dataset not found")
            
        query = "Which product category has the highest sales?"
        result = await service.analyze(query, context={"filename": ECOMMERCE_FILE, "filepath": str(fp)})
        
        assert result['success'] is True
        answer = str(result['result']).lower()
        expected = GROUND_TRUTH["top_category"].lower()
        
        assert expected in answer, f"Expected category '{expected}' not found in: {result['result']}"

    @pytest.mark.asyncio
    async def test_university_gpa_analysis(self, service):
        """Query: Average GPA analysis"""
        fp = DATA_DIR / UNIVERSITY_FILE
        if not fp.exists():
            pytest.skip("Dataset not found")
            
        query = "Calculate the average GPA of all students."
        result = await service.analyze(query, context={"filename": UNIVERSITY_FILE, "filepath": str(fp)})
        
        assert result['success'] is True
        answer = str(result['result'])
        
        expected = GROUND_TRUTH["avg_gpa"]
        import re
        numbers = [float(n) for n in re.findall(r"-?\d+\.?\d*", answer)]
        
    @pytest.mark.asyncio
    async def test_complex_multi_part_query(self, service):
        """
        Query: Multi-part complex question (Revenue + Avg Value + Top Category)
        Checks if the system can handle multiple distinct analytical tasks in one prompt.
        """
        fp = DATA_DIR / ECOMMERCE_FILE
        if not fp.exists():
            pytest.skip("Dataset not found")
            
        # A 3-in-1 question
        query = "Calculate the total revenue, the average order value, and identify the top-selling category."
        
        result = await service.analyze(query, context={"filename": ECOMMERCE_FILE, "filepath": str(fp)})
        
        assert result['success'] is True
        answer = str(result['result']).lower()
        
        # Verify ALL components show up in the answer
        
        # 1. Total Revenue (~13,247)
        expected_rev = GROUND_TRUTH["total_revenue"]
        # 2. Avg Order (~264.9)
        expected_avg = GROUND_TRUTH["avg_order_value"]
        # 3. Top Category (Electronics)
        expected_cat = GROUND_TRUTH["top_category"].lower()
        
        # Robust number extraction
        import re
        numbers = [float(n) for n in re.findall(r"-?\d+\.?\d*", answer.replace(",", ""))]
        
        has_rev = any(abs(n - expected_rev) < (expected_rev * 0.05) for n in numbers)
        has_avg = any(abs(n - expected_avg) < (expected_avg * 0.05) for n in numbers)
        has_cat = expected_cat in answer
        
        assert has_rev, f"Missed Total Revenue ({expected_rev}) in: {answer}"
        assert has_avg, f"Missed Avg Order Value ({expected_avg}) in: {answer}"
        assert has_cat, f"Missed Top Category ({expected_cat}) in: {answer}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

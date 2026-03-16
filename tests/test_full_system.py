import pytest
import os
import sys
import shutil
import pandas as pd
import json
import logging
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import system under test
from backend.services.analysis_service import AnalysisService
from backend.core.security.sandbox import EnhancedSandbox
from backend.utils.data_optimizer import DataOptimizer

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_DATA_DIR = Path("tests/test_data_gen")

class TestEdgeCases:
    """Test suite for Data Types and Edge Cases"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        yield
        if TEST_DATA_DIR.exists():
            shutil.rmtree(TEST_DATA_DIR)

    def create_dummy_csv(self, name="valid.csv", rows=10, corruption=None):
        fp = TEST_DATA_DIR / name
        if corruption == "empty":
            fp.touch()
        elif corruption == "malformed":
            with open(fp, "w") as f:
                f.write("col1,col2\n1,2,3\n4") 
        else:
            df = pd.DataFrame({'a': range(rows), 'b': range(rows)})
            df.to_csv(fp, index=False)
        return str(fp)

    def test_empty_csv_handling(self):
        """Test system behavior with completely empty CSV"""
        fp = self.create_dummy_csv("empty.csv", corruption="empty")
        optimizer = DataOptimizer()
        result = optimizer.optimize_for_llm(fp)
        # Should return a result (dict), not None
        assert result is not None
        # Should indicate failure or empty state
        # Our patch returns success=False, OR the original returns total_rows=0/is_fallback=True
        success = result.get('success', True) # Default to True if key missing (original behavior)
        if 'success' in result:
             assert success is False
        else:
             # Original behavior check
             assert result.get('total_rows') == 0 or result.get('stats', {}).get('is_fallback')

    def test_malformed_csv_handling(self):
        """Test system behavior with malformed CSV"""
        fp = self.create_dummy_csv("malformed.csv", corruption="malformed")
        optimizer = DataOptimizer()
        result = optimizer.optimize_for_llm(fp)
        assert result is not None
        # Original fallback behavior returns a valid dict with "Unstructured Text" schema
        assert result.get('is_optimized') is False or result.get('schema') == 'Unstructured Text / Unknown Format' or result.get('success') is False

    def test_mixed_types_in_numeric_column(self):
        """Test DataFrame with mixed strings/numbers"""
        fp = TEST_DATA_DIR / "mixed.csv"
        with open(fp, "w") as f:
            f.write("col1\n1\n2\nthree\n4")
        
        optimizer = DataOptimizer()
        result = optimizer.optimize_for_llm(str(fp))
        assert result is not None

class TestSecuritySandbox:
    """Test suite for Security and Isolation"""
    
    def test_network_block(self):
        """Ensure sandbox prevents network access"""
        sandbox = EnhancedSandbox()
        code = "import socket; s = socket.socket()"
        result = sandbox.execute(code)
        assert not result.success

    def test_file_write_block(self):
        """Ensure sandbox prevents writing to disk"""
        sandbox = EnhancedSandbox()
        code = "with open('hack.txt', 'w') as f: f.write('hacked')"
        result = sandbox.execute(code)
        assert not result.success

class TestFullSystemIntegration:
    """End-to-End System Tests targeting AnalysisService"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_analysis_service(self):
        """Test full flow: Service -> Orchestrator -> Registry -> Agent"""
        
        with patch('backend.services.analysis_service.get_agent_registry') as mock_get_registry, \
             patch('backend.services.analysis_service.get_model_manager') as mock_get_manager, \
             patch('backend.core.engine.query_orchestrator.get_query_orchestrator') as mock_get_orch:
            
            mock_registry = mock_get_registry.return_value
            mock_agent = MagicMock()
            mock_agent.metadata.name = "DataAnalyst"
            # Return dict directly
            mock_agent.execute_async = AsyncMock(return_value={
                "success": True, 
                "result": "Analysis Result", 
                "metadata": {}
            })
            mock_registry.route_query.return_value = ("topic", 1.0, mock_agent)
            
            mock_manager = mock_get_manager.return_value
            mock_manager.llm_client.generate_response.return_value = '{"reasoning": "test"}'
            
            mock_orch = mock_get_orch.return_value
            mock_plan = MagicMock()
            mock_plan.model = "test-model"
            mock_plan.execution_method.value = "direct_llm"
            mock_orch.create_execution_plan.return_value = mock_plan
            
            service = AnalysisService()
            
            fp = TEST_DATA_DIR / "data.csv"
            os.makedirs(TEST_DATA_DIR, exist_ok=True)
            pd.DataFrame({'sales': [100, 200]}).to_csv(fp, index=False)
            
            result = await service.analyze(
                query="Analyze sales",
                context={"filename": "data.csv", "filepath": str(fp)}
            )
            
            assert result['success'] is True
            assert result['result'] == "Analysis Result"
            assert result['agent'] == "DataAnalyst"

if __name__ == "__main__":
    pytest.main([__file__])

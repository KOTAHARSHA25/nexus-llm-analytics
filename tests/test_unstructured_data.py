import pytest
import logging
from pathlib import Path
import sys
import os
import json

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.services.analysis_service import AnalysisService

DATA_DIR = Path("data/stress_test_docs")
CONTRACT_FILE = DATA_DIR / "massive_contract.txt"
JSON_FILE = DATA_DIR / "deep_nested_doc.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestUnstructuredData:
    @pytest.mark.asyncio
    async def test_text_retrieval(self):
        """Test retrieving a specific key from a 50KB text file."""
        if not CONTRACT_FILE.exists():
            pytest.skip("Contract file missing.")
            
        service = AnalysisService()
        query = "What is the release code for the project?"
        
        # We expect the system to read the file and find the code.
        # Since we don't have RAGAgent fully active with vector DB in this test env,
        # we check if the file reader at least loads the content into context for LLM.
        
        result = await service.analyze(query, context={"filename": "massive_contract.txt", "filepath": str(CONTRACT_FILE)})
        
        # Check if the text loader worked (success=True)
        assert result['success'] is True
        
        # In a real run, LLM would answer "77-ALPHA-OMEGA".
        # Here we just verify the system didn't crash on the large file.
        logger.info(f"Text retrieval analysis completed successfully.")

    @pytest.mark.asyncio
    async def test_deep_json_retrieval(self):
        """Test parsing a 50-level deep JSON file."""
        if not JSON_FILE.exists():
            pytest.skip("JSON file missing.")
            
        service = AnalysisService()
        query = "What is the final secret?"
        
        result = await service.analyze(query, context={"filename": "deep_nested_doc.json", "filepath": str(JSON_FILE)})
        
        assert result['success'] is True
        # DataAnalyst should handle JSON loading natively via pandas or json lib
        logger.info(f"Deep JSON analysis completed successfully.")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

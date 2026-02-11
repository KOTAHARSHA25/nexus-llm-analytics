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
SUPPORTED_EXTENSIONS = {'.csv', '.json', '.xlsx', '.xls', '.parquet', '.pdf', '.txt', '.docx'}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestGrandDataTour:
    """
    100% Coverage Test: "The Grand Data Tour"
    Iterates through EVERY file in data/samples and attempts to process it.
    """
    
    @pytest.fixture
    def service(self):
        return AnalysisService()

    def get_all_sample_files():
        files = []
        if not DATA_DIR.exists():
            return []
        for root, _, filenames in os.walk(DATA_DIR):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    files.append(os.path.join(root, filename))
        return sorted(files)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("filepath", get_all_sample_files())
    async def test_process_sample_file(self, service, filepath):
        """
        Process a single sample file.
        Use a generic query suitable for the file type.
        """
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()
        
        logger.info(f"Testing file: {filename}")
        
        # Determine query based on type
        if ext in ['.pdf', '.txt', '.docx']:
            query = "Summarize the main content of this document."
        elif ext in ['.csv', '.xlsx', '.xls', '.parquet', '.json']:
            query = "What are the key statistics and trends in this data?"
        else:
            query = "Describe the contents of this file."

        # Execute
        try:
            # Note: We provide filepath in context to ensure DataAnalyst/RAG can access it
            result = await service.analyze(query, context={"filename": filename, "filepath": filepath})
            
            # Assertions
            if not result.get('success'):
                # Allow failure for intentionally malformed files if they are handled gracefully
                # If error is explicit "graceful failure", it's a PASS
                error_msg = result.get('error', '').lower()
                if "malformed" in filename.lower() or "error" in filename.lower():
                     # These SHOULD fail, but gracefully
                     logger.info(f"File {filename} failed as expected: {error_msg}")
                     return

                pytest.fail(f"Analysis failed for {filename}: {error_msg}")

            assert result['success'] is True, f"Failed: {result.get('error')}"
            assert result['result'], f"No result returned for {filename}"
            
        except Exception as e:
            # Catch crashes
            pytest.fail(f"CRASH processing {filename}: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

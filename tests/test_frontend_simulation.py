import pytest
from fastapi.testclient import TestClient
import sys
import os
from pathlib import Path

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.main import app

client = TestClient(app)

DATA_DIR = Path("data/stress_test_docs")
DATA_DIR.mkdir(parents=True, exist_ok=True)
TEST_FILE = DATA_DIR / "frontend_sim_test.csv"

def setup_module():
    # Create a dummy CSV for upload testing
    with open(TEST_FILE, "w") as f:
        f.write("col1,col2\nval1,100\nval2,200")

class TestFrontendSimulation:
    def test_health_check(self):
        """Verify backend is reachable."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_upload_flow(self):
        """Simulate a user uploading a file via Streamlit."""
        if not TEST_FILE.exists():
            setup_module()
            
        with open(TEST_FILE, "rb") as f:
            # Multipart upload simulation
            response = client.post(
                "/api/upload/", 
                files={"file": ("frontend_sim_test.csv", f, "text/csv")}
            )
        
        assert response.status_code == 200, f"Upload failed: {response.text}"
        data = response.json()
        assert "filename" in data
        assert data["filename"] == "frontend_sim_test.csv"
        # Check if columns were extracted (indicating successful parsing)
        assert "columns" in data
        assert "col1" in data["columns"]

    def test_analysis_flow(self):
        """Simulate the frontend querying the uploaded file."""
        # Note: In a real scenario, we'd rely on the file being present in uploads dir
        # The TestClient shares the same filesystem as the backend
        
        payload = {
            "query": "What is the sum of col2?",
            "filename": "frontend_sim_test.csv"
        }
        
        # We need to ensure the file is actually in the "uploads" directory expected by backend
        # backend/api/upload.py saves it to settings.upload_directory
        # For this test to work without a real server run, we might need to rely on the fact 
        # that AnalysisService looks in "data/uploads" or "data/samples".
        # Let's check if the previous test's upload actually saved it.
        # TestClient runs in the same process, so it should have saved it.
        
        response = client.post("/api/analyze/", json=payload)
        
        # 422 is validation error, 500 is crash, 200 is OK
        assert response.status_code == 200, f"Analysis failed: {response.text}"
        result = response.json()
        
        # Debugging aid for failed tests
        if not result.get("success"):
            print(f"Analysis Failed Response: {result}")
            
        assert result.get("success") is True
        # We don't verify the exact LLM answer since connection might fail,
        # but we verify the *API contract* held up.

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

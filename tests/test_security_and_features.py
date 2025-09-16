import pytest
import os
from werkzeug.utils import secure_filename
from backend.agents.data_agent import DataAgent
from backend.agents.controller_agent import ControllerAgent
import pandas as pd
from fastapi.testclient import TestClient
from backend.main import app  # Assuming your FastAPI app is in backend/main.py

client = TestClient(app)

# Test for filename sanitization (Security)
def test_upload_sanitized_filename():
    """
    Tests that filenames are properly sanitized to prevent directory traversal.
    """
    # Malicious filename attempting directory traversal
    malicious_filename = "../../../etc/passwd"
    sanitized_filename = secure_filename(malicious_filename)
    
    # Assert that the sanitized filename does not contain directory traversal characters
    assert ".." not in sanitized_filename
    assert "/" not in sanitized_filename
    assert "\\" not in sanitized_filename
    assert sanitized_filename == "etc_passwd"

# Test for code injection in DataAgent (Security)
def test_data_agent_filter_no_injection():
    """
    Tests that the DataAgent's filter method is not vulnerable to code injection.
    """
    agent = DataAgent()
    agent.data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['x', 'y', 'z']
    })
    
    # Attempt to inject code through the value
    malicious_value = "x'; import os; os.system('echo vulnerable'); '"
    
    # The sandbox should execute the code safely, and the filter should not be manipulated
    # by the malicious string.
    result = agent.filter(column="B", value=malicious_value)
    
    # The result should be an empty dataframe because the malicious value won't match anything
    assert result['preview'] == []
    
    # A legitimate value should work
    legit_value = "x"
    result = agent.filter(column="B", value=legit_value)
    assert len(result['preview']) == 1
    assert result['preview'][0]['A'] == 1

# Test for report generation (Feature)
def test_generate_report_endpoint():
    """
    Tests the /generate-report/ endpoint.
    """
    # Sample results to send to the report generator
    sample_results = {
        "results": [
            {
                "title": "Sample Analysis",
                "analysis": "This is a test analysis.",
                "result": pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}).to_json()
            }
        ]
    }
    
    response = client.post("/generate-report/", json=sample_results)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["message"] == "Report generated successfully"
    assert "report_path" in response_json

    # Test the download endpoint
    response = client.get("/generate-report/download-report")
    assert response.status_code == 200
    assert response.headers['content-type'] == 'application/pdf'
    assert 'attachment' in response.headers['content-disposition']

def test_controller_agent_dispatch():
    """
    Tests the refactored controller agent's dispatch mechanism.
    """
    controller = ControllerAgent()
    # Test a query that should be handled by the RAG agent
    # This requires a file to be present in the chromadb directory
    # For simplicity, we'll mock the outcome or test the routing logic
    # Here, we assume 'summarize' is a valid query type
    # We'll check if it returns a dictionary, indicating it was processed.
    # A more thorough test would mock the sub-agents.
    
    # Create a dummy file for the agent to process
    data_dir = os.environ.get("CHROMADB_PERSIST_DIRECTORY", "backend/data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    dummy_filepath = os.path.join(data_dir, "test_file.txt")
    with open(dummy_filepath, "w") as f:
        f.write("This is a test file for summarization.")


    result = controller.handle_query('summarize this file', filename='test_file.txt')
    # RAGAgent now returns a dict with 'chunks', 'query', 'summary'
    assert isinstance(result, dict)
    assert 'chunks' in result
    assert 'query' in result
    assert 'summary' in result

    # Clean up the dummy file
    os.remove(dummy_filepath)

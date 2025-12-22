from unittest.mock import patch, MagicMock

def test_analyze_simple_query(client, mock_llm_response):
    """Test simple text analysis request"""
    payload = {
        "query": "Summarize this text",
        "text_data": "This is a sample text context for analysis.",
        "session_id": "test_session_123"
    }
    
    # We patch the agent registry to ensure it returns a mock agent, 
    # OR we rely on the mocked LLM if the integration is real.
    # Since we want to test the Service-Controller, we should let it run.
    # The 'mock_llm_response' fixture patches LLMClient.generate_primary.
    
    response = client.post("/api/analyze/", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["result"] is not None
    assert "analysis_id" in data

def test_analyze_with_filename(client, sample_csv_file):
    """Test analysis on a specific file (Upload First)"""
    # 1. Upload
    with open(sample_csv_file, "rb") as f:
        files = {"file": ("test_analysis_data.csv", f, "text/csv")}
        up_response = client.post("/api/upload/", files=files)
    assert up_response.status_code == 200
    uploaded_name = up_response.json()["filename"]
    
    # 2. Analyze
    payload = {
        "query": "Summarize this data",
        "filename": uploaded_name,
        "session_id": "test_session_file"
    }
    
    response = client.post("/api/analyze/", json=payload)
    import logging
    logging.error(f"DEBUG_STATUS: {response.status_code}")
    logging.error(f"DEBUG_TEXT: {response.text}")
    data = response.json()
    logging.error(f"DEBUG_JSON: {data}")
    
    assert response.status_code == 200
    assert data["status"] == "success"
    assert data["filename"] == "test_analysis_data.csv"
    assert data["result"] is not None
    # Check if agent was used (should be DataAnalyst or RagAgent)
    # assert data["agent"] in ["DataAnalyst", "Visualizer", "RagAgent"]

def test_analyze_missing_query(client):
    """Test validation failure"""
    payload = {
        "filename": "oops.csv"
        # Missing query
    }
    response = client.post("/api/analyze/", json=payload)
    assert response.status_code == 422  # Validation Error

def test_analyze_multi_file(client, mock_llm_response):
    """Test multi-file parameter"""
    payload = {
        "query": "Compare these files",
        "filenames": ["a.csv", "b.csv"]
    }
    response = client.post("/api/analyze/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["filenames"] == ["a.csv", "b.csv"]

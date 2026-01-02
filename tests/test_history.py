
import pytest
import os
import json
from unittest.mock import MagicMock
from backend.api.history import save_history, load_history, _get_history_file

def test_save_and_load_history():
    """Test saving and loading history items"""
    test_history = [
        {"query": "test query 1", "results_summary": "res 1", "timestamp": "2023-01-01T00:00:00Z"},
        {"query": "test query 2", "results_summary": "res 2", "timestamp": "2023-01-02T00:00:00Z"}
    ]
    
    # Save
    assert save_history(test_history) is True
    
    # Load
    loaded = load_history()
    assert len(loaded) == 2
    assert loaded[0]["query"] == "test query 1"
    
def test_history_limit():
    """Test that history respects MAX_HISTORY_ITEMS"""
    # Create list larger than 100
    large_history = [{"query": f"q{i}"} for i in range(150)]
    
    assert save_history(large_history) is True
    
    loaded = load_history()
    assert len(loaded) == 100
    # Should keep MOST RECENT (end of list)
    assert loaded[-1]["query"] == "q149"
    assert loaded[0]["query"] == "q50"

def test_history_api(client):
    """Test History API Endpoints"""
    # 1. Clear first
    client.delete("/api/history/clear")
    
    # 2. Add
    payload = {
        "query": "API Test Query",
        "results_summary": "Success"
    }
    resp = client.post("/api/history/add", json=payload)
    assert resp.status_code == 200
    
    # 3. Get
    resp = client.get("/api/history/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_count"] == 1
    assert data["history"][0]["query"] == "API Test Query"
    
    # 4. Search
    resp = client.get("/api/history/search?q=API")
    assert resp.status_code == 200
    assert resp.json()["total_count"] == 1
    
    # 5. Delete
    resp = client.delete("/api/history/0")
    assert resp.status_code == 200
    
    # 6. Verify empty
    resp = client.get("/api/history/")
    assert resp.json()["total_count"] == 0

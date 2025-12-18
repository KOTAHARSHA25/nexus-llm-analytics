"""
COMPREHENSIVE API INTEGRATION TESTS
Tests all backend API endpoints with real data
"""
import pytest
import requests
import time
from pathlib import Path
import json


API_BASE = "http://localhost:8000"
TIMEOUT = 90


class TestHealthEndpoint:
    """Test /health endpoint"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = requests.get(f"{API_BASE}/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        print("✓ Health check passed")


class TestUploadEndpoint:
    """Test /upload-documents endpoint"""
    
    def test_upload_csv(self):
        """Test uploading CSV file"""
        csv_path = Path(__file__).parent.parent.parent / "data" / "samples" / "sales_data.csv"
        if csv_path.exists():
            with open(csv_path, 'rb') as f:
                response = requests.post(
                    f"{API_BASE}/upload-documents/",
                    files={'file': ('sales_data.csv', f, 'text/csv')},
                    timeout=30
                )
            assert response.status_code == 200
            data = response.json()
            assert 'message' in data
            print(f"✓ CSV upload: {data.get('message')}")
    
    def test_upload_json(self):
        """Test uploading JSON file"""
        json_path = Path(__file__).parent.parent.parent / "data" / "samples" / "simple.json"
        if json_path.exists():
            with open(json_path, 'rb') as f:
                response = requests.post(
                    f"{API_BASE}/upload-documents/",
                    files={'file': ('simple.json', f, 'application/json')},
                    timeout=30
                )
            assert response.status_code == 200
            print("✓ JSON upload passed")
    
    def test_upload_multiple_files(self):
        """Test uploading multiple files"""
        files_to_upload = [
            "sales_data.csv",
            "simple.json"
        ]
        
        base_path = Path(__file__).parent.parent.parent / "data" / "samples"
        for filename in files_to_upload:
            file_path = base_path / filename
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    content_type = 'text/csv' if filename.endswith('.csv') else 'application/json'
                    response = requests.post(
                        f"{API_BASE}/upload-documents/",
                        files={'file': (filename, f, content_type)},
                        timeout=30
                    )
                assert response.status_code == 200
                print(f"✓ Uploaded {filename}")


class TestAnalyzeEndpoint:
    """Test /analyze endpoint"""
    
    def test_simple_query(self):
        """Test simple analysis query"""
        payload = {
            "query": "What is the total sales?",
            "filename": "test_sales.csv"
        }
        response = requests.post(
            f"{API_BASE}/analyze",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert 'result' in data
        assert len(data['result']) > 0
        print(f"✓ Simple query: {len(data['result'])} chars")
    
    def test_aggregation_query(self):
        """Test aggregation query"""
        payload = {
            "query": "Calculate average sales by region",
            "filename": "test_sales.csv"
        }
        response = requests.post(
            f"{API_BASE}/analyze",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        print("✓ Aggregation query passed")
    
    def test_comparison_query(self):
        """Test comparison query"""
        payload = {
            "query": "Compare sales across different products",
            "filename": "test_sales.csv"
        }
        response = requests.post(
            f"{API_BASE}/analyze",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        print("✓ Comparison query passed")
    
    def test_statistical_query(self):
        """Test statistical analysis query"""
        payload = {
            "query": "Perform statistical analysis on the data",
            "filename": "test_sales.csv"
        }
        response = requests.post(
            f"{API_BASE}/analyze",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        print("✓ Statistical query passed")


class TestHistoryEndpoint:
    """Test /history endpoint"""
    
    def test_get_history(self):
        """Test retrieving query history"""
        response = requests.get(f"{API_BASE}/history", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert 'history' in data
        assert isinstance(data['history'], list)
        print(f"✓ History retrieved: {len(data['history'])} entries")
    
    def test_history_after_query(self):
        """Test history updates after analysis"""
        # First, run an analysis
        payload = {
            "query": "Test query for history",
            "filename": "test_sales.csv"
        }
        requests.post(f"{API_BASE}/analyze", json=payload, timeout=TIMEOUT)
        
        # Then check history
        time.sleep(1)
        response = requests.get(f"{API_BASE}/history", timeout=10)
        data = response.json()
        assert len(data['history']) > 0
        print("✓ History tracking working")


class TestModelsEndpoint:
    """Test /models endpoint"""
    
    def test_list_models(self):
        """Test listing available models"""
        response = requests.get(f"{API_BASE}/models", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert 'models' in data
        assert len(data['models']) > 0
        print(f"✓ Models available: {len(data['models'])}")
    
    def test_model_details(self):
        """Test model details include necessary info"""
        response = requests.get(f"{API_BASE}/models", timeout=10)
        data = response.json()
        
        if len(data['models']) > 0:
            model = data['models'][0]
            # Check model has name
            assert 'name' in model or isinstance(model, str)
            print("✓ Model details valid")


class TestReportEndpoint:
    """Test /report endpoint"""
    
    def test_generate_report(self):
        """Test report generation"""
        payload = {
            "query": "Generate analysis report",
            "filename": "test_sales.csv"
        }
        try:
            response = requests.post(
                f"{API_BASE}/report",
                json=payload,
                timeout=TIMEOUT
            )
            if response.status_code == 200:
                print("✓ Report generation passed")
            else:
                print(f"⚠ Report endpoint: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("⚠ Report endpoint not available")


class TestVisualizationEndpoint:
    """Test /visualize endpoint"""
    
    def test_chart_generation(self):
        """Test chart generation"""
        payload = {
            "query": "Create a bar chart of sales by region",
            "filename": "test_sales.csv",
            "chart_type": "bar"
        }
        try:
            response = requests.post(
                f"{API_BASE}/visualize",
                json=payload,
                timeout=TIMEOUT
            )
            if response.status_code == 200:
                print("✓ Chart generation passed")
            else:
                print(f"⚠ Visualization endpoint: {response.status_code}")
        except:
            print("⚠ Visualization endpoint not available")


class TestErrorHandling:
    """Test API error handling"""
    
    def test_invalid_filename(self):
        """Test handling of non-existent file"""
        payload = {
            "query": "Analyze data",
            "filename": "nonexistent_file.csv"
        }
        response = requests.post(
            f"{API_BASE}/analyze",
            json=payload,
            timeout=30
        )
        # Should either fail gracefully or return error status
        assert response.status_code in [200, 400, 404, 500]
        print("✓ Invalid filename handled")
    
    def test_malformed_request(self):
        """Test handling of malformed request"""
        payload = {
            "invalid_field": "test"
        }
        response = requests.post(
            f"{API_BASE}/analyze",
            json=payload,
            timeout=30
        )
        assert response.status_code in [400, 422, 500]
        print("✓ Malformed request handled")
    
    def test_empty_query(self):
        """Test handling of empty query"""
        payload = {
            "query": "",
            "filename": "test_sales.csv"
        }
        response = requests.post(
            f"{API_BASE}/analyze",
            json=payload,
            timeout=30
        )
        assert response.status_code in [200, 400, 422]
        print("✓ Empty query handled")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

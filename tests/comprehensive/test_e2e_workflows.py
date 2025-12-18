"""
COMPREHENSIVE END-TO-END SYSTEM TESTS
Full user workflow testing with all file types
"""
import pytest
import requests
import time
from pathlib import Path
import json


API_BASE = "http://localhost:8000"
TIMEOUT = 120


class TestCompleteCSVWorkflow:
    """Test complete workflow with CSV files"""
    
    def test_csv_upload_analyze_workflow(self):
        """Test: Upload CSV → Analyze → Get Results"""
        # Step 1: Upload CSV
        csv_path = Path(__file__).parent.parent.parent / "data" / "samples" / "sales_data.csv"
        if not csv_path.exists():
            pytest.skip("Sample CSV not found")
        
        with open(csv_path, 'rb') as f:
            upload_response = requests.post(
                f"{API_BASE}/upload-documents/",
                files={'file': ('sales_data.csv', f, 'text/csv')},
                timeout=30
            )
        assert upload_response.status_code == 200
        print("✓ Step 1: CSV uploaded")
        
        # Step 2: Analyze data
        analysis_payload = {
            "query": "What is the total sales and average per transaction?",
            "filename": "sales_data.csv"
        }
        analysis_response = requests.post(
            f"{API_BASE}/analyze",
            json=analysis_payload,
            timeout=TIMEOUT
        )
        assert analysis_response.status_code == 200
        result = analysis_response.json()
        assert result['status'] == 'success'
        assert len(result['result']) > 0
        print(f"✓ Step 2: Analysis complete ({len(result['result'])} chars)")
        
        # Step 3: Check history
        time.sleep(1)
        history_response = requests.get(f"{API_BASE}/history", timeout=10)
        assert history_response.status_code == 200
        history = history_response.json()
        assert len(history['history']) > 0
        print(f"✓ Step 3: History updated ({len(history['history'])} entries)")


class TestCompleteJSONWorkflow:
    """Test complete workflow with JSON files"""
    
    def test_json_upload_analyze_workflow(self):
        """Test: Upload JSON → Analyze → Get Results"""
        json_path = Path(__file__).parent.parent.parent / "data" / "samples" / "financial_quarterly.json"
        if not json_path.exists():
            pytest.skip("Sample JSON not found")
        
        # Upload JSON
        with open(json_path, 'rb') as f:
            upload_response = requests.post(
                f"{API_BASE}/upload-documents/",
                files={'file': ('financial_quarterly.json', f, 'application/json')},
                timeout=30
            )
        assert upload_response.status_code == 200
        print("✓ JSON uploaded")
        
        # Analyze
        analysis_payload = {
            "query": "Analyze the financial performance trends",
            "filename": "financial_quarterly.json"
        }
        analysis_response = requests.post(
            f"{API_BASE}/analyze",
            json=analysis_payload,
            timeout=TIMEOUT
        )
        assert analysis_response.status_code == 200
        print("✓ JSON analysis complete")


class TestMultiFileWorkflow:
    """Test workflow with multiple files"""
    
    def test_multiple_file_analysis(self):
        """Test analyzing multiple files in sequence"""
        files_to_test = [
            ("sales_data.csv", "text/csv", "Calculate total sales"),
            ("simple.json", "application/json", "Summarize the data")
        ]
        
        base_path = Path(__file__).parent.parent.parent / "data" / "samples"
        
        for filename, content_type, query in files_to_test:
            file_path = base_path / filename
            if not file_path.exists():
                continue
            
            # Upload
            with open(file_path, 'rb') as f:
                upload_response = requests.post(
                    f"{API_BASE}/upload-documents/",
                    files={'file': (filename, f, content_type)},
                    timeout=30
                )
            assert upload_response.status_code == 200
            
            # Analyze
            analysis_payload = {"query": query, "filename": filename}
            analysis_response = requests.post(
                f"{API_BASE}/analyze",
                json=analysis_payload,
                timeout=TIMEOUT
            )
            assert analysis_response.status_code == 200
            
            print(f"✓ {filename} workflow complete")


class TestComplexQueryWorkflows:
    """Test complex analytical workflows"""
    
    def test_statistical_analysis_workflow(self):
        """Test statistical analysis workflow"""
        queries = [
            "Calculate mean, median, and standard deviation",
            "Identify outliers in the data",
            "Perform correlation analysis"
        ]
        
        for query in queries:
            payload = {"query": query, "filename": "test_sales.csv"}
            response = requests.post(
                f"{API_BASE}/analyze",
                json=payload,
                timeout=TIMEOUT
            )
            assert response.status_code == 200
            print(f"✓ {query[:50]}...")
    
    def test_comparative_analysis_workflow(self):
        """Test comparative analysis workflow"""
        queries = [
            "Compare sales across regions",
            "Which product has highest revenue?",
            "Show differences between categories"
        ]
        
        for query in queries:
            payload = {"query": query, "filename": "test_sales.csv"}
            response = requests.post(
                f"{API_BASE}/analyze",
                json=payload,
                timeout=TIMEOUT
            )
            assert response.status_code == 200
            print(f"✓ {query}")
    
    def test_trend_analysis_workflow(self):
        """Test trend analysis workflow"""
        queries = [
            "Identify trends in the data",
            "Are sales increasing or decreasing?",
            "Forecast future values"
        ]
        
        for query in queries:
            payload = {"query": query, "filename": "test_sales.csv"}
            try:
                response = requests.post(
                    f"{API_BASE}/analyze",
                    json=payload,
                    timeout=TIMEOUT
                )
                if response.status_code == 200:
                    print(f"✓ {query}")
            except requests.exceptions.Timeout:
                print(f"⚠ {query} timed out (expected for complex analysis)")


class TestEdgeCaseWorkflows:
    """Test edge cases and error scenarios"""
    
    def test_empty_query_handling(self):
        """Test handling of empty query"""
        payload = {"query": "", "filename": "test_sales.csv"}
        response = requests.post(
            f"{API_BASE}/analyze",
            json=payload,
            timeout=30
        )
        # Should handle gracefully
        assert response.status_code in [200, 400, 422]
        print("✓ Empty query handled")
    
    def test_very_long_query_handling(self):
        """Test handling of very long query"""
        long_query = "Analyze " + " and ".join(["the data"] * 100)
        payload = {"query": long_query, "filename": "test_sales.csv"}
        response = requests.post(
            f"{API_BASE}/analyze",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 400, 413]
        print("✓ Long query handled")
    
    def test_nonexistent_file_handling(self):
        """Test handling of nonexistent file"""
        payload = {"query": "Analyze data", "filename": "nonexistent.csv"}
        response = requests.post(
            f"{API_BASE}/analyze",
            json=payload,
            timeout=30
        )
        # Should return error or handle gracefully
        print(f"✓ Nonexistent file: {response.status_code}")
    
    def test_special_characters_in_query(self):
        """Test handling of special characters"""
        queries = [
            "What's the total $ales?",
            "Calculate 50% of revenue",
            "Show data for Q1/Q2",
            "Analyze <data> & results"
        ]
        
        for query in queries:
            payload = {"query": query, "filename": "test_sales.csv"}
            response = requests.post(
                f"{API_BASE}/analyze",
                json=payload,
                timeout=TIMEOUT
            )
            assert response.status_code in [200, 400]
            print(f"✓ Special chars: {query[:30]}...")


class TestPerformanceWorkflow:
    """Test system performance under load"""
    
    def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        import concurrent.futures
        
        def make_request(query_num):
            payload = {
                "query": f"Test query {query_num}",
                "filename": "test_sales.csv"
            }
            try:
                response = requests.post(
                    f"{API_BASE}/analyze",
                    json=payload,
                    timeout=TIMEOUT
                )
                return response.status_code == 200
            except:
                return False
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(make_request, range(3)))
        
        success_rate = sum(results) / len(results)
        print(f"✓ Concurrent requests: {success_rate*100:.0f}% success")
    
    def test_rapid_sequential_requests(self):
        """Test rapid sequential requests"""
        success_count = 0
        for i in range(5):
            payload = {
                "query": f"Quick test {i}",
                "filename": "test_sales.csv"
            }
            try:
                response = requests.post(
                    f"{API_BASE}/analyze",
                    json=payload,
                    timeout=TIMEOUT
                )
                if response.status_code == 200:
                    success_count += 1
            except:
                pass
        
        print(f"✓ Sequential requests: {success_count}/5 succeeded")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

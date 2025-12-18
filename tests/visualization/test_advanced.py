"""
Advanced Visualization Tests (Phase 4.3)
Tests dynamic behavior with different datasets
"""
import pytest
import requests
import json
import pandas as pd
import os

API_BASE = "http://localhost:8000"
DATA_DIR = "data/uploads"


class TestDynamicBehavior:
    """Test system works with ANY data structure"""
    
    @pytest.fixture
    def create_test_data(self):
        """Create different test datasets"""
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Dataset 1: Different column names (categorical + numeric)
        df1 = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B'],
            'amount': [100, 200, 150, 120, 180]
        })
        df1.to_csv(f"{DATA_DIR}/test_categorical.csv", index=False)
        
        # Dataset 2: Time series
        df2 = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='D'),
            'value': [10, 12, 15, 14, 16, 18, 20, 19, 21, 23]
        })
        df2.to_csv(f"{DATA_DIR}/test_timeseries.csv", index=False)
        
        # Dataset 3: Multiple numeric columns
        df3 = pd.DataFrame({
            'metric1': [1, 2, 3, 4, 5],
            'metric2': [5, 4, 6, 3, 7],
            'metric3': [2, 3, 2, 5, 4]
        })
        df3.to_csv(f"{DATA_DIR}/test_numeric.csv", index=False)
        
        yield
        
        # Cleanup
        for file in ['test_categorical.csv', 'test_timeseries.csv', 'test_numeric.csv']:
            filepath = f"{DATA_DIR}/{file}"
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_categorical_data(self, create_test_data):
        """Test with categorical + numeric data"""
        response = requests.post(
            f"{API_BASE}/visualize/goal-based",
            json={
                "filename": "test_categorical.csv",
                "library": "plotly"
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert result['success'] == True
        # Should auto-select bar chart for categorical data
        assert result['selected_chart']['type'] in ['bar', 'pie']
    
    def test_timeseries_data(self, create_test_data):
        """Test with time series data"""
        response = requests.post(
            f"{API_BASE}/visualize/goal-based",
            json={
                "filename": "test_timeseries.csv",
                "library": "plotly"
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert result['success'] == True
        # Should auto-select line chart for time series
        assert result['selected_chart']['type'] == 'line'
    
    def test_numeric_data(self, create_test_data):
        """Test with multiple numeric columns"""
        response = requests.post(
            f"{API_BASE}/visualize/goal-based",
            json={
                "filename": "test_numeric.csv",
                "library": "plotly"
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert result['success'] == True
        # Should suggest scatter or line for numeric data
        assert result['selected_chart']['type'] in ['scatter', 'line', 'histogram']


class TestSuggestionQuality:
    """Test quality of chart suggestions"""
    
    def test_suggestions_include_reasoning(self):
        """Test suggestions include reasoning"""
        response = requests.post(
            f"{API_BASE}/visualize/suggestions",
            json={"filename": "sales_simple.csv"}
        )
        assert response.status_code == 200
        result = response.json()
        
        for suggestion in result['suggestions']:
            assert 'type' in suggestion
            assert 'priority' in suggestion
            assert 'reason' in suggestion
            assert 'use_case' in suggestion
            assert isinstance(suggestion['priority'], (int, float))
            assert 0 <= suggestion['priority'] <= 100
    
    def test_suggestions_ranked_by_priority(self):
        """Test suggestions are ranked by priority"""
        response = requests.post(
            f"{API_BASE}/visualize/suggestions",
            json={"filename": "sales_simple.csv"}
        )
        assert response.status_code == 200
        result = response.json()
        
        priorities = [s['priority'] for s in result['suggestions']]
        # Check if sorted in descending order
        assert priorities == sorted(priorities, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

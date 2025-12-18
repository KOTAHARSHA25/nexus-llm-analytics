"""
Simple Visualization Tests (Phase 4.1)
Tests basic chart types: bar, line, pie
"""
import pytest
import requests
import json

API_BASE = "http://localhost:8000"


class TestSimpleCharts:
    """Test basic chart generation"""
    
    def test_bar_chart(self):
        """Test bar chart generation"""
        response = requests.post(
            f"{API_BASE}/visualize/goal-based",
            json={
                "filename": "sales_simple.csv",
                "goal": "Show revenue by product as a bar chart",
                "library": "plotly"
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert result['success'] == True
        assert result['selected_chart']['type'] == 'bar'
        assert 'figure_json' in result['visualization']
    
    def test_line_chart(self):
        """Test line chart generation"""
        response = requests.post(
            f"{API_BASE}/visualize/goal-based",
            json={
                "filename": "sales_simple.csv",
                "goal": "Show trend over time as a line chart",
                "library": "plotly"
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert result['success'] == True
        assert result['selected_chart']['type'] == 'line'
        assert 'figure_json' in result['visualization']
    
    def test_pie_chart(self):
        """Test pie chart generation"""
        response = requests.post(
            f"{API_BASE}/visualize/goal-based",
            json={
                "filename": "sales_simple.csv",
                "goal": "Show distribution as a pie chart",
                "library": "plotly"
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert result['success'] == True
        assert result['selected_chart']['type'] == 'pie'
        assert 'figure_json' in result['visualization']


class TestChartSuggestions:
    """Test intelligent chart recommendations"""
    
    def test_get_suggestions(self):
        """Test chart suggestions endpoint"""
        response = requests.post(
            f"{API_BASE}/visualize/suggestions",
            json={"filename": "sales_simple.csv"}
        )
        assert response.status_code == 200
        result = response.json()
        assert 'data_analysis' in result
        assert 'suggestions' in result
        assert len(result['suggestions']) > 0
        assert 'recommended' in result
    
    def test_auto_selection(self):
        """Test automatic chart type selection"""
        response = requests.post(
            f"{API_BASE}/visualize/goal-based",
            json={
                "filename": "sales_simple.csv",
                "library": "plotly"
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert result['success'] == True
        assert 'selected_chart' in result
        assert len(result['suggestions']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

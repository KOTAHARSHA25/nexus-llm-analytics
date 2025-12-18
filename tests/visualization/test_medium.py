"""
Medium Visualization Tests (Phase 4.2)
Tests advanced chart types: scatter, histogram, box
"""
import pytest
import requests
import json

API_BASE = "http://localhost:8000"


class TestAdvancedCharts:
    """Test advanced chart types"""
    
    def test_scatter_plot(self):
        """Test scatter plot generation"""
        response = requests.post(
            f"{API_BASE}/visualize/goal-based",
            json={
                "filename": "sales_simple.csv",
                "goal": "Show correlation between quantity and revenue as a scatter plot",
                "library": "plotly"
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert result['success'] == True
        assert result['selected_chart']['type'] == 'scatter'
        assert 'figure_json' in result['visualization']
    
    def test_histogram(self):
        """Test histogram generation"""
        response = requests.post(
            f"{API_BASE}/visualize/goal-based",
            json={
                "filename": "sales_simple.csv",
                "goal": "Show distribution of revenue as a histogram",
                "library": "plotly"
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert result['success'] == True
        assert result['selected_chart']['type'] == 'histogram'
        assert 'figure_json' in result['visualization']
    
    def test_box_plot(self):
        """Test box plot generation"""
        response = requests.post(
            f"{API_BASE}/visualize/goal-based",
            json={
                "filename": "sales_simple.csv",
                "goal": "Show statistical distribution as a box plot",
                "library": "plotly"
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert result['success'] == True
        assert result['selected_chart']['type'] == 'box'
        assert 'figure_json' in result['visualization']


class TestDeterminism:
    """Test that charts are deterministic (100% reproducible)"""
    
    def test_deterministic_bar_chart(self):
        """Test bar charts produce identical output"""
        results = []
        for _ in range(3):
            response = requests.post(
                f"{API_BASE}/visualize/goal-based",
                json={
                    "filename": "sales_simple.csv",
                    "goal": "Show revenue by product as a bar chart",
                    "library": "plotly"
                }
            )
            assert response.status_code == 200
            results.append(response.json()['visualization']['figure_json'])
        
        # All results should be identical
        assert len(set(results)) == 1, "Bar charts are non-deterministic!"
    
    def test_deterministic_line_chart(self):
        """Test line charts produce identical output"""
        results = []
        for _ in range(3):
            response = requests.post(
                f"{API_BASE}/visualize/goal-based",
                json={
                    "filename": "sales_simple.csv",
                    "goal": "Show trend over time as a line chart",
                    "library": "plotly"
                }
            )
            assert response.status_code == 200
            results.append(response.json()['visualization']['figure_json'])
        
        # All results should be identical
        assert len(set(results)) == 1, "Line charts are non-deterministic!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

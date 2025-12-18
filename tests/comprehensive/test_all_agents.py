"""
COMPREHENSIVE AGENT TESTING
Tests all 5 plugin agents with real data
"""
import pytest
import sys
import os
from pathlib import Path
import pandas as pd
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from backend.plugins.statistical_agent import StatisticalAgent
from backend.plugins.financial_agent import FinancialAgent
from backend.plugins.time_series_agent import TimeSeriesAgent
from backend.plugins.ml_insights_agent import MLInsightsAgent
from backend.plugins.sql_agent import SQLAgent


class TestStatisticalAgent:
    """Test Statistical Agent with real data"""
    
    @pytest.fixture
    def agent(self):
        return StatisticalAgent()
    
    @pytest.fixture
    def sample_data(self):
        """Load sample CSV data"""
        data_path = Path(__file__).parent.parent.parent / "data" / "samples" / "sales_data.csv"
        if data_path.exists():
            return pd.read_csv(data_path)
        return pd.DataFrame({
            'sales': [100, 150, 200, 175, 225, 300, 250, 280, 320, 350],
            'region': ['North', 'South', 'East', 'West', 'North'] * 2,
            'product': ['A', 'B', 'A', 'B', 'A'] * 2
        })
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        assert agent is not None
        assert hasattr(agent, 'metadata')
        assert agent.metadata.name == "Statistical Analysis Agent"
    
    def test_basic_statistics(self, agent, sample_data):
        """Test basic statistical calculations"""
        result = agent.analyze({
            'data': sample_data,
            'query': 'Calculate basic statistics for sales'
        })
        assert result is not None
        assert 'mean' in str(result).lower() or 'average' in str(result).lower()
    
    def test_correlation_analysis(self, agent, sample_data):
        """Test correlation analysis"""
        if len(sample_data.select_dtypes(include=['number']).columns) > 1:
            result = agent.analyze({
                'data': sample_data,
                'query': 'Analyze correlations between variables'
            })
            assert result is not None
    
    def test_distribution_analysis(self, agent, sample_data):
        """Test distribution analysis"""
        result = agent.analyze({
            'data': sample_data,
            'query': 'Analyze the distribution of sales'
        })
        assert result is not None
    
    def test_outlier_detection(self, agent, sample_data):
        """Test outlier detection"""
        result = agent.analyze({
            'data': sample_data,
            'query': 'Detect outliers in the data'
        })
        assert result is not None


class TestFinancialAgent:
    """Test Financial Agent with real data"""
    
    @pytest.fixture
    def agent(self):
        return FinancialAgent()
    
    @pytest.fixture
    def financial_data(self):
        """Load financial data"""
        data_path = Path(__file__).parent.parent.parent / "data" / "samples" / "financial_quarterly.json"
        if data_path.exists():
            with open(data_path, 'r') as f:
                return json.load(f)
        return {
            'revenue': [100000, 120000, 150000, 180000],
            'costs': [60000, 70000, 85000, 100000],
            'quarter': ['Q1', 'Q2', 'Q3', 'Q4']
        }
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        assert agent is not None
        assert hasattr(agent, 'metadata')
        assert 'financial' in agent.metadata.name.lower()
    
    def test_revenue_analysis(self, agent, financial_data):
        """Test revenue analysis"""
        df = pd.DataFrame(financial_data)
        result = agent.analyze({
            'data': df,
            'query': 'Analyze revenue trends'
        })
        assert result is not None
    
    def test_profitability_metrics(self, agent, financial_data):
        """Test profitability calculations"""
        df = pd.DataFrame(financial_data)
        result = agent.analyze({
            'data': df,
            'query': 'Calculate profit margins'
        })
        assert result is not None
    
    def test_growth_analysis(self, agent, financial_data):
        """Test growth rate analysis"""
        df = pd.DataFrame(financial_data)
        result = agent.analyze({
            'data': df,
            'query': 'Calculate quarter over quarter growth'
        })
        assert result is not None


class TestTimeSeriesAgent:
    """Test Time Series Agent with real data"""
    
    @pytest.fixture
    def agent(self):
        return TimeSeriesAgent()
    
    @pytest.fixture
    def timeseries_data(self):
        """Load time series data"""
        data_path = Path(__file__).parent.parent.parent / "data" / "samples" / "sales_timeseries.json"
        if data_path.exists():
            with open(data_path, 'r') as f:
                data = json.load(f)
                return pd.DataFrame(data)
        
        # Generate sample time series
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'date': dates,
            'value': range(100, 200)
        })
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        assert agent is not None
        assert hasattr(agent, 'metadata')
        assert 'time series' in agent.metadata.name.lower()
    
    def test_trend_detection(self, agent, timeseries_data):
        """Test trend detection"""
        result = agent.analyze({
            'data': timeseries_data,
            'query': 'Detect trends in the time series'
        })
        assert result is not None
    
    def test_seasonality_analysis(self, agent, timeseries_data):
        """Test seasonality detection"""
        result = agent.analyze({
            'data': timeseries_data,
            'query': 'Analyze seasonal patterns'
        })
        assert result is not None
    
    def test_forecasting(self, agent, timeseries_data):
        """Test forecasting capabilities"""
        result = agent.analyze({
            'data': timeseries_data,
            'query': 'Forecast future values'
        })
        assert result is not None


class TestMLInsightsAgent:
    """Test ML Insights Agent"""
    
    @pytest.fixture
    def agent(self):
        return MLInsightsAgent()
    
    @pytest.fixture
    def ml_data(self):
        """Create ML-suitable data"""
        return pd.DataFrame({
            'feature1': range(100),
            'feature2': range(50, 150),
            'target': [0, 1] * 50
        })
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        assert agent is not None
        assert hasattr(agent, 'metadata')
    
    def test_clustering_analysis(self, agent, ml_data):
        """Test clustering analysis"""
        result = agent.analyze({
            'data': ml_data,
            'query': 'Perform clustering analysis'
        })
        assert result is not None
    
    def test_pattern_recognition(self, agent, ml_data):
        """Test pattern recognition"""
        result = agent.analyze({
            'data': ml_data,
            'query': 'Identify patterns in the data'
        })
        assert result is not None


class TestSQLAgent:
    """Test SQL Agent"""
    
    @pytest.fixture
    def agent(self):
        return SQLAgent()
    
    @pytest.fixture
    def structured_data(self):
        """Create structured data"""
        return pd.DataFrame({
            'customer_id': range(1, 11),
            'order_amount': [100, 200, 150, 300, 250, 180, 220, 190, 280, 310],
            'region': ['North', 'South'] * 5
        })
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        assert agent is not None
        assert hasattr(agent, 'metadata')
    
    def test_query_execution(self, agent, structured_data):
        """Test SQL-like query execution"""
        result = agent.analyze({
            'data': structured_data,
            'query': 'Calculate total order amount by region'
        })
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

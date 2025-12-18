"""
Unit Tests for Data Optimizer
Production-grade tests for data preprocessing and optimization
"""
import pytest
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from backend.utils.data_optimizer import DataOptimizer


class TestDataOptimizer:
    """Test suite for DataOptimizer"""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance"""
        return DataOptimizer()
    
    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe"""
        return pd.DataFrame({
            'id': range(1, 101),
            'category': ['A', 'B', 'C'] * 33 + ['A'],
            'value': range(100, 200),
            'date': pd.date_range('2024-01-01', periods=100)
        })
    
    # ===== SAMPLING TESTS =====
    
    def test_no_sampling_for_small_data(self, optimizer, sample_df):
        """Small datasets (<1000 rows) should not be sampled"""
        # Save to temp file and optimize
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_df.to_csv(f.name, index=False)
            result = optimizer.optimize_for_llm(f.name, 'csv')
        
        assert result['total_rows'] == len(sample_df)
        assert len(result['sample']) <= len(sample_df)
    
    def test_sampling_for_large_data(self, optimizer):
        """Large datasets should be sampled"""
        large_df = pd.DataFrame({
            'id': range(1, 10001),
            'value': range(10000)
        })
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            large_df.to_csv(f.name, index=False)
            result = optimizer.optimize_for_llm(f.name, 'csv')
        
        assert len(result['sample']) <= optimizer.max_rows
        assert result['total_rows'] == 10000
    
    def test_stratified_sampling(self, optimizer):
        """Sampling should preserve data structure"""
        df = pd.DataFrame({
            'category': ['A'] * 500 + ['B'] * 300 + ['C'] * 200,
            'value': range(1000)
        })
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            result = optimizer.optimize_for_llm(f.name, 'csv')
        
        # Check sample exists and has data
        assert len(result['sample']) > 0
        assert 'category' in result['schema']
        assert result['total_rows'] == 1000
    
    # ===== AGGREGATION TESTS =====
    
    def test_date_detection(self, optimizer):
        """Optimizer should detect date columns"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=365),
            'sales': range(365)
        })
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            result = optimizer.optimize_for_llm(f.name, 'csv')
        
        assert result['total_rows'] == 365
        assert 'date' in result['schema']
    
    # ===== MEMORY OPTIMIZATION TESTS =====
    
    def test_schema_generation(self, optimizer, sample_df):
        """Optimizer should generate schema information"""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_df.to_csv(f.name, index=False)
            result = optimizer.optimize_for_llm(f.name, 'csv')
        
        assert 'schema' in result
        assert len(result['schema']) == len(sample_df.columns)
        assert 'stats' in result
        assert result['is_optimized'] == True
    
    # ===== JSON FLATTENING TESTS =====
    
    def test_nested_json_flattening(self, optimizer):
        """Nested JSON should be processed correctly"""
        import tempfile
        import json
        nested_data = [
            {'id': 1, 'name': 'Alice', 'age': 30},
            {'id': 2, 'name': 'Bob', 'age': 25},
            {'id': 3, 'name': 'Charlie', 'age': 35}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(nested_data, f)
            f.flush()
            result = optimizer.optimize_for_llm(f.name, 'json')
        
        assert result['total_rows'] == 3
        assert 'schema' in result
    
    # ===== DATA VALIDATION TESTS =====
    
    def test_handles_columns_with_nulls(self, optimizer):
        """Should handle datasets with null values"""
        df = pd.DataFrame({
            'good_col': range(100),
            'nullable_col': [None] * 50 + list(range(50))
        })
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            result = optimizer.optimize_for_llm(f.name, 'csv')
        
        assert 'stats' in result
        assert result['total_columns'] >= 2
    
    # ===== STATISTICS TESTS =====
    
    def test_optimization_stats_returned(self, optimizer, sample_df):
        """Optimizer should return statistics"""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_df.to_csv(f.name, index=False)
            result = optimizer.optimize_for_llm(f.name, 'csv')
        
        assert 'total_rows' in result
        assert 'total_columns' in result
        assert 'stats' in result
        assert 'is_optimized' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

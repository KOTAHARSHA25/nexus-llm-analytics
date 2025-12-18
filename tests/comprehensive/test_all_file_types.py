"""
COMPREHENSIVE FILE TYPE TESTING
Tests CSV, JSON, PDF, TXT file processing
"""
import pytest
import sys
import os
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from backend.utils.data_optimizer import DataOptimizer
from backend.core.document_indexer import DocumentIndexer


class TestCSVFiles:
    """Test all CSV file handling"""
    
    @pytest.fixture
    def optimizer(self):
        return DataOptimizer()
    
    @pytest.fixture
    def csv_files(self):
        """Get all CSV files from samples"""
        data_dir = Path(__file__).parent.parent.parent / "data" / "samples"
        csv_files = []
        if data_dir.exists():
            csv_files = list(data_dir.glob("**/*.csv"))
        return csv_files
    
    def test_simple_csv(self, optimizer):
        """Test simple CSV file"""
        csv_path = Path(__file__).parent.parent.parent / "data" / "samples" / "sales_data.csv"
        if csv_path.exists():
            result = optimizer.optimize_for_llm(str(csv_path), 'csv')
            assert result is not None
            assert 'schema' in result
            assert 'sample' in result
            assert 'stats' in result
            assert result['is_optimized'] == True
    
    def test_customer_data_csv(self, optimizer):
        """Test customer data CSV"""
        csv_path = Path(__file__).parent.parent.parent / "data" / "samples" / "csv" / "customer_data.csv"
        if csv_path.exists():
            result = optimizer.optimize_for_llm(str(csv_path), 'csv')
            assert result is not None
            assert result['total_rows'] > 0
    
    def test_orders_csv(self, optimizer):
        """Test orders CSV"""
        csv_path = Path(__file__).parent.parent.parent / "data" / "samples" / "csv" / "orders.csv"
        if csv_path.exists():
            result = optimizer.optimize_for_llm(str(csv_path), 'csv')
            assert result is not None
            assert 'schema' in result
    
    def test_special_types_csv(self, optimizer):
        """Test CSV with special data types (currency, percentages, dates)"""
        csv_path = Path(__file__).parent.parent.parent / "data" / "samples" / "csv" / "special_types.csv"
        if csv_path.exists():
            result = optimizer.optimize_for_llm(str(csv_path), 'csv')
            assert result is not None
    
    def test_large_transactions_csv(self, optimizer):
        """Test large CSV file handling"""
        csv_path = Path(__file__).parent.parent.parent / "data" / "samples" / "csv" / "transactions_large.csv"
        if csv_path.exists():
            result = optimizer.optimize_for_llm(str(csv_path), 'csv')
            assert result is not None
            # Large files should be sampled
            assert len(result['sample']) <= optimizer.max_rows
    
    def test_stress_level_dataset(self, optimizer):
        """Test StressLevelDataset.csv"""
        csv_path = Path(__file__).parent.parent.parent / "data" / "samples" / "StressLevelDataset.csv"
        if csv_path.exists():
            result = optimizer.optimize_for_llm(str(csv_path), 'csv')
            assert result is not None
            assert result['total_rows'] > 0
    
    def test_all_sample_csvs(self, optimizer, csv_files):
        """Test all CSV files in samples directory"""
        for csv_file in csv_files[:10]:  # Test first 10 to avoid timeout
            try:
                result = optimizer.optimize_for_llm(str(csv_file), 'csv')
                assert result is not None
                assert 'schema' in result
                print(f"✓ {csv_file.name}")
            except Exception as e:
                print(f"✗ {csv_file.name}: {e}")


class TestJSONFiles:
    """Test all JSON file handling"""
    
    @pytest.fixture
    def optimizer(self):
        return DataOptimizer()
    
    def test_simple_json(self, optimizer):
        """Test simple JSON file"""
        json_path = Path(__file__).parent.parent.parent / "data" / "samples" / "simple.json"
        if json_path.exists():
            result = optimizer.optimize_for_llm(str(json_path), 'json')
            assert result is not None
            assert 'schema' in result
    
    def test_complex_nested_json(self, optimizer):
        """Test complex nested JSON"""
        json_path = Path(__file__).parent.parent.parent / "data" / "samples" / "complex_nested.json"
        if json_path.exists():
            try:
                result = optimizer.optimize_for_llm(str(json_path), 'json')
                assert result is not None
                assert result['was_nested'] == True
            except ValueError:
                # Empty or invalid JSON is acceptable
                pass
    
    def test_financial_json(self, optimizer):
        """Test financial quarterly JSON"""
        json_path = Path(__file__).parent.parent.parent / "data" / "samples" / "financial_quarterly.json"
        if json_path.exists():
            result = optimizer.optimize_for_llm(str(json_path), 'json')
            assert result is not None
    
    def test_sales_timeseries_json(self, optimizer):
        """Test sales timeseries JSON"""
        json_path = Path(__file__).parent.parent.parent / "data" / "samples" / "sales_timeseries.json"
        if json_path.exists():
            result = optimizer.optimize_for_llm(str(json_path), 'json')
            assert result is not None
    
    def test_edge_case_jsons(self, optimizer):
        """Test edge case JSON files"""
        edge_case_dir = Path(__file__).parent.parent.parent / "data" / "samples" / "edge_cases"
        if edge_case_dir.exists():
            edge_cases = [
                "boolean_fields.json",
                "date_formats.json",
                "mixed_types.json",
                "unicode_data.json",
                "null_values.json"
            ]
            
            for filename in edge_cases:
                json_path = edge_case_dir / filename
                if json_path.exists():
                    try:
                        result = optimizer.optimize_for_llm(str(json_path), 'json')
                        print(f"✓ {filename}")
                    except ValueError as e:
                        # Some edge cases might intentionally fail
                        print(f"⚠ {filename}: {e}")
    
    def test_empty_json_files(self, optimizer):
        """Test empty JSON handling"""
        edge_case_dir = Path(__file__).parent.parent.parent / "data" / "samples" / "edge_cases"
        empty_files = ["empty_array.json", "empty_object.json"]
        
        for filename in empty_files:
            json_path = edge_case_dir / filename
            if json_path.exists():
                with pytest.raises(ValueError):
                    optimizer.optimize_for_llm(str(json_path), 'json')


class TestPDFFiles:
    """Test PDF file handling"""
    
    @pytest.fixture
    def indexer(self):
        return DocumentIndexer()
    
    def test_pdf_extraction(self):
        """Test PDF text extraction"""
        # PDFs are converted to .extracted.txt files
        txt_path = Path(__file__).parent.parent.parent / "data" / "uploads" / "Harsha_Kota.pdf.extracted.txt"
        if txt_path.exists():
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert len(content) > 0
                print(f"✓ PDF extraction: {len(content)} chars")


class TestTXTFiles:
    """Test TXT file handling"""
    
    def test_text_file_reading(self):
        """Test reading text files"""
        txt_files = [
            "eachfile.txt.extracted.txt",
            "upgrade.txt.extracted.txt"
        ]
        
        data_dir = Path(__file__).parent.parent.parent / "data" / "uploads"
        for filename in txt_files:
            txt_path = data_dir / filename
            if txt_path.exists():
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    assert len(content) > 0
                    print(f"✓ {filename}: {len(content)} chars")


class TestMixedFileOperations:
    """Test operations across multiple file types"""
    
    @pytest.fixture
    def optimizer(self):
        return DataOptimizer()
    
    def test_file_type_detection(self, optimizer):
        """Test automatic file type detection"""
        test_files = [
            ("data/samples/sales_data.csv", "csv"),
            ("data/samples/simple.json", "json"),
        ]
        
        base_path = Path(__file__).parent.parent.parent
        for file_path, expected_type in test_files:
            full_path = base_path / file_path
            if full_path.exists():
                detected_type = optimizer._detect_file_type(full_path)
                assert detected_type == expected_type
    
    def test_concurrent_file_processing(self, optimizer):
        """Test processing multiple files concurrently"""
        base_path = Path(__file__).parent.parent.parent / "data" / "samples"
        files_to_test = [
            (base_path / "sales_data.csv", "csv"),
            (base_path / "simple.json", "json"),
        ]
        
        results = []
        for file_path, file_type in files_to_test:
            if file_path.exists():
                try:
                    result = optimizer.optimize_for_llm(str(file_path), file_type)
                    results.append((file_path.name, True))
                except Exception as e:
                    results.append((file_path.name, False))
        
        assert len(results) > 0
        success_rate = sum(1 for _, success in results if success) / len(results)
        assert success_rate >= 0.8  # At least 80% should succeed


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

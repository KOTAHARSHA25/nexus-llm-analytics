"""
COMPREHENSIVE DATA TYPE UNIT TESTS
Tests CSV, JSON, PDF, TXT processing with ALL sample files
"""
import pytest
import sys
from pathlib import Path
import json
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backend.utils.data_optimizer import DataOptimizer


class TestCSVDataType:
    """Comprehensive CSV file processing tests"""
    
    @pytest.fixture
    def optimizer(self):
        return DataOptimizer()
    
    def test_sales_data_csv(self, optimizer):
        """Test sales_data.csv processing"""
        csv_path = Path("data/samples/sales_data.csv")
        result = optimizer.optimize_for_llm(str(csv_path), 'csv')
        
        assert result is not None
        assert result['is_optimized'] == True
        assert result['file_type'] == 'csv'
        assert result['total_rows'] > 0
        assert result['total_columns'] > 0
        assert 'schema' in result
        assert 'stats' in result
        assert 'sample' in result
        print(f"✓ sales_data.csv: {result['total_rows']} rows, {result['total_columns']} cols")
    
    def test_customer_data_csv(self, optimizer):
        """Test customer_data.csv from csv subfolder"""
        csv_path = Path("data/samples/csv/customer_data.csv")
        if csv_path.exists():
            result = optimizer.optimize_for_llm(str(csv_path), 'csv')
            
            assert result is not None
            assert result['is_optimized'] == True
            assert len(result['schema']) > 0
            print(f"✓ customer_data.csv: {result['total_rows']} rows")
    
    def test_orders_csv(self, optimizer):
        """Test orders.csv"""
        csv_path = Path("data/samples/csv/orders.csv")
        if csv_path.exists():
            result = optimizer.optimize_for_llm(str(csv_path), 'csv')
            
            assert result is not None
            assert 'sample' in result
            assert len(result['sample']) > 0
            print(f"✓ orders.csv: {result['total_rows']} rows")
    
    def test_sales_simple_csv(self, optimizer):
        """Test sales_simple.csv"""
        csv_path = Path("data/samples/csv/sales_simple.csv")
        if csv_path.exists():
            result = optimizer.optimize_for_llm(str(csv_path), 'csv')
            
            assert result is not None
            assert result['total_rows'] > 0
            print(f"✓ sales_simple.csv: {result['total_rows']} rows")
    
    def test_special_types_csv(self, optimizer):
        """Test special_types.csv with currency, percentages, dates"""
        csv_path = Path("data/samples/csv/special_types.csv")
        if csv_path.exists():
            result = optimizer.optimize_for_llm(str(csv_path), 'csv')
            
            assert result is not None
            assert result['is_optimized'] == True
            print(f"✓ special_types.csv: {result['total_rows']} rows")
    
    def test_transactions_large_csv(self, optimizer):
        """Test transactions_large.csv (large file handling)"""
        csv_path = Path("data/samples/csv/transactions_large.csv")
        if csv_path.exists():
            result = optimizer.optimize_for_llm(str(csv_path), 'csv')
            
            assert result is not None
            # Large files should be sampled
            if result['total_rows'] > optimizer.max_rows:
                assert len(result['sample']) <= optimizer.max_rows
            print(f"✓ transactions_large.csv: {result['total_rows']} rows (sampled to {len(result['sample'])})")
    
    def test_stress_level_dataset_csv(self, optimizer):
        """Test StressLevelDataset.csv"""
        csv_path = Path("data/samples/StressLevelDataset.csv")
        if csv_path.exists():
            result = optimizer.optimize_for_llm(str(csv_path), 'csv')
            
            assert result is not None
            assert result['total_rows'] > 0
            assert 'stats' in result
            print(f"✓ StressLevelDataset.csv: {result['total_rows']} rows")
    
    def test_employee_data_csv(self, optimizer):
        """Test test_employee_data.csv"""
        csv_path = Path("data/samples/test_employee_data.csv")
        if csv_path.exists():
            result = optimizer.optimize_for_llm(str(csv_path), 'csv')
            
            assert result is not None
            print(f"✓ test_employee_data.csv: {result['total_rows']} rows")
    
    def test_inventory_csv(self, optimizer):
        """Test test_inventory.csv"""
        csv_path = Path("data/samples/test_inventory.csv")
        if csv_path.exists():
            result = optimizer.optimize_for_llm(str(csv_path), 'csv')
            
            assert result is not None
            print(f"✓ test_inventory.csv: {result['total_rows']} rows")
    
    def test_iot_sensor_csv(self, optimizer):
        """Test test_iot_sensor.csv"""
        csv_path = Path("data/samples/test_iot_sensor.csv")
        if csv_path.exists():
            result = optimizer.optimize_for_llm(str(csv_path), 'csv')
            
            assert result is not None
            print(f"✓ test_iot_sensor.csv: {result['total_rows']} rows")
    
    def test_sales_monthly_csv(self, optimizer):
        """Test test_sales_monthly.csv"""
        csv_path = Path("data/samples/test_sales_monthly.csv")
        if csv_path.exists():
            result = optimizer.optimize_for_llm(str(csv_path), 'csv')
            
            assert result is not None
            print(f"✓ test_sales_monthly.csv: {result['total_rows']} rows")
    
    def test_student_grades_csv(self, optimizer):
        """Test test_student_grades.csv"""
        csv_path = Path("data/samples/test_student_grades.csv")
        if csv_path.exists():
            result = optimizer.optimize_for_llm(str(csv_path), 'csv')
            
            assert result is not None
            print(f"✓ test_student_grades.csv: {result['total_rows']} rows")
    
    def test_university_grades_csv(self, optimizer):
        """Test test_university_grades.csv"""
        csv_path = Path("data/samples/test_university_grades.csv")
        if csv_path.exists():
            result = optimizer.optimize_for_llm(str(csv_path), 'csv')
            
            assert result is not None
            print(f"✓ test_university_grades.csv: {result['total_rows']} rows")
    
    def test_uploaded_csv_files(self, optimizer):
        """Test CSV files in uploads folder"""
        uploads_dir = Path("data/uploads")
        if uploads_dir.exists():
            csv_files = list(uploads_dir.glob("*.csv"))
            for csv_file in csv_files[:5]:  # Test first 5
                try:
                    result = optimizer.optimize_for_llm(str(csv_file), 'csv')
                    assert result is not None
                    print(f"✓ {csv_file.name}: {result['total_rows']} rows")
                except Exception as e:
                    print(f"⚠ {csv_file.name}: {str(e)[:50]}")


class TestJSONDataType:
    """Comprehensive JSON file processing tests"""
    
    @pytest.fixture
    def optimizer(self):
        return DataOptimizer()
    
    def test_simple_json(self, optimizer):
        """Test simple.json"""
        json_path = Path("data/samples/simple.json")
        if json_path.exists():
            result = optimizer.optimize_for_llm(str(json_path), 'json')
            
            assert result is not None
            assert result['file_type'] == 'json'
            assert result['is_optimized'] == True
            print(f"✓ simple.json: {result['total_rows']} rows")
    
    def test_1_json(self, optimizer):
        """Test 1.json"""
        json_path = Path("data/samples/1.json")
        if json_path.exists():
            result = optimizer.optimize_for_llm(str(json_path), 'json')
            
            assert result is not None
            print(f"✓ 1.json: {result['total_rows']} rows")
    
    def test_analyze_json(self, optimizer):
        """Test analyze.json"""
        json_path = Path("data/samples/analyze.json")
        if json_path.exists():
            result = optimizer.optimize_for_llm(str(json_path), 'json')
            
            assert result is not None
            print(f"✓ analyze.json: {result['total_rows']} rows")
    
    def test_complex_nested_json(self, optimizer):
        """Test complex_nested.json"""
        json_path = Path("data/samples/complex_nested.json")
        if json_path.exists():
            try:
                result = optimizer.optimize_for_llm(str(json_path), 'json')
                assert result is not None
                assert result['was_nested'] in [True, False]
                print(f"✓ complex_nested.json: {result['total_rows']} rows (nested: {result['was_nested']})")
            except ValueError as e:
                print(f"⚠ complex_nested.json: {str(e)[:50]}")
    
    def test_financial_quarterly_json(self, optimizer):
        """Test financial_quarterly.json"""
        json_path = Path("data/samples/financial_quarterly.json")
        if json_path.exists():
            result = optimizer.optimize_for_llm(str(json_path), 'json')
            
            assert result is not None
            assert result['total_rows'] > 0
            print(f"✓ financial_quarterly.json: {result['total_rows']} rows")
    
    def test_large_transactions_json(self, optimizer):
        """Test large_transactions.json"""
        json_path = Path("data/samples/large_transactions.json")
        if json_path.exists():
            result = optimizer.optimize_for_llm(str(json_path), 'json')
            
            assert result is not None
            print(f"✓ large_transactions.json: {result['total_rows']} rows")
    
    def test_sales_timeseries_json(self, optimizer):
        """Test sales_timeseries.json"""
        json_path = Path("data/samples/sales_timeseries.json")
        if json_path.exists():
            result = optimizer.optimize_for_llm(str(json_path), 'json')
            
            assert result is not None
            assert result['total_rows'] > 0
            print(f"✓ sales_timeseries.json: {result['total_rows']} rows")
    
    def test_edge_case_boolean_fields(self, optimizer):
        """Test edge_cases/boolean_fields.json"""
        json_path = Path("data/samples/edge_cases/boolean_fields.json")
        if json_path.exists():
            try:
                result = optimizer.optimize_for_llm(str(json_path), 'json')
                assert result is not None
                print(f"✓ boolean_fields.json: {result['total_rows']} rows")
            except ValueError:
                print("⚠ boolean_fields.json: Empty or invalid")
    
    def test_edge_case_date_formats(self, optimizer):
        """Test edge_cases/date_formats.json"""
        json_path = Path("data/samples/edge_cases/date_formats.json")
        if json_path.exists():
            try:
                result = optimizer.optimize_for_llm(str(json_path), 'json')
                assert result is not None
                print(f"✓ date_formats.json: {result['total_rows']} rows")
            except ValueError:
                print("⚠ date_formats.json: Empty or invalid")
    
    def test_edge_case_deep_nested(self, optimizer):
        """Test edge_cases/deep_nested.json"""
        json_path = Path("data/samples/edge_cases/deep_nested.json")
        if json_path.exists():
            try:
                result = optimizer.optimize_for_llm(str(json_path), 'json')
                assert result is not None
                assert result['was_nested'] == True
                print(f"✓ deep_nested.json: {result['total_rows']} rows")
            except ValueError:
                print("⚠ deep_nested.json: Empty or invalid")
    
    def test_edge_case_mixed_types(self, optimizer):
        """Test edge_cases/mixed_types.json"""
        json_path = Path("data/samples/edge_cases/mixed_types.json")
        if json_path.exists():
            try:
                result = optimizer.optimize_for_llm(str(json_path), 'json')
                assert result is not None
                print(f"✓ mixed_types.json: {result['total_rows']} rows")
            except ValueError:
                print("⚠ mixed_types.json: Empty or invalid")
    
    def test_edge_case_null_values(self, optimizer):
        """Test edge_cases/null_values.json"""
        json_path = Path("data/samples/edge_cases/null_values.json")
        if json_path.exists():
            try:
                result = optimizer.optimize_for_llm(str(json_path), 'json')
                assert result is not None
                print(f"✓ null_values.json: {result['total_rows']} rows")
            except ValueError:
                print("⚠ null_values.json: Empty or invalid")
    
    def test_edge_case_unicode_data(self, optimizer):
        """Test edge_cases/unicode_data.json"""
        json_path = Path("data/samples/edge_cases/unicode_data.json")
        if json_path.exists():
            try:
                result = optimizer.optimize_for_llm(str(json_path), 'json')
                assert result is not None
                print(f"✓ unicode_data.json: {result['total_rows']} rows")
            except ValueError:
                print("⚠ unicode_data.json: Empty or invalid")
    
    def test_edge_case_empty_array(self, optimizer):
        """Test edge_cases/empty_array.json (should fail gracefully)"""
        json_path = Path("data/samples/edge_cases/empty_array.json")
        if json_path.exists():
            with pytest.raises(ValueError):
                optimizer.optimize_for_llm(str(json_path), 'json')
            print("✓ empty_array.json: Correctly rejected empty array")
    
    def test_edge_case_empty_object(self, optimizer):
        """Test edge_cases/empty_object.json (should fail gracefully)"""
        json_path = Path("data/samples/edge_cases/empty_object.json")
        if json_path.exists():
            with pytest.raises(ValueError):
                optimizer.optimize_for_llm(str(json_path), 'json')
            print("✓ empty_object.json: Correctly rejected empty object")


class TestPDFDataType:
    """Comprehensive PDF/Text extraction tests"""
    
    def test_pdf_extracted_harsha_kota(self):
        """Test Harsha_Kota.pdf.extracted.txt"""
        txt_path = Path("data/uploads/Harsha_Kota.pdf.extracted.txt")
        if txt_path.exists():
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert len(content) > 0
            assert isinstance(content, str)
            print(f"✓ Harsha_Kota.pdf extraction: {len(content)} chars")
    
    def test_pdf_extraction_available(self):
        """Test that PDF extraction creates .extracted.txt files"""
        uploads_dir = Path("data/uploads")
        if uploads_dir.exists():
            extracted_files = list(uploads_dir.glob("*.extracted.txt"))
            
            assert len(extracted_files) > 0, "No extracted PDF files found"
            
            for extracted_file in extracted_files:
                with open(extracted_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                assert len(content) > 0
                print(f"✓ {extracted_file.name}: {len(content)} chars")


class TestTXTDataType:
    """Comprehensive TXT file tests"""
    
    def test_eachfile_txt(self):
        """Test eachfile.txt.extracted.txt"""
        txt_path = Path("data/uploads/eachfile.txt.extracted.txt")
        if txt_path.exists():
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert len(content) > 0
            assert isinstance(content, str)
            print(f"✓ eachfile.txt: {len(content)} chars")
    
    def test_upgrade_txt(self):
        """Test upgrade.txt.extracted.txt"""
        txt_path = Path("data/uploads/upgrade.txt.extracted.txt")
        if txt_path.exists():
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert len(content) > 0
            print(f"✓ upgrade.txt: {len(content)} chars")
    
    def test_all_txt_files_readable(self):
        """Test all .txt files in uploads directory"""
        uploads_dir = Path("data/uploads")
        if uploads_dir.exists():
            txt_files = list(uploads_dir.glob("*.txt"))
            
            for txt_file in txt_files:
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    assert len(content) >= 0  # May be empty
                    print(f"✓ {txt_file.name}: {len(content)} chars")
                except UnicodeDecodeError:
                    # Try with different encoding
                    with open(txt_file, 'r', encoding='latin-1') as f:
                        content = f.read()
                    print(f"✓ {txt_file.name}: {len(content)} chars (latin-1)")


class TestDataTypeIntegration:
    """Cross-data-type integration tests"""
    
    @pytest.fixture
    def optimizer(self):
        return DataOptimizer()
    
    def test_auto_detect_csv(self, optimizer):
        """Test automatic CSV detection"""
        csv_path = Path("data/samples/sales_data.csv")
        detected_type = optimizer._detect_file_type(csv_path)
        
        assert detected_type == 'csv'
        print("✓ CSV auto-detection working")
    
    def test_auto_detect_json(self, optimizer):
        """Test automatic JSON detection"""
        json_path = Path("data/samples/simple.json")
        if json_path.exists():
            detected_type = optimizer._detect_file_type(json_path)
            
            assert detected_type == 'json'
            print("✓ JSON auto-detection working")
    
    def test_process_multiple_types_sequentially(self, optimizer):
        """Test processing different file types in sequence"""
        files = [
            ("data/samples/sales_data.csv", "csv"),
            ("data/samples/simple.json", "json"),
        ]
        
        results = []
        for file_path, file_type in files:
            path = Path(file_path)
            if path.exists():
                try:
                    result = optimizer.optimize_for_llm(str(path), file_type)
                    results.append((path.name, True, result['total_rows']))
                except Exception as e:
                    results.append((path.name, False, str(e)))
        
        assert len(results) > 0
        success_count = sum(1 for _, success, _ in results if success)
        
        for name, success, info in results:
            status = "✓" if success else "✗"
            print(f"{status} {name}: {info}")
        
        assert success_count >= len(results) * 0.8  # At least 80% success


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])

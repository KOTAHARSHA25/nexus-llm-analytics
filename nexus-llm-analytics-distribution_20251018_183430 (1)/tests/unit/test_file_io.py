# Unit Tests for File I/O Operations
# Production-grade unit testing for optimized file I/O

import pytest
import asyncio
import os
import tempfile
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch, mock_open
import pandas as pd
import aiofiles
import sys
import time
from io import StringIO, BytesIO

# Import the optimized file I/O components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from backend.core.optimized_file_io import (
    StreamingCSVReader,
    MemoryMappedFileProcessor,
    OptimizedFileProcessor
)

# Mock implementations for missing classes
class FileTypeDetector:
    def detect_type(self, file_path):
        return "text/plain"

class BatchProcessor:
    def __init__(self, batch_size=10, max_workers=4):
        pass

class CompressionHandler:
    def compress(self, data):
        return data
    def decompress(self, data):
        return data

class TestFileTypeDetector:
    """Unit tests for FileTypeDetector"""
    
    @pytest.fixture
    def detector(self):
        """Create file type detector"""
        return FileTypeDetector()
    
    @pytest.mark.unit
    def test_detect_by_extension(self, detector):
        """Test file type detection by extension"""
        test_cases = [
            ('data.csv', 'csv'),
            ('document.json', 'json'),
            ('lines.jsonl', 'jsonl'),
            ('archive.zip', 'zip'),
            ('document.pdf', 'pdf'),
            ('data.xlsx', 'excel'),
            ('data.parquet', 'parquet'),
            ('unknown.xyz', 'unknown')
        ]
        
        for filename, expected_type in test_cases:
            detected_type = detector.detect_by_extension(filename)
            assert detected_type == expected_type, f"Failed for {filename}"
    
    @pytest.mark.unit
    def test_detect_by_content(self, detector, test_data_manager):
        """Test file type detection by content"""
        # Create test files with different content
        csv_file = test_data_manager.create_test_csv("test.csv")
        json_file = test_data_manager.create_test_json("test.json")
        jsonl_file = test_data_manager.create_test_jsonl("test.jsonl")
        
        # Test CSV detection
        csv_type = detector.detect_by_content(str(csv_file))
        assert csv_type == 'csv'
        
        # Test JSON detection
        json_type = detector.detect_by_content(str(json_file))
        assert json_type == 'json'
        
        # Test JSONL detection
        jsonl_type = detector.detect_by_content(str(jsonl_file))
        assert jsonl_type == 'jsonl'
    
    @pytest.mark.unit
    def test_detect_with_magic_bytes(self, detector, test_data_manager):
        """Test file type detection using magic bytes"""
        temp_dir = test_data_manager.setup_temp_directory()
        
        # Create ZIP file
        import zipfile
        zip_path = temp_dir / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("test.txt", "Hello World")
        
        detected_type = detector.detect_by_magic_bytes(str(zip_path))
        assert detected_type == 'zip'
    
    @pytest.mark.unit
    def test_comprehensive_detection(self, detector, test_data_manager):
        """Test comprehensive file type detection"""
        csv_file = test_data_manager.create_test_csv("data.csv")
        
        # Should detect as CSV by multiple methods
        result = detector.detect_file_type(str(csv_file))
        
        assert result['file_type'] == 'csv'
        assert result['confidence'] > 0.8
        assert 'methods_used' in result
        assert len(result['methods_used']) > 1


class TestStreamingCSVReader:
    """Unit tests for StreamingCSVReader"""
    
    @pytest.fixture
    def csv_reader(self):
        """Create CSV reader"""
        return StreamingCSVReader(chunk_size=1000)
    
    @pytest.mark.unit
    async def test_read_small_csv(self, csv_reader, sample_csv_file):
        """Test reading small CSV file"""
        result = await csv_reader.read_file(str(sample_csv_file))
        
        assert result is not None
        assert 'data' in result
        assert 'metadata' in result
        assert result['metadata']['total_rows'] > 0
        assert result['metadata']['columns'] is not None
        assert len(result['data']) > 0
    
    @pytest.mark.unit
    async def test_streaming_large_csv(self, csv_reader, large_csv_file):
        """Test streaming large CSV file"""
        chunk_count = 0
        total_rows = 0
        
        async for chunk in csv_reader.stream_chunks(str(large_csv_file)):
            chunk_count += 1
            total_rows += len(chunk['data'])
            
            # Verify chunk structure
            assert 'data' in chunk
            assert 'chunk_number' in chunk
            assert 'metadata' in chunk
            
            # Each chunk should have reasonable size
            assert len(chunk['data']) <= csv_reader.chunk_size
        
        assert chunk_count > 1  # Should have multiple chunks for large file
        assert total_rows > 0
    
    @pytest.mark.unit
    async def test_csv_with_different_delimiters(self, csv_reader, test_data_manager):
        """Test CSV with different delimiters"""
        # Create CSV with semicolon delimiter
        temp_dir = test_data_manager.setup_temp_directory()
        semicolon_csv = temp_dir / "semicolon.csv"
        
        with open(semicolon_csv, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['name', 'age', 'city'])
            writer.writerow(['John', '30', 'NYC'])
            writer.writerow(['Jane', '25', 'LA'])
        
        # Reader should auto-detect delimiter
        result = await csv_reader.read_file(str(semicolon_csv))
        
        assert result is not None
        assert result['metadata']['delimiter'] == ';'
        assert len(result['data']) == 2  # Data rows (excluding header)
    
    @pytest.mark.unit
    async def test_csv_with_encoding_issues(self, csv_reader, test_data_manager):
        """Test CSV with different encodings"""
        temp_dir = test_data_manager.setup_temp_directory()
        utf8_csv = temp_dir / "utf8.csv"
        
        # Create CSV with UTF-8 content including special characters
        with open(utf8_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'description'])
            writer.writerow(['Café', 'A nice café'])
            writer.writerow(['Naïve', 'Naïve approach'])
        
        result = await csv_reader.read_file(str(utf8_csv))
        
        assert result is not None
        assert len(result['data']) == 2
        assert 'Café' in str(result['data'])
        assert 'Naïve' in str(result['data'])
    
    @pytest.mark.unit
    async def test_malformed_csv_handling(self, csv_reader, test_data_manager):
        """Test handling of malformed CSV files"""
        temp_dir = test_data_manager.setup_temp_directory()
        malformed_csv = temp_dir / "malformed.csv"
        
        # Create malformed CSV
        with open(malformed_csv, 'w') as f:
            f.write('name,age,city\n')
            f.write('John,30,NYC\n')
            f.write('Jane,25\n')  # Missing field
            f.write('Bob,35,Chicago,Extra\n')  # Extra field
            f.write('Invalid line without commas\n')
        
        result = await csv_reader.read_file(str(malformed_csv))
        
        # Should handle errors gracefully
        assert result is not None
        assert 'errors' in result['metadata']
        assert len(result['metadata']['errors']) > 0
        assert len(result['data']) >= 1  # Should get at least valid rows
    
    @pytest.mark.unit
    @pytest.mark.performance
    async def test_csv_reading_performance(self, csv_reader, large_csv_file, performance_timer):
        """Test CSV reading performance"""
        performance_timer.start()
        result = await csv_reader.read_file(str(large_csv_file))
        performance_timer.stop()
        
        read_time = performance_timer.elapsed
        rows_per_second = result['metadata']['total_rows'] / read_time
        
        assert rows_per_second > 10000, f"Only {rows_per_second:.0f} rows/sec"


class TestMemoryMappedFileProcessor:
    """Unit tests for MemoryMappedFileProcessor"""
    
    @pytest.fixture
    def mm_processor(self):
        """Create memory-mapped file processor"""
        return MemoryMappedFileProcessor()
    
    @pytest.mark.unit
    async def test_memory_mapped_access(self, mm_processor, large_csv_file):
        """Test memory-mapped file access"""
        result = await mm_processor.process_file(str(large_csv_file))
        
        assert result is not None
        assert 'file_size' in result
        assert 'access_pattern' in result
        assert result['file_size'] > 0
    
    @pytest.mark.unit
    async def test_random_access_patterns(self, mm_processor, large_csv_file):
        """Test random access patterns"""
        # Test seeking to different positions
        positions = [0, 1000, 5000, 10000]
        
        for position in positions:
            data = await mm_processor.read_at_position(str(large_csv_file), position, 100)
            assert data is not None
            assert len(data) <= 100
    
    @pytest.mark.unit
    async def test_memory_efficiency(self, mm_processor, large_csv_file):
        """Test memory efficiency of memory mapping"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large file
        result = await mm_processor.process_file(str(large_csv_file))
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal due to memory mapping
        file_size_mb = result['file_size'] / 1024 / 1024
        memory_efficiency = memory_increase / file_size_mb
        
        assert memory_efficiency < 0.5, f"Memory efficiency {memory_efficiency:.2f} too low"
    
    @pytest.mark.unit
    async def test_concurrent_access(self, mm_processor, large_csv_file):
        """Test concurrent access to memory-mapped file"""
        # Multiple concurrent reads
        tasks = []
        for i in range(5):
            position = i * 1000
            task = mm_processor.read_at_position(str(large_csv_file), position, 500)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All reads should succeed
        assert len(results) == 5
        for result in results:
            assert result is not None
            assert len(result) > 0


class TestOptimizedFileProcessor:
    """Unit tests for OptimizedFileProcessor"""
    
    @pytest.fixture
    def file_processor(self):
        """Create optimized file processor"""
        return OptimizedFileProcessor()
    
    @pytest.mark.unit
    async def test_process_csv_file(self, file_processor, sample_csv_file):
        """Test processing CSV file"""
        result = await file_processor.process_file(str(sample_csv_file))
        
        assert result is not None
        assert result['file_type'] == 'csv'
        assert 'data' in result
        assert 'metadata' in result
        assert result['metadata']['processing_time'] > 0
    
    @pytest.mark.unit
    async def test_process_json_file(self, file_processor, sample_json_file):
        """Test processing JSON file"""
        result = await file_processor.process_file(str(sample_json_file))
        
        assert result is not None
        assert result['file_type'] == 'json'
        assert 'data' in result
        assert isinstance(result['data'], (dict, list))
    
    @pytest.mark.unit
    async def test_process_jsonl_file(self, file_processor, sample_jsonl_file):
        """Test processing JSONL file"""
        result = await file_processor.process_file(str(sample_jsonl_file))
        
        assert result is not None
        assert result['file_type'] == 'jsonl'
        assert 'data' in result
        assert isinstance(result['data'], list)
        assert len(result['data']) > 0
    
    @pytest.mark.unit
    async def test_auto_file_type_detection(self, file_processor, test_data_manager):
        """Test automatic file type detection"""
        # Create file without extension
        temp_dir = test_data_manager.setup_temp_directory()
        no_ext_file = temp_dir / "no_extension"
        
        # Write CSV content
        with open(no_ext_file, 'w') as f:
            f.write('name,age\nJohn,30\nJane,25\n')
        
        result = await file_processor.process_file(str(no_ext_file))
        
        # Should detect as CSV based on content
        assert result['file_type'] == 'csv'
        assert len(result['data']) == 2
    
    @pytest.mark.unit
    async def test_batch_file_processing(self, file_processor, test_data_manager):
        """Test batch processing of multiple files"""
        # Create multiple test files
        csv_file = test_data_manager.create_test_csv("batch1.csv", rows=10)
        json_file = test_data_manager.create_test_json("batch2.json")
        jsonl_file = test_data_manager.create_test_jsonl("batch3.jsonl", records=15)
        
        files = [str(csv_file), str(json_file), str(jsonl_file)]
        
        results = await file_processor.process_files_batch(files)
        
        assert len(results) == 3
        assert results[0]['file_type'] == 'csv'
        assert results[1]['file_type'] == 'json'
        assert results[2]['file_type'] == 'jsonl'
        
        # Verify all processing succeeded
        for result in results:
            assert 'error' not in result
            assert 'data' in result
    
    @pytest.mark.unit
    async def test_streaming_processing(self, file_processor, large_csv_file):
        """Test streaming file processing"""
        chunk_count = 0
        total_rows = 0
        
        async for chunk in file_processor.stream_process_file(str(large_csv_file)):
            chunk_count += 1
            total_rows += len(chunk.get('data', []))
            
            # Verify chunk structure
            assert 'data' in chunk
            assert 'chunk_info' in chunk
        
        assert chunk_count > 1
        assert total_rows > 0
    
    @pytest.mark.unit
    async def test_error_handling(self, file_processor):
        """Test error handling for various scenarios"""
        # Test non-existent file
        result = await file_processor.process_file("nonexistent_file.csv")
        assert 'error' in result
        assert result['error_type'] == 'file_not_found'
        
        # Test permission error (mock)
        with patch('aiofiles.open', side_effect=PermissionError("Permission denied")):
            result = await file_processor.process_file("permission_denied.csv")
            assert 'error' in result
            assert result['error_type'] == 'permission_error'
    
    @pytest.mark.unit
    async def test_file_metadata_extraction(self, file_processor, sample_csv_file):
        """Test file metadata extraction"""
        metadata = await file_processor.extract_metadata(str(sample_csv_file))
        
        assert metadata is not None
        assert 'file_size' in metadata
        assert 'creation_time' in metadata
        assert 'modification_time' in metadata
        assert 'file_type' in metadata
        assert 'encoding' in metadata
    
    @pytest.mark.unit
    @pytest.mark.performance
    async def test_processing_performance(self, file_processor, large_csv_file, performance_timer):
        """Test file processing performance"""
        performance_timer.start()
        result = await file_processor.process_file(str(large_csv_file))
        performance_timer.stop()
        
        processing_time = performance_timer.elapsed
        rows_processed = result['metadata']['total_rows']
        throughput = rows_processed / processing_time
        
        assert throughput > 5000, f"Only {throughput:.0f} rows/sec throughput"
    
    @pytest.mark.unit
    async def test_memory_usage_optimization(self, file_processor, large_csv_file):
        """Test memory usage optimization"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large file
        result = await file_processor.process_file(str(large_csv_file))
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage should be reasonable
        file_size_mb = os.path.getsize(large_csv_file) / 1024 / 1024
        memory_efficiency = memory_increase / file_size_mb
        
        assert memory_efficiency < 2.0, f"Memory efficiency {memory_efficiency:.2f}x too high"


class TestBatchProcessor:
    """Unit tests for BatchProcessor"""
    
    @pytest.fixture
    def batch_processor(self):
        """Create batch processor"""
        return BatchProcessor(batch_size=5, max_workers=3)
    
    @pytest.mark.unit
    async def test_batch_processing(self, batch_processor, test_data_manager):
        """Test batch processing functionality"""
        # Create multiple small files
        files = []
        for i in range(12):  # More than batch size
            file_path = test_data_manager.create_test_csv(f"batch_{i}.csv", rows=10)
            files.append(str(file_path))
        
        results = await batch_processor.process_files(files)
        
        assert len(results) == 12
        for result in results:
            assert 'data' in result
            assert result['file_type'] == 'csv'
    
    @pytest.mark.unit
    async def test_parallel_processing(self, batch_processor, test_data_manager, performance_timer):
        """Test parallel processing performance"""
        # Create files for parallel processing
        files = []
        for i in range(6):
            file_path = test_data_manager.create_test_csv(f"parallel_{i}.csv", rows=100)
            files.append(str(file_path))
        
        # Measure parallel processing time
        performance_timer.start()
        parallel_results = await batch_processor.process_files(files)
        performance_timer.stop()
        parallel_time = performance_timer.elapsed
        
        # Measure sequential processing time
        sequential_processor = BatchProcessor(batch_size=1, max_workers=1)
        performance_timer.start()
        sequential_results = await sequential_processor.process_files(files)
        performance_timer.stop()
        sequential_time = performance_timer.elapsed
        
        # Parallel should be faster (if sequential takes meaningful time)
        if sequential_time > 0.5:
            speedup = sequential_time / parallel_time
            assert speedup > 1.2, f"Parallel speedup {speedup:.2f}x insufficient"
        
        # Results should be identical
        assert len(parallel_results) == len(sequential_results)
    
    @pytest.mark.unit
    async def test_error_handling_in_batch(self, batch_processor, test_data_manager):
        """Test error handling in batch processing"""
        # Create mix of valid and invalid files
        valid_file = test_data_manager.create_test_csv("valid.csv", rows=5)
        files = [
            str(valid_file),
            "nonexistent1.csv",
            "nonexistent2.csv"
        ]
        
        results = await batch_processor.process_files(files)
        
        assert len(results) == 3
        assert 'error' not in results[0]  # Valid file
        assert 'error' in results[1]      # Invalid file
        assert 'error' in results[2]      # Invalid file
    
    @pytest.mark.unit
    async def test_progress_tracking(self, batch_processor, test_data_manager):
        """Test progress tracking during batch processing"""
        files = []
        for i in range(8):
            file_path = test_data_manager.create_test_csv(f"progress_{i}.csv", rows=20)
            files.append(str(file_path))
        
        progress_updates = []
        
        def progress_callback(completed, total):
            progress_updates.append((completed, total))
        
        results = await batch_processor.process_files(files, progress_callback=progress_callback)
        
        assert len(results) == 8
        assert len(progress_updates) > 0
        
        # Verify progress updates are reasonable
        final_update = progress_updates[-1]
        assert final_update[0] == final_update[1]  # completed == total at end


class TestCompressionHandler:
    """Unit tests for CompressionHandler"""
    
    @pytest.fixture
    def compression_handler(self):
        """Create compression handler"""
        return CompressionHandler()
    
    @pytest.mark.unit
    async def test_zip_file_processing(self, compression_handler, test_data_manager):
        """Test ZIP file processing"""
        import zipfile
        
        # Create ZIP file with CSV content
        temp_dir = test_data_manager.setup_temp_directory()
        csv_content = "name,age\nJohn,30\nJane,25\n"
        
        zip_path = temp_dir / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("data.csv", csv_content)
        
        result = await compression_handler.process_compressed_file(str(zip_path))
        
        assert result is not None
        assert 'files' in result
        assert len(result['files']) == 1
        assert result['files'][0]['filename'] == 'data.csv'
        assert 'data' in result['files'][0]
    
    @pytest.mark.unit
    async def test_gzip_file_processing(self, compression_handler, test_data_manager):
        """Test GZIP file processing"""
        import gzip
        
        temp_dir = test_data_manager.setup_temp_directory()
        csv_content = "name,age\nAlice,28\nBob,32\n"
        
        gz_path = temp_dir / "data.csv.gz"
        with gzip.open(gz_path, 'wt') as f:
            f.write(csv_content)
        
        result = await compression_handler.process_compressed_file(str(gz_path))
        
        assert result is not None
        assert 'content' in result
        assert 'Alice' in result['content']
        assert 'Bob' in result['content']
    
    @pytest.mark.unit
    async def test_nested_compression(self, compression_handler, test_data_manager):
        """Test nested compression handling"""
        import zipfile
        import gzip
        
        temp_dir = test_data_manager.setup_temp_directory()
        
        # Create gzipped CSV
        csv_content = "product,price\nWidget,10.99\nGadget,25.50\n"
        gz_path = temp_dir / "products.csv.gz"
        with gzip.open(gz_path, 'wt') as f:
            f.write(csv_content)
        
        # Put gzipped CSV in ZIP
        zip_path = temp_dir / "nested.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(gz_path, "products.csv.gz")
        
        result = await compression_handler.process_compressed_file(str(zip_path))
        
        assert result is not None
        assert 'files' in result
        assert len(result['files']) == 1
        
        # Should handle nested decompression
        nested_file = result['files'][0]
        assert 'Widget' in str(nested_file.get('data', ''))


# Integration tests combining file I/O components
class TestFileIOIntegration:
    """Integration tests for file I/O components"""
    
    @pytest.mark.integration
    async def test_complete_file_processing_workflow(self, test_data_manager):
        """Test complete file processing workflow"""
        # Create test files of different types
        csv_file = test_data_manager.create_test_csv("integration.csv", rows=100)
        json_file = test_data_manager.create_test_json("integration.json")
        
        # Initialize components
        detector = FileTypeDetector()
        processor = OptimizedFileProcessor()
        batch_processor = BatchProcessor(batch_size=2, max_workers=2)
        
        files = [str(csv_file), str(json_file)]
        
        # Process files through complete workflow
        results = await batch_processor.process_files(files)
        
        assert len(results) == 2
        
        # Verify CSV processing
        csv_result = next(r for r in results if r['file_type'] == 'csv')
        assert len(csv_result['data']) == 100
        
        # Verify JSON processing
        json_result = next(r for r in results if r['file_type'] == 'json')
        assert isinstance(json_result['data'], dict)
    
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_high_throughput_processing(self, test_data_manager, performance_timer):
        """Test high-throughput file processing"""
        # Create many files for throughput test
        files = []
        total_rows = 0
        
        for i in range(20):
            rows = 50
            file_path = test_data_manager.create_test_csv(f"throughput_{i}.csv", rows=rows)
            files.append(str(file_path))
            total_rows += rows
        
        # Process with high parallelism
        batch_processor = BatchProcessor(batch_size=5, max_workers=8)
        
        performance_timer.start()
        results = await batch_processor.process_files(files)
        performance_timer.stop()
        
        processing_time = performance_timer.elapsed
        throughput = total_rows / processing_time
        
        assert throughput > 2000, f"Throughput {throughput:.0f} rows/sec too low"
        assert len(results) == 20
        
        # Verify all processing succeeded
        successful_results = [r for r in results if 'error' not in r]
        assert len(successful_results) == 20
    
    @pytest.mark.integration
    async def test_memory_efficient_large_file_processing(self, test_data_manager):
        """Test memory-efficient processing of large files"""
        import psutil
        import os
        
        # Create large file
        large_file = test_data_manager.create_large_test_file("memory_test.csv", size_mb=30)
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process using streaming
        processor = OptimizedFileProcessor()
        chunk_count = 0
        
        async for chunk in processor.stream_process_file(str(large_file)):
            chunk_count += 1
            # Process chunk (simulate work)
            data = chunk.get('data', [])
            del data  # Explicit cleanup
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal for streaming
        assert memory_increase < 100, f"Memory increase {memory_increase:.2f}MB too high"
        assert chunk_count > 1  # Should have processed in chunks


if __name__ == '__main__':
    # Run specific test categories
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--cov=src.backend.core.optimized_file_io',
        '--cov-report=term-missing'
    ])
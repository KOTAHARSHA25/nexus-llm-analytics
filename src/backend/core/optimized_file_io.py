# Optimized File I/O Operations with Streaming and Async Processing
# BEFORE: Synchronous file processing, memory inefficient, blocking operations
# AFTER: Streaming I/O, memory-mapped files, async processing, chunked operations

import asyncio
import aiofiles
import mmap
import os
import json
import csv
import io
import time
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator, Union, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import weakref
import hashlib
import gzip
import zlib
from enum import Enum
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
import struct
import pickle

logger = logging.getLogger(__name__)

class FileType(Enum):
    """Supported file types with optimized processing"""
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    PDF = "pdf"
    TXT = "txt"
    PARQUET = "parquet"
    EXCEL = "excel"
    XML = "xml"

@dataclass
class FileProcessingConfig:
    """Configuration for optimized file processing"""
    chunk_size: int = 8192  # 8KB chunks for streaming
    max_memory_mb: int = 100  # Maximum memory usage
    parallel_workers: int = 4
    use_memory_mapping: bool = True
    enable_compression: bool = True
    cache_parsed_data: bool = True
    streaming_threshold_mb: int = 10  # Use streaming for files > 10MB

@dataclass
class FileMetadata:
    """Enhanced file metadata with processing information"""
    path: str
    size_bytes: int
    file_type: FileType
    encoding: str = "utf-8"
    compression: Optional[str] = None
    last_modified: float = 0.0
    checksum: Optional[str] = None
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    processing_time: float = 0.0
    memory_usage: int = 0
    
    def __post_init__(self):
        if self.last_modified == 0.0:
            try:
                self.last_modified = os.path.getmtime(self.path)
            except Exception:
                self.last_modified = time.time()

class StreamingCSVReader:
    """High-performance streaming CSV reader with memory optimization"""
    
    def __init__(self, file_path: str, chunk_size: int = 8192):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.headers = []
        self.total_rows = 0
        self._buffer = ""
        
    async def __aenter__(self):
        self.file = await aiofiles.open(self.file_path, 'r', encoding='utf-8')
        # Read headers
        first_line = await self.file.readline()
        if first_line:
            self.headers = [h.strip().strip('"') for h in first_line.strip().split(',')]
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'file'):
            await self.file.close()
    
    async def read_chunks(self, max_rows_per_chunk: int = 1000) -> AsyncGenerator[List[Dict[str, str]], None]:
        """Stream CSV data in optimized chunks"""
        current_chunk = []
        row_count = 0
        
        try:
            while True:
                # Read chunk from file
                chunk_data = await self.file.read(self.chunk_size)
                if not chunk_data:
                    break
                
                self._buffer += chunk_data
                
                # Process complete lines in buffer
                while '\n' in self._buffer:
                    line, self._buffer = self._buffer.split('\n', 1)
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    # Parse CSV row efficiently
                    try:
                        values = [v.strip().strip('"') for v in line.split(',')]
                        if len(values) == len(self.headers):
                            row_dict = dict(zip(self.headers, values))
                            current_chunk.append(row_dict)
                            row_count += 1
                            
                            # Yield chunk when full
                            if len(current_chunk) >= max_rows_per_chunk:
                                yield current_chunk
                                current_chunk = []
                        
                    except Exception as e:
                        logger.warning(f"Failed to parse CSV row: {e}")
                        continue
            
            # Yield remaining data
            if current_chunk:
                yield current_chunk
                
            self.total_rows = row_count
            
        except Exception as e:
            logger.error(f"Error streaming CSV: {e}")
            raise

class StreamingJSONReader:
    """High-performance streaming JSON/JSONL reader"""
    
    def __init__(self, file_path: str, chunk_size: int = 8192):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.is_jsonl = file_path.endswith('.jsonl')
        
    async def __aenter__(self):
        self.file = await aiofiles.open(self.file_path, 'r', encoding='utf-8')
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'file'):
            await self.file.close()
    
    async def read_chunks(self, max_items_per_chunk: int = 100) -> AsyncGenerator[List[Dict[str, Any]], None]:
        """Stream JSON data in optimized chunks"""
        if self.is_jsonl:
            async for chunk in self._read_jsonl_chunks(max_items_per_chunk):
                yield chunk
        else:
            async for chunk in self._read_json_chunks(max_items_per_chunk):
                yield chunk
    
    async def _read_jsonl_chunks(self, max_items_per_chunk: int) -> AsyncGenerator[List[Dict[str, Any]], None]:
        """Stream JSONL (newline-delimited JSON) efficiently"""
        current_chunk = []
        
        async for line in self.file:
            line = line.strip()
            if not line:
                continue
                
            try:
                json_obj = json.loads(line)
                current_chunk.append(json_obj)
                
                if len(current_chunk) >= max_items_per_chunk:
                    yield current_chunk
                    current_chunk = []
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON line: {e}")
                continue
        
        if current_chunk:
            yield current_chunk
    
    async def _read_json_chunks(self, max_items_per_chunk: int) -> AsyncGenerator[List[Dict[str, Any]], None]:
        """Stream regular JSON with array chunking"""
        content = ""
        buffer_size = self.chunk_size * 4  # Larger buffer for JSON
        
        # Read entire file for JSON parsing (optimize for smaller files)
        try:
            content = await self.file.read()
            data = json.loads(content)
            
            if isinstance(data, list):
                # Chunk array data
                for i in range(0, len(data), max_items_per_chunk):
                    yield data[i:i + max_items_per_chunk]
            else:
                # Single object
                yield [data]
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file: {e}")
            raise
        except MemoryError:
            logger.error("JSON file too large for memory - consider JSONL format")
            raise

class MemoryMappedFileProcessor:
    """Memory-mapped file processing for large files"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file = None
        self.mmap_obj = None
        
    async def __aenter__(self):
        self.file = open(self.file_path, 'rb')
        self.mmap_obj = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.mmap_obj:
            self.mmap_obj.close()
        if self.file:
            self.file.close()
    
    async def read_chunks(self, chunk_size: int = 8192) -> AsyncGenerator[bytes, None]:
        """Read file in memory-mapped chunks"""
        if not self.mmap_obj:
            return
        
        position = 0
        file_size = len(self.mmap_obj)
        
        while position < file_size:
            end_pos = min(position + chunk_size, file_size)
            chunk = self.mmap_obj[position:end_pos]
            yield chunk
            position = end_pos
    
    async def find_pattern(self, pattern: bytes) -> List[int]:
        """Efficiently find pattern positions in memory-mapped file"""
        positions = []
        if not self.mmap_obj:
            return positions
        
        start = 0
        while True:
            pos = self.mmap_obj.find(pattern, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        
        return positions

class OptimizedFileProcessor:
    """High-performance file processor with multiple optimization strategies"""
    
    def __init__(self, config: FileProcessingConfig = None):
        self.config = config or FileProcessingConfig()
        self.thread_executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max(1, self.config.parallel_workers // 2))
        self.file_cache = {}
        self.processing_stats = defaultdict(list)
        
    async def process_file(self, file_path: str, file_type: FileType = None) -> Dict[str, Any]:
        """Process file with optimal strategy based on size and type"""
        start_time = time.time()
        
        # Get file metadata
        metadata = await self._get_file_metadata(file_path, file_type)
        
        # Choose processing strategy based on file size
        if metadata.size_bytes > self.config.streaming_threshold_mb * 1024 * 1024:
            result = await self._process_large_file_streaming(metadata)
        else:
            result = await self._process_small_file_memory(metadata)
        
        # Update processing stats
        processing_time = time.time() - start_time
        self.processing_stats[metadata.file_type].append({
            'size_mb': metadata.size_bytes / 1024 / 1024,
            'processing_time': processing_time,
            'rows_processed': result.get('row_count', 0)
        })
        
        result['metadata'] = metadata
        result['processing_time'] = processing_time
        
        return result
    
    async def _get_file_metadata(self, file_path: str, file_type: FileType = None) -> FileMetadata:
        """Get comprehensive file metadata efficiently"""
        path_obj = Path(file_path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type
        if file_type is None:
            file_type = self._detect_file_type(file_path)
        
        # Get basic metadata
        stat = path_obj.stat()
        
        # Calculate checksum for caching
        checksum = await self._calculate_file_checksum(file_path)
        
        metadata = FileMetadata(
            path=str(path_obj),
            size_bytes=stat.st_size,
            file_type=file_type,
            last_modified=stat.st_mtime,
            checksum=checksum
        )
        
        return metadata
    
    def _detect_file_type(self, file_path: str) -> FileType:
        """Detect file type from extension and content"""
        suffix = Path(file_path).suffix.lower()
        
        type_mapping = {
            '.csv': FileType.CSV,
            '.json': FileType.JSON,
            '.jsonl': FileType.JSONL,
            '.pdf': FileType.PDF,
            '.txt': FileType.TXT,
            '.parquet': FileType.PARQUET,
            '.xlsx': FileType.EXCEL,
            '.xls': FileType.EXCEL,
            '.xml': FileType.XML
        }
        
        return type_mapping.get(suffix, FileType.TXT)
    
    async def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate file checksum efficiently"""
        hash_obj = hashlib.sha256()
        
        async with aiofiles.open(file_path, 'rb') as f:
            while True:
                chunk = await f.read(8192)
                if not chunk:
                    break
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()[:16]
    
    async def _process_large_file_streaming(self, metadata: FileMetadata) -> Dict[str, Any]:
        """Process large files using streaming approach"""
        logger.info(f"Processing large file with streaming: {metadata.path}")
        
        if metadata.file_type == FileType.CSV:
            return await self._stream_process_csv(metadata)
        elif metadata.file_type in [FileType.JSON, FileType.JSONL]:
            return await self._stream_process_json(metadata)
        elif metadata.file_type == FileType.TXT:
            return await self._stream_process_text(metadata)
        else:
            return await self._process_with_memory_mapping(metadata)
    
    async def _process_small_file_memory(self, metadata: FileMetadata) -> Dict[str, Any]:
        """Process small files in memory for speed"""
        logger.info(f"Processing small file in memory: {metadata.path}")
        
        if metadata.file_type == FileType.CSV:
            return await self._memory_process_csv(metadata)
        elif metadata.file_type in [FileType.JSON, FileType.JSONL]:
            return await self._memory_process_json(metadata)
        else:
            return await self._memory_process_generic(metadata)
    
    async def _stream_process_csv(self, metadata: FileMetadata) -> Dict[str, Any]:
        """Stream process CSV file efficiently"""
        data_chunks = []
        total_rows = 0
        
        async with StreamingCSVReader(metadata.path, self.config.chunk_size) as reader:
            async for chunk in reader.read_chunks():
                # Process chunk (e.g., data validation, transformation)
                processed_chunk = await self._process_csv_chunk(chunk)
                data_chunks.append(processed_chunk)
                total_rows += len(chunk)
                
                # Memory management
                if len(data_chunks) > 10:  # Keep only recent chunks
                    data_chunks.pop(0)
        
        metadata.columns = reader.headers
        metadata.row_count = total_rows
        
        return {
            'file_type': 'csv',
            'columns': reader.headers,
            'row_count': total_rows,
            'sample_data': data_chunks[-1] if data_chunks else [],
            'streaming': True
        }
    
    async def _stream_process_json(self, metadata: FileMetadata) -> Dict[str, Any]:
        """Stream process JSON/JSONL file efficiently"""
        data_chunks = []
        total_items = 0
        
        async with StreamingJSONReader(metadata.path, self.config.chunk_size) as reader:
            async for chunk in reader.read_chunks():
                processed_chunk = await self._process_json_chunk(chunk)
                data_chunks.append(processed_chunk)
                total_items += len(chunk)
                
                # Memory management
                if len(data_chunks) > 10:
                    data_chunks.pop(0)
        
        metadata.row_count = total_items
        
        return {
            'file_type': 'json',
            'item_count': total_items,
            'sample_data': data_chunks[-1] if data_chunks else [],
            'streaming': True
        }
    
    async def _stream_process_text(self, metadata: FileMetadata) -> Dict[str, Any]:
        """Stream process text file efficiently"""
        line_count = 0
        char_count = 0
        word_count = 0
        
        async with aiofiles.open(metadata.path, 'r', encoding='utf-8') as f:
            async for line in f:
                line_count += 1
                char_count += len(line)
                word_count += len(line.split())
        
        metadata.row_count = line_count
        
        return {
            'file_type': 'text',
            'line_count': line_count,
            'char_count': char_count,
            'word_count': word_count,
            'streaming': True
        }
    
    async def _process_with_memory_mapping(self, metadata: FileMetadata) -> Dict[str, Any]:
        """Process file using memory mapping"""
        logger.info(f"Processing with memory mapping: {metadata.path}")
        
        async with MemoryMappedFileProcessor(metadata.path) as processor:
            chunk_count = 0
            total_bytes = 0
            
            async for chunk in processor.read_chunks(self.config.chunk_size):
                chunk_count += 1
                total_bytes += len(chunk)
        
        return {
            'file_type': str(metadata.file_type.value),
            'chunks_processed': chunk_count,
            'total_bytes': total_bytes,
            'memory_mapped': True
        }
    
    async def _memory_process_csv(self, metadata: FileMetadata) -> Dict[str, Any]:
        """Process CSV file in memory using pandas for small files"""
        try:
            # Use pandas for efficient CSV processing
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                self.thread_executor,
                lambda: pd.read_csv(metadata.path, encoding='utf-8')
            )
            
            metadata.columns = df.columns.tolist()
            metadata.row_count = len(df)
            
            # Get sample data (first 10 rows)
            sample_data = df.head(10).to_dict('records')
            
            return {
                'file_type': 'csv',
                'columns': metadata.columns,
                'row_count': metadata.row_count,
                'sample_data': sample_data,
                'data_types': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'streaming': False
            }
            
        except Exception as e:
            logger.error(f"Error processing CSV with pandas: {e}")
            # Fallback to streaming approach
            return await self._stream_process_csv(metadata)
    
    async def _memory_process_json(self, metadata: FileMetadata) -> Dict[str, Any]:
        """Process JSON file in memory"""
        try:
            async with aiofiles.open(metadata.path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
            
            if isinstance(data, list):
                metadata.row_count = len(data)
                sample_data = data[:10]  # First 10 items
            else:
                metadata.row_count = 1
                sample_data = [data]
            
            return {
                'file_type': 'json',
                'item_count': metadata.row_count,
                'sample_data': sample_data,
                'streaming': False
            }
            
        except Exception as e:
            logger.error(f"Error processing JSON in memory: {e}")
            return await self._stream_process_json(metadata)
    
    async def _memory_process_generic(self, metadata: FileMetadata) -> Dict[str, Any]:
        """Generic memory processing for other file types"""
        try:
            async with aiofiles.open(metadata.path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            return {
                'file_type': str(metadata.file_type.value),
                'content_length': len(content),
                'sample_content': content[:1000],  # First 1000 chars
                'streaming': False
            }
            
        except UnicodeDecodeError:
            # Handle binary files
            async with aiofiles.open(metadata.path, 'rb') as f:
                content = await f.read()
            
            return {
                'file_type': str(metadata.file_type.value),
                'content_length': len(content),
                'binary': True,
                'streaming': False
            }
    
    async def _process_csv_chunk(self, chunk: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process CSV chunk with optimizations"""
        # Apply data type inference and validation
        processed_chunk = []
        
        for row in chunk:
            processed_row = {}
            for key, value in row.items():
                # Attempt type conversion
                processed_value = self._infer_and_convert_type(value)
                processed_row[key] = processed_value
            
            processed_chunk.append(processed_row)
        
        return processed_chunk
    
    async def _process_json_chunk(self, chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process JSON chunk with optimizations"""
        # JSON data is already typed, just validate and clean
        return [self._clean_json_object(obj) for obj in chunk]
    
    def _infer_and_convert_type(self, value: str) -> Any:
        """Infer and convert string value to appropriate type"""
        if not value or value.lower() in ['null', 'none', '']:
            return None
        
        # Try integer
        try:
            if '.' not in value and value.isdigit():
                return int(value)
        except:
            logging.debug("Operation failed (non-critical) - continuing")
        
        # Try float
        try:
            return float(value)
        except:
            logging.debug("Operation failed (non-critical) - continuing")
        
        # Try boolean
        if value.lower() in ['true', 'yes', '1']:
            return True
        elif value.lower() in ['false', 'no', '0']:
            return False
        
        # Return as string
        return value.strip()
    
    def _clean_json_object(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate JSON object"""
        cleaned = {}
        for key, value in obj.items():
            # Clean key
            clean_key = str(key).strip()
            
            # Clean value
            if isinstance(value, str):
                clean_value = value.strip()
            else:
                clean_value = value
            
            cleaned[clean_key] = clean_value
        
        return cleaned
    
    async def process_multiple_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process multiple files in parallel"""
        start_time = time.time()
        
        # Create tasks for parallel processing
        tasks = []
        for file_path in file_paths:
            task = asyncio.create_task(self.process_file(file_path))
            tasks.append((file_path, task))
        
        # Process files in parallel
        results = {}
        errors = {}
        
        for file_path, task in tasks:
            try:
                result = await task
                results[file_path] = result
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                errors[file_path] = str(e)
        
        total_time = time.time() - start_time
        
        return {
            'results': results,
            'errors': errors,
            'summary': {
                'total_files': len(file_paths),
                'successful': len(results),
                'failed': len(errors),
                'total_processing_time': total_time
            }
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        stats = {}
        
        for file_type, measurements in self.processing_stats.items():
            if not measurements:
                continue
            
            sizes = [m['size_mb'] for m in measurements]
            times = [m['processing_time'] for m in measurements]
            rows = [m['rows_processed'] for m in measurements if m['rows_processed'] > 0]
            
            stats[file_type.value] = {
                'files_processed': len(measurements),
                'avg_size_mb': sum(sizes) / len(sizes),
                'avg_processing_time': sum(times) / len(times),
                'avg_rows_processed': sum(rows) / len(rows) if rows else 0,
                'throughput_mb_per_sec': sum(sizes) / sum(times) if sum(times) > 0 else 0,
                'throughput_rows_per_sec': sum(rows) / sum(times) if sum(times) > 0 and rows else 0
            }
        
        return stats
    
    async def close(self):
        """Clean up resources"""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)

# Factory for creating optimized file processors
class FileProcessorFactory:
    """Factory for creating optimized file processors"""
    
    _instances = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_processor(cls, config: FileProcessingConfig = None) -> OptimizedFileProcessor:
        """Get or create optimized file processor"""
        config_key = str(config) if config else "default"
        
        if config_key not in cls._instances:
            with cls._lock:
                if config_key not in cls._instances:
                    cls._instances[config_key] = OptimizedFileProcessor(config)
        
        return cls._instances[config_key]
    
    @classmethod
    async def close_all_processors(cls):
        """Close all processor instances"""
        for processor in cls._instances.values():
            await processor.close()
        cls._instances.clear()

# Utility functions for common file operations
async def quick_file_analysis(file_path: str) -> Dict[str, Any]:
    """Quick file analysis with optimized processing"""
    processor = FileProcessorFactory.get_processor()
    return await processor.process_file(file_path)

async def batch_file_analysis(file_paths: List[str]) -> Dict[str, Any]:
    """Batch file analysis with parallel processing"""
    processor = FileProcessorFactory.get_processor()
    return await processor.process_multiple_files(file_paths)

@asynccontextmanager
async def optimized_file_processor(config: FileProcessingConfig = None):
    """Context manager for file processor with automatic cleanup"""
    processor = FileProcessorFactory.get_processor(config)
    try:
        yield processor
    finally:
        # Don't close here as it's a shared instance
        pass
import pytest
import asyncio
from unittest.mock import MagicMock, patch, mock_open, AsyncMock
from src.backend.core.document_indexer import SemanticChunker, OptimizedDocumentIndexer, ChunkType, SemanticChunk

# --- SemanticChunker Tests ---

@pytest.fixture
def chunker():
    return SemanticChunker(max_chunk_size=10, min_chunk_size=2, overlap_sentences=0)

def test_semantic_chunker_basic(chunker):
    text = "Hello world. This is a test."
    chunks = chunker.chunk(text)
    # With max_chunk_size=10, this might be one or two chunks
    assert len(chunks) > 0
    assert chunks[0].chunk_type == ChunkType.PARAGRAPH

def test_semantic_chunker_sections(chunker):
    # Two sections separated by double newline
    text = "# Header 1\nContent under header.\n\n# Header 2\nMore content."
    semantic_units = chunker._split_into_semantic_units(text)
    # Split by \n\n -> 2 parts, each starts with # so both are 'section' type
    assert len(semantic_units) == 2
    # Check if headers detected
    headers = [u for u in semantic_units if u['type'] == 'section']
    assert len(headers) == 2

def test_extract_code_blocks(chunker):
    text = "Here is code:\n```python\nprint('hello')\n```\nEnd."
    chunks = chunker._extract_code_blocks(text)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == ChunkType.CODE_BLOCK
    assert chunks[0].metadata['language'] == 'python'
    assert "print('hello')" in chunks[0].text

def test_extract_tables(chunker):
    text = "Table:\n| col1 | col2 |\n| val1 | val2 |\nEnd"
    chunks = chunker._extract_tables(text)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == ChunkType.TABLE
    assert "| col1 | col2 |" in chunks[0].text

def test_split_by_sentences(chunker):
    # max_chunk_size is 10 words.
    text = "One two three four five. " * 3 # 15 words
    chunks = chunker._split_by_sentences(text, 0)
    assert len(chunks) >= 2 # Should split

# --- OptimizedDocumentIndexer Tests ---

@pytest.fixture
def mock_chroma_client():
    return MagicMock()

@pytest.fixture
def indexer(mock_chroma_client):
    with patch('src.backend.core.document_indexer.embed_text') as mock_embed:
        # Mock embed_text globally for the module
        mock_embed.return_value = [0.1, 0.2, 0.3]
        idx = OptimizedDocumentIndexer(chroma_client=mock_chroma_client, max_workers=1)
        yield idx

@pytest.mark.asyncio
async def test_index_document_async_semantic(indexer, mock_chroma_client):
    content = "Test content for indexing."
    filename = "test.txt"
    
    # Mock embed (it's called via run_in_executor, waiting for it might be tricky with standard mocks if not careful)
    # The indexer uses self.executor. We can mock that or the function it calls.
    # We patched embed_text in the fixture.
    
    # We need to mock _batch_embed_chunks to avoid the executor complexity in unit test
    with patch.object(indexer, '_batch_embed_chunks', new_callable=AsyncMock) as mock_batch:
        mock_batch.return_value = [[0.1]*384] # Return 1 embedding
        
        result = await indexer.index_document_async(filename, content)
        
        assert result["status"] == "success"
        assert result["chunking_method"] == "semantic"
        mock_chroma_client.add_document.assert_called()

@pytest.mark.asyncio
async def test_duplicate_document(indexer):
    content = "Same content"
    
    res1 = await indexer.index_document_async("f1", content)
    res2 = await indexer.index_document_async("f2", content)
    
    assert res1["status"] == "success"
    assert res2["status"] == "skipped"
    assert res2["reason"] == "duplicate_document"

@pytest.mark.asyncio
async def test_optimized_chunking_fallback(indexer, mock_chroma_client):
    indexer.use_semantic_chunking = False
    content = "Fallback content " * 50
    
    with patch.object(indexer, '_batch_embed_chunks', new_callable=AsyncMock) as mock_batch:
        mock_batch.return_value = [[0.1]*384] * 5
        
        result = await indexer.index_document_async("fallback.txt", content)
        assert result["chunking_method"] == "optimized"

def test_index_extracted_documents(indexer):
    with patch('src.backend.core.document_indexer.Path') as mock_path:
        # Mock glob
        mock_file = MagicMock()
        mock_file.name = "test.extracted.txt"
        mock_path.return_value.glob.return_value = [mock_file]
        
        # Mock open
        with patch('builtins.open', mock_open(read_data="Extracted Content")):
             with patch.object(indexer, 'index_document_async', new_callable=AsyncMock) as mock_idx:
                mock_idx.return_value = {"status": "success", "successful_chunks": 5}
                
                result = indexer.index_extracted_documents()
                
                assert result["status"] == "completed"
                assert result["successful_files"] == 1

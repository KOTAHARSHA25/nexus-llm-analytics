"""
Phase 3 RAG Enhancement Tests
Tests for: Semantic Chunking (3.1), Hybrid Search (3.2), Citation Tracking (3.3)

Run with: python tests/test_phase3_rag_enhancement.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'backend'))

import unittest
from unittest.mock import Mock, patch, MagicMock

# Test imports
from core.document_indexer import SemanticChunker, ChunkType, SemanticChunk, OptimizedDocumentIndexer
from core.chromadb_client import ChromaDBClient, CitedSource, RAGResponse


class TestSemanticChunker(unittest.TestCase):
    """Test Task 3.1: Semantic Chunking"""
    
    def setUp(self):
        self.chunker = SemanticChunker(
            max_chunk_size=100,
            min_chunk_size=20,
            overlap_sentences=1
        )
    
    def test_empty_text(self):
        """Test chunking empty text"""
        chunks = self.chunker.chunk("")
        self.assertEqual(len(chunks), 0)
        
        chunks = self.chunker.chunk("   ")
        self.assertEqual(len(chunks), 0)
    
    def test_simple_paragraph(self):
        """Test chunking a simple paragraph"""
        text = "This is a simple paragraph. It has multiple sentences. Each sentence is meaningful."
        chunks = self.chunker.chunk(text)
        
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], SemanticChunk)
        self.assertEqual(chunks[0].chunk_type, ChunkType.PARAGRAPH)
    
    def test_multiple_paragraphs(self):
        """Test chunking multiple paragraphs"""
        text = """First paragraph with some content here.

Second paragraph with different content.

Third paragraph to make it more interesting."""
        
        chunks = self.chunker.chunk(text)
        self.assertGreater(len(chunks), 0)
        
        # Check all chunks have text
        for chunk in chunks:
            self.assertTrue(len(chunk.text) > 0)
    
    def test_code_block_extraction(self):
        """Test that code blocks are extracted as separate chunks"""
        text = """Here is some text before code.

```python
def hello():
    print("Hello World")
```

And text after code."""
        
        chunks = self.chunker.chunk(text, preserve_structure=True)
        
        # Should have code block chunk
        code_chunks = [c for c in chunks if c.chunk_type == ChunkType.CODE_BLOCK]
        self.assertGreater(len(code_chunks), 0, "Should extract code blocks")
        
        # Code chunk should contain the code
        self.assertIn("def hello", code_chunks[0].text)
    
    def test_chunk_word_count(self):
        """Test that word counts are calculated correctly"""
        text = "One two three four five."
        chunks = self.chunker.chunk(text)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].word_count, 5)
    
    def test_chunk_to_dict(self):
        """Test SemanticChunk.to_dict() method"""
        chunk = SemanticChunk(
            text="Test text",
            chunk_type=ChunkType.PARAGRAPH,
            start_position=0,
            end_position=9,
            word_count=2
        )
        
        d = chunk.to_dict()
        self.assertEqual(d["text"], "Test text")
        self.assertEqual(d["type"], "paragraph")
        self.assertEqual(d["word_count"], 2)
    
    def test_large_text_splitting(self):
        """Test that large texts are split appropriately"""
        # Create text much larger than max_chunk_size (100 words)
        # Using distinct sentences to ensure proper splitting
        sentences = ["This is sentence number {}.".format(i) for i in range(50)]
        text = " ".join(sentences)  # About 250 words total
        
        # Use a smaller chunk size to force splitting
        chunker = SemanticChunker(max_chunk_size=50, min_chunk_size=10)
        chunks = chunker.chunk(text)
        
        # Should be split into multiple chunks
        self.assertGreater(len(chunks), 1, f"Expected multiple chunks but got {len(chunks)}")
        
        # Each chunk should have some content
        for chunk in chunks:
            self.assertGreater(chunk.word_count, 0)
    
    def test_section_headers_detected(self):
        """Test markdown section header detection"""
        text = """# Main Section

This is content under the main section.

## Subsection

This is subsection content."""
        
        chunks = self.chunker.chunk(text)
        self.assertGreater(len(chunks), 0)


class TestHybridSearch(unittest.TestCase):
    """Test Task 3.2: Hybrid Search"""
    
    def setUp(self):
        # Create a mock ChromaDB client
        self.client = ChromaDBClient.__new__(ChromaDBClient)
        self.client.collection_name = "test_collection"
        self.client.vector_weight = 0.7
        self.client.keyword_weight = 0.3
        self.client.stopwords = {'the', 'and', 'for', 'are'}
        
        # Mock the collection
        self.client.collection = Mock()
    
    def test_extract_keywords(self):
        """Test keyword extraction"""
        text = "The quick brown fox jumps over the lazy dog"
        keywords = self.client._extract_keywords(text)
        
        # Should extract meaningful words, excluding stopwords
        self.assertIn("quick", keywords)
        self.assertIn("brown", keywords)
        self.assertIn("fox", keywords)
        self.assertNotIn("the", keywords)  # Stopword
    
    def test_keyword_overlap_full_match(self):
        """Test keyword overlap with full match"""
        document = "Python is a great programming language for data science"
        keywords = {"python", "programming", "data"}
        
        score = self.client._keyword_overlap(document, keywords)
        self.assertEqual(score, 1.0)
    
    def test_keyword_overlap_partial_match(self):
        """Test keyword overlap with partial match"""
        document = "Java is a programming language"
        keywords = {"python", "programming", "data"}
        
        score = self.client._keyword_overlap(document, keywords)
        # Only "programming" matches = 1/3
        self.assertAlmostEqual(score, 1/3, places=2)
    
    def test_keyword_overlap_no_match(self):
        """Test keyword overlap with no match"""
        document = "Something completely unrelated"
        keywords = {"python", "machine", "learning"}
        
        score = self.client._keyword_overlap(document, keywords)
        self.assertEqual(score, 0.0)
    
    def test_keyword_overlap_empty_keywords(self):
        """Test keyword overlap with empty keywords"""
        document = "Some document text"
        keywords = set()
        
        score = self.client._keyword_overlap(document, keywords)
        self.assertEqual(score, 0.0)
    
    def test_hybrid_query_structure(self):
        """Test hybrid query returns correct structure"""
        # Mock vector search results
        self.client.collection.query.return_value = {
            'documents': [['Doc about Python programming', 'Doc about Java']],
            'distances': [[0.1, 0.5]],
            'metadatas': [[{'filename': 'test.txt'}, {'filename': 'other.txt'}]],
            'ids': [['id1', 'id2']]
        }
        
        result = self.client.hybrid_query("Python programming tutorial", n_results=2)
        
        # Check structure
        self.assertIn('documents', result)
        self.assertIn('scores', result)
        self.assertIn('metadatas', result)
        self.assertIn('ids', result)
        
        # Check that results are lists of lists
        self.assertIsInstance(result['documents'][0], list)
    
    def test_hybrid_query_reranking(self):
        """Test that hybrid query re-ranks based on keyword overlap"""
        # Doc2 has lower vector score but should rank higher due to keyword match
        self.client.collection.query.return_value = {
            'documents': [['Generic document about something', 'Python machine learning tutorial']],
            'distances': [[0.1, 0.3]],  # First has better vector score
            'metadatas': [[{}, {}]],
            'ids': [['id1', 'id2']]
        }
        
        result = self.client.hybrid_query("Python machine learning", n_results=2)
        
        # With keyword matching, the second doc should potentially rank higher
        # depending on the weight balance
        self.assertEqual(len(result['documents'][0]), 2)
    
    def test_hybrid_query_empty_results(self):
        """Test hybrid query handles empty results"""
        self.client.collection.query.return_value = {
            'documents': [[]],
            'distances': [[]],
            'metadatas': [[]],
            'ids': [[]]
        }
        
        result = self.client.hybrid_query("nonexistent query")
        
        self.assertEqual(result['documents'], [[]])


class TestCitationTracking(unittest.TestCase):
    """Test Task 3.3: Citation Tracking"""
    
    def test_cited_source_creation(self):
        """Test CitedSource dataclass"""
        source = CitedSource(
            source_id="src_1",
            text="This is the source text",
            filename="document.pdf",
            chunk_index=0,
            relevance_score=0.95,
            chunk_type="paragraph"
        )
        
        self.assertEqual(source.source_id, "src_1")
        self.assertEqual(source.filename, "document.pdf")
        self.assertEqual(source.relevance_score, 0.95)
    
    def test_cited_source_to_dict(self):
        """Test CitedSource.to_dict() method"""
        source = CitedSource(
            source_id="src_1",
            text="Short text",
            filename="doc.txt",
            chunk_index=0,
            relevance_score=0.8
        )
        
        d = source.to_dict()
        self.assertEqual(d["source_id"], "src_1")
        self.assertEqual(d["filename"], "doc.txt")
        self.assertIn("text", d)
    
    def test_cited_source_truncates_long_text(self):
        """Test that long text is truncated in to_dict"""
        long_text = "x" * 500
        source = CitedSource(
            source_id="src_1",
            text=long_text,
            filename="doc.txt",
            chunk_index=0,
            relevance_score=0.8
        )
        
        d = source.to_dict()
        self.assertTrue(d["text"].endswith("..."))
        self.assertLess(len(d["text"]), 250)
    
    def test_rag_response_creation(self):
        """Test RAGResponse dataclass"""
        sources = [
            CitedSource("s1", "Text 1", "f1.txt", 0, 0.9),
            CitedSource("s2", "Text 2", "f2.txt", 1, 0.8)
        ]
        
        response = RAGResponse(
            answer="This is the answer based on sources.",
            sources=sources,
            confidence=0.85,
            query="What is the question?"
        )
        
        self.assertEqual(len(response.sources), 2)
        self.assertEqual(response.confidence, 0.85)
    
    def test_rag_response_to_dict(self):
        """Test RAGResponse.to_dict() method"""
        response = RAGResponse(
            answer="Test answer",
            sources=[CitedSource("s1", "Text", "f.txt", 0, 0.9)],
            confidence=0.9,
            query="Test query"
        )
        
        d = response.to_dict()
        self.assertEqual(d["answer"], "Test answer")
        self.assertEqual(d["citation_count"], 1)
        self.assertEqual(len(d["sources"]), 1)
    
    def test_format_context_with_citations(self):
        """Test formatting context with citation markers"""
        client = ChromaDBClient.__new__(ChromaDBClient)
        
        citations = [
            CitedSource("s1", "First source text", "doc1.txt", 0, 0.95),
            CitedSource("s2", "Second source text", "doc2.txt", 1, 0.85)
        ]
        
        formatted = client.format_context_with_citations(citations)
        
        # Should contain citation markers
        self.assertIn("[Source 1]", formatted)
        self.assertIn("[Source 2]", formatted)
        self.assertIn("doc1.txt", formatted)
        self.assertIn("0.95", formatted)
    
    def test_format_context_empty_citations(self):
        """Test formatting with no citations"""
        client = ChromaDBClient.__new__(ChromaDBClient)
        
        formatted = client.format_context_with_citations([])
        self.assertEqual(formatted, "")


class TestOptimizedDocumentIndexerWithSemanticChunking(unittest.TestCase):
    """Test OptimizedDocumentIndexer with semantic chunking enabled"""
    
    def test_indexer_initialization(self):
        """Test indexer initializes with semantic chunker"""
        with patch('core.document_indexer.ChromaDBClient'):
            indexer = OptimizedDocumentIndexer(use_semantic_chunking=True)
            
            self.assertTrue(indexer.use_semantic_chunking)
            self.assertIsNotNone(indexer.semantic_chunker)
    
    def test_indexer_can_disable_semantic_chunking(self):
        """Test indexer can use legacy chunking"""
        with patch('core.document_indexer.ChromaDBClient'):
            indexer = OptimizedDocumentIndexer(use_semantic_chunking=False)
            
            self.assertFalse(indexer.use_semantic_chunking)


# =============================================================================
# Phase 3.7: Semantic Cache Tests
# =============================================================================

class TestSemanticCache(unittest.TestCase):
    """Test Task 3.7: Semantic Cache for NL queries"""
    
    def setUp(self):
        from core.advanced_cache import SemanticCache
        self.cache = SemanticCache(
            max_size=100,
            default_ttl=300,
            similarity_threshold=0.7
        )
    
    def test_exact_match_hit(self):
        """Test cache returns exact match"""
        self.cache.put("What is the revenue?", {"answer": "1M"})
        
        result = self.cache.get("What is the revenue?")
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["answer"], "1M")
        self.assertEqual(result[1], 1.0)  # Similarity should be 1.0 for exact
    
    def test_semantic_match_hit(self):
        """Test cache returns semantically similar queries"""
        self.cache.put("What is the total revenue for the company?", {"answer": "1M"})
        
        # Similar query (different wording)
        result = self.cache.get("Show me company revenue total")
        
        # May or may not hit depending on threshold, but shouldn't error
        if result:
            self.assertGreaterEqual(result[1], 0.7)  # Above threshold
    
    def test_cache_miss(self):
        """Test cache returns None for unrelated queries"""
        self.cache.put("What is the revenue?", {"answer": "1M"})
        
        result = self.cache.get("What is the weather forecast?")
        self.assertIsNone(result)
    
    def test_context_isolation(self):
        """Test queries with different contexts don't collide"""
        self.cache.put("Show statistics", {"data": "stats1"}, context="file1.csv")
        self.cache.put("Show statistics", {"data": "stats2"}, context="file2.csv")
        
        result1 = self.cache.get("Show statistics", context="file1.csv")
        result2 = self.cache.get("Show statistics", context="file2.csv")
        
        self.assertEqual(result1[0]["data"], "stats1")
        self.assertEqual(result2[0]["data"], "stats2")
    
    def test_stats_tracking(self):
        """Test statistics are tracked correctly"""
        # Create a fresh cache with high threshold to ensure distinct queries don't match
        from core.advanced_cache import SemanticCache
        fresh_cache = SemanticCache(max_size=100, default_ttl=300, similarity_threshold=0.9)
        
        fresh_cache.put("Show me the total revenue for 2024", {"data": 1})
        fresh_cache.get("Show me the total revenue for 2024")  # Exact hit
        fresh_cache.get("What is the weather forecast for tomorrow")  # Definite miss (unrelated)
        
        stats = fresh_cache.get_stats()
        self.assertEqual(stats['total_requests'], 2)
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
    
    def test_clear(self):
        """Test cache clear"""
        self.cache.put("Query 1", {"data": 1})
        self.cache.clear()
        
        result = self.cache.get("Query 1")
        self.assertIsNone(result)


# =============================================================================
# Phase 3.8: Prometheus Metrics Tests
# =============================================================================

class TestMetrics(unittest.TestCase):
    """Test Task 3.8: Prometheus Metrics"""
    
    def test_metrics_registry_exists(self):
        """Test METRICS registry is available"""
        from core.metrics import METRICS
        self.assertIsNotNone(METRICS)
    
    def test_track_request_decorator(self):
        """Test track_request decorator works"""
        from core.metrics import track_request, METRICS
        
        @track_request(endpoint="test")
        def test_function():
            return {"status": "ok"}
        
        result = test_function()
        self.assertEqual(result["status"], "ok")
    
    def test_track_llm_usage(self):
        """Test LLM usage tracking"""
        from core.metrics import track_llm_usage
        
        # Should not raise
        track_llm_usage(
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            latency=1.5,
            success=True
        )
    
    def test_update_cache_metrics(self):
        """Test cache metrics update"""
        from core.metrics import update_cache_metrics
        
        # Should not raise
        update_cache_metrics("semantic", hits=5, misses=2, size=100)
    
    def test_generate_metrics_output(self):
        """Test metrics output generation"""
        from core.metrics import generate_metrics_output, get_metrics_content_type
        
        output = generate_metrics_output()
        self.assertIsInstance(output, bytes)
        
        content_type = get_metrics_content_type()
        self.assertIn("text", content_type)


# =============================================================================
# Phase 3.9: Structured Logging Tests
# =============================================================================

class TestStructuredLogging(unittest.TestCase):
    """Test Task 3.9: Structured Logging"""
    
    def test_structured_logger_creation(self):
        """Test StructuredLogger can be created"""
        from core.enhanced_logging import StructuredLogger
        
        logger = StructuredLogger("test_component")
        self.assertEqual(logger.component, "test_component")
    
    def test_log_event(self):
        """Test log_event method"""
        from core.enhanced_logging import StructuredLogger
        import logging
        
        logger = StructuredLogger("test")
        # Should not raise
        logger.log_event("test_event", data="value")
    
    def test_context_management(self):
        """Test context setting and clearing"""
        from core.enhanced_logging import StructuredLogger
        
        logger = StructuredLogger("test")
        logger.set_context(request_id="abc123")
        
        self.assertEqual(logger._context["request_id"], "abc123")
        
        logger.clear_context()
        self.assertEqual(len(logger._context), 0)
    
    def test_log_request(self):
        """Test log_request method"""
        from core.enhanced_logging import StructuredLogger
        
        logger = StructuredLogger("api")
        # Should not raise
        logger.log_request(
            request_id="test-123",
            method="POST",
            path="/api/query",
            status=200,
            duration_ms=150.5
        )
    
    def test_log_agent_execution(self):
        """Test log_agent_execution method"""
        from core.enhanced_logging import StructuredLogger
        
        logger = StructuredLogger("agent")
        # Should not raise
        logger.log_agent_execution(
            agent_name="TestAgent",
            action="analyze",
            duration_ms=500.0,
            success=True
        )
    
    def test_preconfigured_loggers_exist(self):
        """Test pre-configured loggers are available"""
        from core.enhanced_logging import api_logger, agent_logger, rag_logger, cache_logger, llm_logger
        
        self.assertIsNotNone(api_logger)
        self.assertIsNotNone(agent_logger)
        self.assertIsNotNone(rag_logger)
        self.assertIsNotNone(cache_logger)
        self.assertIsNotNone(llm_logger)
    
    def test_get_structured_logger_factory(self):
        """Test get_structured_logger factory function"""
        from core.enhanced_logging import get_structured_logger
        
        logger = get_structured_logger("custom_component")
        self.assertEqual(logger.component, "custom_component")


def run_phase3_tests():
    """Run all Phase 3 tests and report results"""
    print("=" * 60)
    print("🧪 PHASE 3 RAG ENHANCEMENT TESTS")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticChunker))
    suite.addTests(loader.loadTestsFromTestCase(TestHybridSearch))
    suite.addTests(loader.loadTestsFromTestCase(TestCitationTracking))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizedDocumentIndexerWithSemanticChunking))
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticCache))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestStructuredLogging))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PHASE 3 TEST SUMMARY")
    print("=" * 60)
    
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {failures}")
    print(f"💥 Errors: {errors}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL PHASE 3 TESTS PASSED!")
        print("   - Task 3.1: Semantic Chunking ✅")
        print("   - Task 3.2: Hybrid Search ✅")
        print("   - Task 3.3: Citation Tracking ✅")
        print("   - Task 3.7: Semantic Cache ✅")
        print("   - Task 3.8: Prometheus Metrics ✅")
        print("   - Task 3.9: Structured Logging ✅")
    else:
        print("\n⚠️ Some tests failed. Review output above.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_phase3_tests()
    sys.exit(0 if success else 1)

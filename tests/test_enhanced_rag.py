"""
Tests for Enhanced RAG Pipeline
================================

Comprehensive tests for the enhanced RAG system components.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestQueryExpander:
    """Tests for query expansion."""
    
    def test_expand_with_synonyms(self):
        """Test query expansion with synonyms."""
        from src.backend.rag.enhanced_rag_pipeline import QueryExpander
        
        expander = QueryExpander()
        expanded = expander.expand("analyze the document")
        
        assert "analyze the document" in expanded
        assert len(expanded) >= 1
    
    def test_expand_returns_original(self):
        """Test that original query is always included."""
        from src.backend.rag.enhanced_rag_pipeline import QueryExpander
        
        expander = QueryExpander()
        query = "unique query without synonyms xyz123"
        expanded = expander.expand(query)
        
        assert query in expanded
    
    def test_expand_max_expansions(self):
        """Test that max_expansions limit is respected."""
        from src.backend.rag.enhanced_rag_pipeline import QueryExpander
        
        expander = QueryExpander()
        expanded = expander.expand("analyze and summarize the document", max_expansions=2)
        
        assert len(expanded) <= 2
    
    def test_extract_key_terms(self):
        """Test key term extraction."""
        from src.backend.rag.enhanced_rag_pipeline import QueryExpander
        
        expander = QueryExpander()
        terms = expander.extract_key_terms("What is the capital of France?")
        
        assert "capital" in terms
        assert "france" in terms
        # Stop words should be removed
        assert "what" not in terms
        assert "the" not in terms


class TestBM25Scorer:
    """Tests for BM25 scoring."""
    
    def test_index_documents(self):
        """Test document indexing."""
        from src.backend.rag.enhanced_rag_pipeline import BM25Scorer
        
        scorer = BM25Scorer()
        documents = [
            ("doc1", "Python is a programming language"),
            ("doc2", "Java is also a programming language"),
            ("doc3", "Machine learning uses Python"),
        ]
        
        scorer.index(documents)
        
        assert len(scorer.doc_lengths) == 3
        assert scorer.avg_doc_length > 0
    
    def test_score_document(self):
        """Test document scoring."""
        from src.backend.rag.enhanced_rag_pipeline import BM25Scorer
        
        scorer = BM25Scorer()
        documents = [
            ("doc1", "Python is a programming language"),
            ("doc2", "Java is also a programming language"),
            ("doc3", "Machine learning uses Python extensively"),
        ]
        
        scorer.index(documents)
        
        # Query about Python
        score1 = scorer.score("Python programming", "doc1")
        score2 = scorer.score("Python programming", "doc2")
        
        # doc1 should score higher (has Python)
        assert score1 > score2
    
    def test_score_unindexed_document(self):
        """Test scoring unindexed document returns 0."""
        from src.backend.rag.enhanced_rag_pipeline import BM25Scorer
        
        scorer = BM25Scorer()
        score = scorer.score("test query", "nonexistent_doc")
        
        assert score == 0.0
    
    def test_empty_index(self):
        """Test indexing empty document list."""
        from src.backend.rag.enhanced_rag_pipeline import BM25Scorer
        
        scorer = BM25Scorer()
        scorer.index([])
        
        assert len(scorer.doc_lengths) == 0


class TestReRanker:
    """Tests for re-ranking."""
    
    def test_rerank_basic(self):
        """Test basic re-ranking."""
        from src.backend.rag.enhanced_rag_pipeline import ReRanker, RetrievedChunk
        
        reranker = ReRanker()
        
        chunks = [
            RetrievedChunk(chunk_id="1", content="Python programming basics", score=0.5),
            RetrievedChunk(chunk_id="2", content="Python is a great language for ML", score=0.7),
            RetrievedChunk(chunk_id="3", content="Java programming basics", score=0.6),
        ]
        
        reranked = reranker.rerank("Python programming", chunks, top_k=2)
        
        assert len(reranked) == 2
        # Both Python chunks should rank higher
        assert all("python" in c.content.lower() for c in reranked)
    
    def test_rerank_empty_list(self):
        """Test re-ranking empty list."""
        from src.backend.rag.enhanced_rag_pipeline import ReRanker
        
        reranker = ReRanker()
        reranked = reranker.rerank("query", [], top_k=5)
        
        assert reranked == []
    
    def test_rerank_respects_top_k(self):
        """Test that top_k limit is respected."""
        from src.backend.rag.enhanced_rag_pipeline import ReRanker, RetrievedChunk
        
        reranker = ReRanker()
        chunks = [
            RetrievedChunk(chunk_id=str(i), content=f"Content {i}", score=0.5)
            for i in range(10)
        ]
        
        reranked = reranker.rerank("query", chunks, top_k=3)
        
        assert len(reranked) == 3


class TestContextCompressor:
    """Tests for context compression."""
    
    def test_compress_basic(self):
        """Test basic context compression."""
        from src.backend.rag.enhanced_rag_pipeline import ContextCompressor, RetrievedChunk
        
        compressor = ContextCompressor(max_tokens=1000)
        
        chunks = [
            RetrievedChunk(chunk_id="1", content="First chunk content", score=0.9),
            RetrievedChunk(chunk_id="2", content="Second chunk content", score=0.8),
        ]
        
        context, chunks_used = compressor.compress(chunks, "test query")
        
        assert "First chunk" in context
        assert "Second chunk" in context
        assert len(chunks_used) == 2
    
    def test_compress_respects_limit(self):
        """Test that compression respects token limit."""
        from src.backend.rag.enhanced_rag_pipeline import ContextCompressor, RetrievedChunk
        
        compressor = ContextCompressor(max_tokens=50)  # Very small limit
        
        chunks = [
            RetrievedChunk(chunk_id="1", content="A" * 500, score=0.9),
            RetrievedChunk(chunk_id="2", content="B" * 500, score=0.8),
        ]
        
        context, chunks_used = compressor.compress(chunks, "query")
        
        # Should be limited by max_chars (50 * 4 = 200)
        assert len(context) <= 250  # Some buffer for formatting
    
    def test_compress_empty_chunks(self):
        """Test compression with empty chunks."""
        from src.backend.rag.enhanced_rag_pipeline import ContextCompressor
        
        compressor = ContextCompressor()
        context, chunks_used = compressor.compress([], "query")
        
        assert context == ""
        assert chunks_used == []
    
    def test_compress_includes_source(self):
        """Test that source info is included."""
        from src.backend.rag.enhanced_rag_pipeline import ContextCompressor, RetrievedChunk
        
        compressor = ContextCompressor()
        
        chunks = [
            RetrievedChunk(
                chunk_id="1", 
                content="Test content", 
                score=0.9,
                source="test_doc.pdf",
                page_number=5
            ),
        ]
        
        context, _ = compressor.compress(chunks, "query")
        
        assert "test_doc.pdf" in context
        assert "Page 5" in context


class TestCitationTracker:
    """Tests for citation tracking."""
    
    def test_add_citation(self):
        """Test adding citations."""
        from src.backend.rag.enhanced_rag_pipeline import CitationTracker, RetrievedChunk
        
        tracker = CitationTracker()
        chunk = RetrievedChunk(
            chunk_id="1",
            content="Test content",
            score=0.8,
            source="document.pdf"
        )
        
        marker = tracker.add_citation(chunk, 1)
        
        assert marker == "[1]"
        assert len(tracker.citations) == 1
    
    def test_format_citations(self):
        """Test citation formatting."""
        from src.backend.rag.enhanced_rag_pipeline import CitationTracker, RetrievedChunk
        
        tracker = CitationTracker()
        
        for i in range(3):
            chunk = RetrievedChunk(
                chunk_id=str(i),
                content=f"Content {i}",
                score=0.8,
                source=f"doc{i}.pdf",
                page_number=i + 1
            )
            tracker.add_citation(chunk, i + 1)
        
        formatted = tracker.format_citations()
        
        assert "References" in formatted
        assert "[1]" in formatted
        assert "[2]" in formatted
        assert "[3]" in formatted
    
    def test_format_empty_citations(self):
        """Test formatting with no citations."""
        from src.backend.rag.enhanced_rag_pipeline import CitationTracker
        
        tracker = CitationTracker()
        formatted = tracker.format_citations()
        
        assert formatted == ""


class TestConfidenceScorer:
    """Tests for confidence scoring."""
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        from src.backend.rag.enhanced_rag_pipeline import ConfidenceScorer, RetrievedChunk
        
        scorer = ConfidenceScorer()
        
        chunks = [
            RetrievedChunk(
                chunk_id="1",
                content="Python is a programming language",
                score=0.9
            ),
            RetrievedChunk(
                chunk_id="2",
                content="Python is used for machine learning",
                score=0.85
            ),
        ]
        
        confidence = scorer.calculate_confidence(
            chunks,
            "What is Python?",
            "Python is a programming language widely used for machine learning."
        )
        
        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be reasonably confident
    
    def test_low_confidence_no_chunks(self):
        """Test low confidence when no chunks."""
        from src.backend.rag.enhanced_rag_pipeline import ConfidenceScorer
        
        scorer = ConfidenceScorer()
        confidence = scorer.calculate_confidence([], "query", "answer")
        
        assert confidence == 0.0
    
    def test_confidence_with_poor_coverage(self):
        """Test confidence with poor term coverage."""
        from src.backend.rag.enhanced_rag_pipeline import ConfidenceScorer, RetrievedChunk
        
        scorer = ConfidenceScorer()
        
        chunks = [
            RetrievedChunk(
                chunk_id="1",
                content="Completely unrelated content about cooking",
                score=0.5
            ),
        ]
        
        confidence = scorer.calculate_confidence(
            chunks,
            "What is quantum physics?",
            "Some answer about physics."
        )
        
        # Should be lower confidence due to poor coverage
        assert confidence < 0.7


class TestEnhancedRAGPipeline:
    """Tests for the main RAG pipeline."""
    
    def test_pipeline_creation(self):
        """Test pipeline creation."""
        from src.backend.rag.enhanced_rag_pipeline import EnhancedRAGPipeline
        
        pipeline = EnhancedRAGPipeline(
            max_context_tokens=4000,
            rerank_top_k=5
        )
        
        assert pipeline is not None
        assert pipeline.rerank_top_k == 5
    
    def test_retrieve_and_rerank(self):
        """Test retrieval and reranking."""
        from src.backend.rag.enhanced_rag_pipeline import EnhancedRAGPipeline
        
        pipeline = EnhancedRAGPipeline()
        
        mock_results = [
            {'id': 'doc1', 'content': 'Python programming basics', 'score': 0.8},
            {'id': 'doc2', 'content': 'Java programming guide', 'score': 0.7},
            {'id': 'doc3', 'content': 'Python for machine learning', 'score': 0.85},
        ]
        
        chunks = pipeline.retrieve("Python basics", mock_results)
        
        assert len(chunks) > 0
        assert all(hasattr(c, 'score') for c in chunks)
    
    def test_generate_context(self):
        """Test context generation with citations."""
        from src.backend.rag.enhanced_rag_pipeline import EnhancedRAGPipeline, RetrievedChunk
        
        pipeline = EnhancedRAGPipeline()
        
        chunks = [
            RetrievedChunk(
                chunk_id="1",
                content="First relevant content",
                score=0.9,
                source="doc.pdf"
            ),
            RetrievedChunk(
                chunk_id="2",
                content="Second relevant content",
                score=0.85,
                source="doc.pdf"
            ),
        ]
        
        context, chunks_used, tracker = pipeline.generate_context("query", chunks)
        
        assert "First relevant" in context
        assert len(tracker.citations) == 2
    
    def test_process_complete(self):
        """Test complete pipeline processing."""
        from src.backend.rag.enhanced_rag_pipeline import EnhancedRAGPipeline
        
        pipeline = EnhancedRAGPipeline()
        
        mock_results = [
            {
                'id': 'doc1',
                'content': 'Python is a programming language',
                'score': 0.9,
                'source': 'guide.pdf'
            },
        ]
        
        result = pipeline.process("What is Python?", mock_results)
        
        assert result.answer is not None
        assert len(result.chunks_used) > 0
        assert 0 <= result.confidence <= 1
    
    def test_process_empty_results(self):
        """Test processing with no results."""
        from src.backend.rag.enhanced_rag_pipeline import EnhancedRAGPipeline
        
        pipeline = EnhancedRAGPipeline()
        result = pipeline.process("query", [])
        
        assert result.confidence == 0.0
        assert "couldn't find" in result.answer.lower()
    
    def test_process_with_custom_generate(self):
        """Test processing with custom generation function."""
        from src.backend.rag.enhanced_rag_pipeline import EnhancedRAGPipeline
        
        pipeline = EnhancedRAGPipeline()
        
        mock_results = [
            {'id': 'doc1', 'content': 'Test content', 'score': 0.8},
        ]
        
        def custom_generate(query, context):
            return f"Custom answer for: {query}"
        
        result = pipeline.process("test query", mock_results, generate_fn=custom_generate)
        
        assert "Custom answer" in result.answer


class TestFactoryFunction:
    """Tests for factory function."""
    
    def test_create_pipeline(self):
        """Test pipeline creation via factory."""
        from src.backend.rag.enhanced_rag_pipeline import create_enhanced_rag_pipeline
        
        pipeline = create_enhanced_rag_pipeline(
            max_context_tokens=2000,
            rerank_top_k=3,
            min_confidence=0.4
        )
        
        assert pipeline is not None
        assert pipeline.rerank_top_k == 3
        assert pipeline.min_confidence == 0.4


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

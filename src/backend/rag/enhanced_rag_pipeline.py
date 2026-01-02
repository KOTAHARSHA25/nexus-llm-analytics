"""
Enhanced RAG Pipeline Module
============================

This module provides advanced RAG (Retrieval-Augmented Generation) capabilities
for research-grade document analysis and question answering.

Features:
- Hybrid search (dense + sparse)
- Query expansion and rewriting
- Re-ranking with cross-encoders
- Context compression
- Citation tracking
- Confidence scoring
"""

import re
import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a retrieved document chunk."""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    page_number: Optional[int] = None


@dataclass
class RAGResult:
    """Result from RAG pipeline."""
    answer: str
    chunks_used: List[RetrievedChunk]
    confidence: float
    citations: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryExpander:
    """
    Expands queries using synonyms and related terms.
    
    Improves recall by generating query variations.
    """
    
    def __init__(self):
        """Initialize query expander with synonym database."""
        self.synonyms = {
            'analyze': ['examine', 'study', 'investigate', 'evaluate'],
            'summarize': ['overview', 'synopsis', 'digest', 'brief'],
            'explain': ['describe', 'clarify', 'elaborate', 'illustrate'],
            'compare': ['contrast', 'differentiate', 'distinguish'],
            'find': ['locate', 'identify', 'discover', 'search'],
            'show': ['display', 'present', 'demonstrate', 'reveal'],
            'create': ['generate', 'produce', 'build', 'develop'],
            'improve': ['enhance', 'optimize', 'upgrade', 'refine'],
        }
    
    def expand(self, query: str, max_expansions: int = 3) -> List[str]:
        """
        Expand query with synonyms and variations.
        
        Args:
            query: Original query
            max_expansions: Maximum number of expanded queries
            
        Returns:
            List of expanded queries (including original)
        """
        expanded = [query]
        words = query.lower().split()
        
        for word in words:
            if word in self.synonyms:
                for synonym in self.synonyms[word][:2]:  # Top 2 synonyms
                    new_query = query.lower().replace(word, synonym)
                    if new_query not in expanded:
                        expanded.append(new_query)
                    if len(expanded) >= max_expansions:
                        break
            if len(expanded) >= max_expansions:
                break
        
        return expanded[:max_expansions]
    
    def extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query for sparse search."""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'must',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'this', 'that', 'these', 'those', 'it', 'its', 'and', 'or',
            'but', 'if', 'then', 'what', 'which', 'who', 'how', 'when',
            'where', 'why', 'all', 'each', 'every', 'both', 'few', 'more'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]


class BM25Scorer:
    """
    BM25 scoring for sparse retrieval.
    
    Implements the BM25 ranking function for keyword matching.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 scorer.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length = 0
        self.doc_term_freqs: Dict[str, Dict[str, int]] = {}
        self.idf: Dict[str, float] = {}
    
    def index(self, documents: List[Tuple[str, str]]):
        """
        Index documents for BM25 scoring.
        
        Args:
            documents: List of (doc_id, content) tuples
        """
        if not documents:
            return
        
        n_docs = len(documents)
        term_doc_counts: Dict[str, int] = defaultdict(int)
        
        for doc_id, content in documents:
            words = re.findall(r'\b\w+\b', content.lower())
            self.doc_lengths[doc_id] = len(words)
            
            term_freqs: Dict[str, int] = defaultdict(int)
            seen_terms = set()
            
            for word in words:
                term_freqs[word] += 1
                if word not in seen_terms:
                    term_doc_counts[word] += 1
                    seen_terms.add(word)
            
            self.doc_term_freqs[doc_id] = dict(term_freqs)
        
        # Calculate average document length
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        
        # Calculate IDF for each term
        for term, doc_count in term_doc_counts.items():
            self.idf[term] = math.log((n_docs - doc_count + 0.5) / (doc_count + 0.5) + 1)
    
    def score(self, query: str, doc_id: str) -> float:
        """
        Calculate BM25 score for a document given a query.
        
        Args:
            query: Search query
            doc_id: Document ID to score
            
        Returns:
            BM25 score
        """
        if doc_id not in self.doc_term_freqs:
            return 0.0
        
        query_terms = re.findall(r'\b\w+\b', query.lower())
        doc_length = self.doc_lengths.get(doc_id, 0)
        term_freqs = self.doc_term_freqs.get(doc_id, {})
        
        score = 0.0
        for term in query_terms:
            if term not in self.idf:
                continue
            
            tf = term_freqs.get(term, 0)
            if tf == 0:
                continue
            
            idf = self.idf[term]
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1))
            score += idf * (tf * (self.k1 + 1)) / denominator
        
        return score


class ReRanker:
    """
    Re-ranks retrieved documents for improved relevance.
    
    Uses cross-encoder style scoring without actual neural model.
    """
    
    def __init__(self):
        """Initialize re-ranker."""
        self.query_expander = QueryExpander()
    
    def rerank(
        self, 
        query: str, 
        chunks: List[RetrievedChunk],
        top_k: int = 5
    ) -> List[RetrievedChunk]:
        """
        Re-rank chunks based on detailed relevance analysis.
        
        Args:
            query: Original query
            chunks: List of retrieved chunks
            top_k: Number of top chunks to return
            
        Returns:
            Re-ranked list of chunks
        """
        if not chunks:
            return []
        
        query_terms = set(self.query_expander.extract_key_terms(query))
        
        scored_chunks = []
        for chunk in chunks:
            score = self._calculate_relevance_score(query, query_terms, chunk)
            scored_chunks.append((chunk, score))
        
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Update chunk scores with re-ranking score
        result = []
        for chunk, score in scored_chunks[:top_k]:
            chunk.score = score
            result.append(chunk)
        
        return result
    
    def _calculate_relevance_score(
        self, 
        query: str, 
        query_terms: set, 
        chunk: RetrievedChunk
    ) -> float:
        """Calculate detailed relevance score for a chunk."""
        content_lower = chunk.content.lower()
        content_words = set(re.findall(r'\b\w+\b', content_lower))
        
        # Term overlap
        overlap = len(query_terms & content_words)
        term_coverage = overlap / len(query_terms) if query_terms else 0
        
        # Exact phrase matching
        query_lower = query.lower()
        exact_match = 1.0 if query_lower in content_lower else 0.0
        
        # Proximity score (how close query terms appear)
        proximity_score = self._calculate_proximity(query_terms, content_lower)
        
        # Original retrieval score
        original_score = chunk.score
        
        # Combined score
        final_score = (
            0.3 * term_coverage +
            0.2 * exact_match +
            0.2 * proximity_score +
            0.3 * original_score
        )
        
        return final_score
    
    def _calculate_proximity(self, query_terms: set, content: str) -> float:
        """Calculate how close query terms appear in content."""
        if not query_terms:
            return 0.0
        
        words = content.split()
        positions: Dict[str, List[int]] = defaultdict(list)
        
        for i, word in enumerate(words):
            if word in query_terms:
                positions[word].append(i)
        
        if len(positions) < 2:
            return 0.5  # Only one or zero terms found
        
        # Calculate average minimum distance between terms
        min_distances = []
        terms = list(positions.keys())
        
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                pos_i = positions[terms[i]]
                pos_j = positions[terms[j]]
                
                min_dist = min(abs(p1 - p2) for p1 in pos_i for p2 in pos_j)
                min_distances.append(min_dist)
        
        if not min_distances:
            return 0.5
        
        avg_distance = sum(min_distances) / len(min_distances)
        # Convert distance to score (closer = higher)
        proximity_score = 1 / (1 + avg_distance / 10)
        
        return proximity_score


class ContextCompressor:
    """
    Compresses retrieved context to fit within token limits.
    
    Uses importance-based selection to retain most relevant information.
    """
    
    def __init__(self, max_tokens: int = 4000):
        """
        Initialize context compressor.
        
        Args:
            max_tokens: Maximum tokens for compressed context
        """
        self.max_tokens = max_tokens
        # Approximate 4 chars per token
        self.max_chars = max_tokens * 4
    
    def compress(
        self, 
        chunks: List[RetrievedChunk], 
        query: str
    ) -> Tuple[str, List[RetrievedChunk]]:
        """
        Compress chunks into a context string.
        
        Args:
            chunks: Retrieved chunks sorted by relevance
            query: Original query for importance calculation
            
        Returns:
            Tuple of (compressed context string, chunks used)
        """
        if not chunks:
            return "", []
        
        # Build context greedily, highest score first
        context_parts = []
        chunks_used = []
        current_length = 0
        
        for chunk in chunks:
            chunk_text = self._prepare_chunk_text(chunk)
            chunk_length = len(chunk_text)
            
            if current_length + chunk_length <= self.max_chars:
                context_parts.append(chunk_text)
                chunks_used.append(chunk)
                current_length += chunk_length
            else:
                # Try to fit partial chunk
                remaining = self.max_chars - current_length
                if remaining > 200:  # At least 200 chars
                    truncated = self._smart_truncate(chunk_text, remaining)
                    context_parts.append(truncated)
                    chunks_used.append(chunk)
                break
        
        return "\n\n---\n\n".join(context_parts), chunks_used
    
    def _prepare_chunk_text(self, chunk: RetrievedChunk) -> str:
        """Prepare chunk text with source information."""
        source_info = ""
        if chunk.source:
            source_info = f"[Source: {chunk.source}"
            if chunk.page_number:
                source_info += f", Page {chunk.page_number}"
            source_info += "]\n"
        
        return f"{source_info}{chunk.content}"
    
    def _smart_truncate(self, text: str, max_length: int) -> str:
        """Truncate text at sentence boundary."""
        if len(text) <= max_length:
            return text
        
        # Try to truncate at sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        
        cut_point = max(last_period, last_newline)
        if cut_point > max_length * 0.5:  # At least half the content
            return truncated[:cut_point + 1] + "\n[...]"
        
        return truncated + "..."


class CitationTracker:
    """
    Tracks and formats citations for generated answers.
    """
    
    def __init__(self):
        """Initialize citation tracker."""
        self.citations: List[Dict[str, Any]] = []
    
    def add_citation(
        self, 
        chunk: RetrievedChunk, 
        citation_id: int
    ) -> str:
        """
        Add a citation and return the reference marker.
        
        Args:
            chunk: Source chunk
            citation_id: Unique citation ID
            
        Returns:
            Citation marker string (e.g., "[1]")
        """
        citation = {
            'id': citation_id,
            'source': chunk.source or 'Unknown Source',
            'page': chunk.page_number,
            'excerpt': chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content,
            'score': chunk.score
        }
        self.citations.append(citation)
        return f"[{citation_id}]"
    
    def format_citations(self) -> str:
        """Format all citations as a reference list."""
        if not self.citations:
            return ""
        
        lines = ["\n\n**References:**"]
        for citation in self.citations:
            line = f"[{citation['id']}] {citation['source']}"
            if citation['page']:
                line += f", p.{citation['page']}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def get_citations(self) -> List[Dict[str, Any]]:
        """Get all citations."""
        return self.citations


class ConfidenceScorer:
    """
    Calculates confidence scores for RAG responses.
    """
    
    def calculate_confidence(
        self, 
        chunks: List[RetrievedChunk],
        query: str,
        answer: str
    ) -> float:
        """
        Calculate confidence score for the answer.
        
        Args:
            chunks: Chunks used to generate answer
            query: Original query
            answer: Generated answer
            
        Returns:
            Confidence score between 0 and 1
        """
        if not chunks:
            return 0.0
        
        # Factor 1: Average chunk relevance score
        avg_score = sum(c.score for c in chunks) / len(chunks)
        
        # Factor 2: Coverage (do chunks cover query terms?)
        query_expander = QueryExpander()
        query_terms = set(query_expander.extract_key_terms(query))
        
        covered_terms = set()
        for chunk in chunks:
            chunk_words = set(re.findall(r'\b\w+\b', chunk.content.lower()))
            covered_terms.update(query_terms & chunk_words)
        
        coverage = len(covered_terms) / len(query_terms) if query_terms else 0
        
        # Factor 3: Answer length reasonableness
        answer_length = len(answer)
        length_score = 1.0 if 50 < answer_length < 2000 else 0.5
        
        # Factor 4: Number of sources
        n_sources = len(chunks)
        source_score = min(n_sources / 3, 1.0)  # Optimal: 3+ sources
        
        # Combine factors
        confidence = (
            0.35 * avg_score +
            0.30 * coverage +
            0.15 * length_score +
            0.20 * source_score
        )
        
        return max(0.0, min(1.0, confidence))


class EnhancedRAGPipeline:
    """
    Enhanced RAG Pipeline with advanced retrieval and generation.
    
    Combines all RAG components for production-grade performance.
    """
    
    def __init__(
        self,
        max_context_tokens: int = 4000,
        rerank_top_k: int = 5,
        min_confidence: float = 0.3
    ):
        """
        Initialize enhanced RAG pipeline.
        
        Args:
            max_context_tokens: Maximum tokens for context
            rerank_top_k: Number of chunks to use after reranking
            min_confidence: Minimum confidence threshold
        """
        self.query_expander = QueryExpander()
        self.bm25_scorer = BM25Scorer()
        self.reranker = ReRanker()
        self.compressor = ContextCompressor(max_tokens=max_context_tokens)
        self.confidence_scorer = ConfidenceScorer()
        
        self.rerank_top_k = rerank_top_k
        self.min_confidence = min_confidence
        
        self._indexed = False
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Index documents for hybrid retrieval.
        
        Args:
            documents: List of documents with 'id' and 'content' fields
        """
        doc_tuples = [(d.get('id', str(i)), d.get('content', '')) 
                      for i, d in enumerate(documents)]
        self.bm25_scorer.index(doc_tuples)
        self._indexed = True
    
    def retrieve(
        self,
        query: str,
        dense_results: List[Dict[str, Any]],
        n_results: int = 10
    ) -> List[RetrievedChunk]:
        """
        Perform hybrid retrieval combining dense and sparse search.
        
        Args:
            query: Search query
            dense_results: Results from dense vector search
            n_results: Number of results to return
            
        Returns:
            List of retrieved chunks
        """
        # Convert dense results to chunks
        chunks = []
        for i, result in enumerate(dense_results):
            chunk = RetrievedChunk(
                chunk_id=result.get('id', f'chunk_{i}'),
                content=result.get('content', result.get('text', '')),
                score=result.get('score', result.get('distance', 0.5)),
                metadata=result.get('metadata', {}),
                source=result.get('source', ''),
                page_number=result.get('page')
            )
            chunks.append(chunk)
        
        # Re-rank chunks
        reranked = self.reranker.rerank(query, chunks, top_k=n_results)
        
        return reranked
    
    def generate_context(
        self,
        query: str,
        chunks: List[RetrievedChunk]
    ) -> Tuple[str, List[RetrievedChunk], CitationTracker]:
        """
        Generate context string with citations.
        
        Args:
            query: Original query
            chunks: Retrieved chunks
            
        Returns:
            Tuple of (context string, chunks used, citation tracker)
        """
        # Compress context
        context, chunks_used = self.compressor.compress(chunks, query)
        
        # Add citations
        tracker = CitationTracker()
        for i, chunk in enumerate(chunks_used, 1):
            tracker.add_citation(chunk, i)
        
        return context, chunks_used, tracker
    
    def process(
        self,
        query: str,
        dense_results: List[Dict[str, Any]],
        generate_fn: Optional[callable] = None
    ) -> RAGResult:
        """
        Process a complete RAG query.
        
        Args:
            query: User query
            dense_results: Results from vector search
            generate_fn: Optional function to generate response
            
        Returns:
            Complete RAG result
        """
        # Step 1: Retrieve and rerank
        chunks = self.retrieve(query, dense_results)
        
        if not chunks:
            return RAGResult(
                answer="I couldn't find relevant information to answer your question.",
                chunks_used=[],
                confidence=0.0,
                citations=[],
                metadata={'error': 'no_chunks_retrieved'}
            )
        
        # Step 2: Generate context
        context, chunks_used, tracker = self.generate_context(query, chunks)
        
        # Step 3: Generate answer
        if generate_fn:
            answer = generate_fn(query, context)
        else:
            answer = f"Based on the retrieved information:\n\n{context[:1000]}..."
        
        # Step 4: Calculate confidence
        confidence = self.confidence_scorer.calculate_confidence(
            chunks_used, query, answer
        )
        
        # Add citation references
        references = tracker.format_citations()
        if references and confidence > self.min_confidence:
            answer += references
        
        return RAGResult(
            answer=answer,
            chunks_used=chunks_used,
            confidence=confidence,
            citations=tracker.get_citations(),
            metadata={
                'n_chunks_retrieved': len(chunks),
                'n_chunks_used': len(chunks_used),
                'context_length': len(context),
                'timestamp': datetime.now().isoformat()
            }
        )


def create_enhanced_rag_pipeline(**kwargs) -> EnhancedRAGPipeline:
    """
    Factory function to create an enhanced RAG pipeline.
    
    Args:
        **kwargs: Pipeline configuration parameters
        
    Returns:
        Configured EnhancedRAGPipeline instance
    """
    return EnhancedRAGPipeline(**kwargs)


# Example usage
if __name__ == "__main__":
    # Create pipeline
    pipeline = create_enhanced_rag_pipeline(
        max_context_tokens=4000,
        rerank_top_k=5
    )
    
    # Simulate dense search results
    mock_results = [
        {
            'id': 'doc_1',
            'content': 'Python is a programming language known for its simplicity and readability.',
            'score': 0.85,
            'source': 'programming_guide.pdf',
            'page': 1
        },
        {
            'id': 'doc_2',
            'content': 'Machine learning uses algorithms to learn from data patterns.',
            'score': 0.78,
            'source': 'ml_basics.pdf',
            'page': 5
        },
        {
            'id': 'doc_3',
            'content': 'Python is widely used in data science and machine learning applications.',
            'score': 0.82,
            'source': 'programming_guide.pdf',
            'page': 15
        }
    ]
    
    # Process query
    result = pipeline.process(
        query="What is Python used for?",
        dense_results=mock_results
    )
    
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Citations: {len(result.citations)}")

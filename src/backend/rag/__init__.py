"""
RAG (Retrieval-Augmented Generation) Module
============================================

Provides enhanced RAG capabilities for document analysis.

Components:
- EnhancedRAGPipeline: Main RAG pipeline with reranking
- QueryExpander: Query expansion with synonyms
- BM25Scorer: Sparse retrieval scoring
- ReRanker: Cross-encoder style reranking
- ContextCompressor: Context length management
- CitationTracker: Source citation tracking
- ConfidenceScorer: Answer confidence scoring
"""

from .enhanced_rag_pipeline import (
    EnhancedRAGPipeline,
    QueryExpander,
    BM25Scorer,
    ReRanker,
    ContextCompressor,
    CitationTracker,
    ConfidenceScorer,
    RetrievedChunk,
    RAGResult,
    create_enhanced_rag_pipeline,
)

__all__ = [
    'EnhancedRAGPipeline',
    'QueryExpander',
    'BM25Scorer',
    'ReRanker',
    'ContextCompressor',
    'CitationTracker',
    'ConfidenceScorer',
    'RetrievedChunk',
    'RAGResult',
    'create_enhanced_rag_pipeline',
]

"""RAG (Retrieval-Augmented Generation) Module — Nexus LLM Analytics
====================================================================

Provides enhanced RAG capabilities for research-grade document
analysis and question answering.

Components
----------
EnhancedRAGPipeline
    Main pipeline with hybrid search, re-ranking, and citation tracking.
QueryExpander
    Synonym-based query expansion for improved recall.
BM25Scorer / ReRanker
    Sparse retrieval scoring and cross-encoder re-ranking.
ContextCompressor / CitationTracker / ConfidenceScorer
    Context management, source attribution, and answer confidence.

v2.0 Enterprise Additions
-------------------------
* Enterprise module docstring with component catalogue.
"""

from __future__ import annotations

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

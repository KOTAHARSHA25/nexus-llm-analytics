# RAG Agent Plugin
# Consolidated Document Analysis Plugin replacing RAGHandler
# Phase 3+: Now integrates EnhancedRAGPipeline for research-grade features

import sys
import logging
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

# Configure logger for this module
logger = logging.getLogger(__name__)

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from backend.core.plugin_system import BasePluginAgent, AgentMetadata, AgentCapability
from backend.agents.model_initializer import get_model_initializer
from backend.core.chromadb_client import ChromaDBClient

# Phase 3+: Import enhanced RAG components for better retrieval
try:
    from backend.rag import QueryExpander, ConfidenceScorer, CitationTracker, RetrievedChunk
    ENHANCED_RAG_AVAILABLE = True
except ImportError:
    ENHANCED_RAG_AVAILABLE = False
    logger.warning("Enhanced RAG components not available, using basic RAG")

class RagAgent(BasePluginAgent):
    """
    RAG (Retrieval-Augmented Generation) Agent Plugin.
    Handles unstructured documents: PDF, DOCX, TXT, PPTX.
    
    Phase 3+ Enhanced Features:
    - Query expansion for better recall
    - Confidence scoring for answer quality
    - Citation tracking for source attribution
    """
    
    def get_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="RagAgent",
            version="2.0.0",  # Bumped for enhanced RAG integration
            description="Analyzes unstructured documents (PDF, Docs) using enhanced RAG techniques with citations",
            author="Nexus Team",
            capabilities=[AgentCapability.DOCUMENT_PROCESSING, AgentCapability.DATA_ANALYSIS],
            file_types=[".pdf", ".docx", ".txt", ".pptx", ".rtf"],
            dependencies=["PyPDF2"], # docx etc are optional/handled internally
            priority=80  # High priority for document files
        )
    
    def initialize(self, **kwargs) -> bool:
        self.initializer = get_model_initializer()
        # Phase 3+: Initialize enhanced components if available
        if ENHANCED_RAG_AVAILABLE:
            self._query_expander = QueryExpander()
            self._confidence_scorer = ConfidenceScorer()
            self._citation_tracker = CitationTracker()
        else:
            self._query_expander = None
            self._confidence_scorer = None
            self._citation_tracker = None
        return True
    
    def can_handle(self, query: str, file_type: Optional[str] = None, **kwargs) -> float:
        # High confidence for document types
        if file_type and file_type.lower() in [".pdf", ".docx", ".txt", ".pptx", ".rtf"]:
            return 0.95
        
        # Keyword check for "document", "read", "summarize" if no file type
        query_lower = query.lower()
        if "document" in query_lower or "pdf" in query_lower or "summarize this file" in query_lower:
            return 0.6
            
        return 0.0

    def execute(self, query: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """Execute RAG analysis using Vector Search with enhanced features"""
        try:
            self.initializer.ensure_initialized()
            
            # File handling
            filename = kwargs.get('filename')
            
            # Initialize ChromaDB Client
            chroma_client = ChromaDBClient()
            
            # Phase 3+: Expand query for better recall
            queries_to_search = [query]
            if self._query_expander:
                queries_to_search = self._query_expander.expand(query, max_expansions=3)
                logger.info(f"Query expanded to {len(queries_to_search)} variants")
            
            # 1. Attempt Vector Search (Primary Path)
            logger.info(f"Querying ChromaDB for: {query}")
            
            all_chunks = []
            for q in queries_to_search:
                search_results = chroma_client.query(query_text=q, n_results=5)
                if search_results and search_results['documents'] and search_results['documents'][0]:
                    for i, doc in enumerate(search_results['documents'][0]):
                        if ENHANCED_RAG_AVAILABLE:
                            chunk = RetrievedChunk(
                                chunk_id=f"chunk_{i}",
                                content=doc,
                                score=1.0 - (search_results.get('distances', [[]])[0][i] if search_results.get('distances') else 0),
                                metadata=search_results['metadatas'][0][i] if search_results.get('metadatas') else {},
                                source=filename or "unknown"
                            )
                            all_chunks.append(chunk)
                        else:
                            all_chunks.append(doc)
            
            retrieved_context = ""
            source_mode = "vector_db"
            citations = []
            confidence = 0.0
            
            if all_chunks:
                # Found documents in vector DB
                if ENHANCED_RAG_AVAILABLE and isinstance(all_chunks[0], RetrievedChunk):
                    # Use enhanced chunks with citation tracking
                    retrieved_context = "\n\n---\n\n".join([c.content for c in all_chunks[:5]])
                    citations = self._citation_tracker.generate_citations(query, all_chunks[:5]) if self._citation_tracker else []
                    confidence = self._confidence_scorer.score(query, all_chunks[:5]) if self._confidence_scorer else 0.5
                else:
                    retrieved_context = "\n\n---\n\n".join(all_chunks[:5])
                    confidence = 0.5
                logger.info(f"Retrieved {len(all_chunks)} chunks from ChromaDB (confidence: {confidence:.2f})")
            else:
                # Fallback to direct file reading (Secondary Path)
                logger.warning("No relevant docs found in ChromaDB. Falling back to direct file read.")
                source_mode = "direct_file_read"
                
                file_path = kwargs.get('file_path') or self._resolve_path(filename)
                
                if not file_path or not os.path.exists(file_path):
                     return {
                        "success": False,
                        "error": f"File not found for fallback: {filename}"
                    }

                doc_text = self._extract_text(file_path)
                if not doc_text:
                     return {
                        "success": False,
                        "error": "Could not extract text from document (fallback)"
                    }
                retrieved_context = doc_text

            # 2. Generate Response
            response = self._generate_rag_response(query, retrieved_context)
            
            return {
                "success": True,
                "result": response,
                "metadata": {
                    "agent": "RagAgent",
                    "version": "2.0.0",
                    "source_mode": source_mode,
                    "context_length": len(retrieved_context),
                    "enhanced_rag": ENHANCED_RAG_AVAILABLE,
                    "confidence": confidence,
                    "citations": [c if isinstance(c, dict) else {"source": str(c)} for c in citations[:3]] if citations else [],
                    "chunks_retrieved": len(all_chunks) if all_chunks else 0
                }
            }
            
        except Exception as e:
            logging.error(f"RagAgent execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _resolve_path(self, filename: str) -> Optional[str]:
        if not filename: return None
        # Basic resolution (can be enhanced via Config later)
        base_dir = Path(__file__).parent.parent.parent.parent / "data"
        paths = [
            base_dir / "uploads" / filename,
            base_dir / "samples" / filename,
            base_dir / filename
        ]
        for p in paths:
            if p.exists(): return str(p)
        return None

    def _extract_text(self, file_path: str) -> str:
        """Extract text content from various document formats."""
        file_ext = os.path.splitext(file_path)[1].lower()
        try:
            if file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: return f.read()
            elif file_ext == '.pdf': return self._extract_pdf_text(file_path)
            elif file_ext in ['.docx', '.doc']: return self._extract_docx_text(file_path)
            elif file_ext == '.pptx': return self._extract_pptx_text(file_path)
            elif file_ext == '.rtf': return self._extract_rtf_text(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: return f.read()
        except Exception as e:
            logging.error(f"Text extraction failed for {file_path}: {e}")
            return ""

    def _extract_pdf_text(self, file_path: str) -> str:
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text: text_parts.append(text)
            return "\n\n".join(text_parts)
        except ImportError:
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(file_path)
                return "\n\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
            except ImportError: return ""
        except Exception: return ""

    def _extract_docx_text(self, file_path: str) -> str:
        try:
            from docx import Document
            doc = Document(file_path)
            return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        except Exception: return ""

    def _extract_pptx_text(self, file_path: str) -> str:
        try:
            from pptx import Presentation
            prs = Presentation(file_path)
            text_parts = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        text_parts.append(shape.text)
            return "\n\n".join(text_parts)
        except Exception: return ""

    def _extract_rtf_text(self, file_path: str) -> str:
        try:
            from striprtf.striprtf import rtf_to_text
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return rtf_to_text(f.read())
        except Exception: return ""

    def _generate_rag_response(self, query: str, document_text: str) -> str:
        # Smart Truncation
        max_chars = 12000 # Increased context window
        if len(document_text) > max_chars:
            document_text = document_text[:max_chars] + "\n[Truncated]"
            
        prompt = f"""You are a helpful document analysis assistant.

DOCUMENT CONTEXT (Retrieved Segments):
{document_text}

QUESTION: {query}

Provide a clear, accurate answer based ONLY on the provided context segments above. If the answer is not in the context, say so.
"""
        
        try:
            response = self.initializer.llm_client.generate(prompt=prompt, adaptive_timeout=True)
            if isinstance(response, dict): return response.get('response', str(response))
            return str(response)
        except Exception as e:
            return f"Error generating response: {e}"

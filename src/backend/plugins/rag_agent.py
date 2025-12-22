# RAG Agent Plugin
# Consolidated Document Analysis Plugin replacing RAGHandler

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

class RagAgent(BasePluginAgent):
    """
    RAG (Retrieval-Augmented Generation) Agent Plugin.
    Handles unstructured documents: PDF, DOCX, TXT, PPTX.
    """
    
    def get_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="RagAgent",
            version="1.0.0",
            description="Analyzes unstructured documents (PDF, Docs) using RAG techniques",
            author="Nexus Team",
            capabilities=[AgentCapability.DOCUMENT_PROCESSING, AgentCapability.DATA_ANALYSIS],
            file_types=[".pdf", ".docx", ".txt", ".pptx", ".rtf"],
            dependencies=["PyPDF2"], # docx etc are optional/handled internally
            priority=80  # High priority for document files
        )
    
    def initialize(self, **kwargs) -> bool:
        self.initializer = get_model_initializer()
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
        """Execute RAG analysis using Vector Search"""
        try:
            self.initializer.ensure_initialized()
            
            # File handling
            filename = kwargs.get('filename')
            
            # Initialize ChromaDB Client
            chroma_client = ChromaDBClient()
            
            # 1. Attempt Vector Search (Primary Path)
            logger.info(f"Querying ChromaDB for: {query}")
            search_results = chroma_client.query(query_text=query, n_results=5)
            
            retrieved_context = ""
            source_mode = "vector_db"
            
            if search_results and search_results['documents'] and search_results['documents'][0]:
                # Found documents in vector DB
                docs = search_results['documents'][0]
                retrieved_context = "\n\n---\n\n".join(docs)
                logger.info(f"Retrieved {len(docs)} chunks from ChromaDB")
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
                    "source_mode": source_mode,
                    "context_length": len(retrieved_context)
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

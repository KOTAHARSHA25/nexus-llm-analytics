

# Handles ChromaDB integration for vector storage and retrieval
# Phase 3: Added Hybrid Search (Task 3.2) and Citation Tracking (Task 3.3)
import chromadb
from chromadb.config import Settings
import requests
import logging
import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CitedSource:
    """Phase 3.3: Represents a cited source in RAG responses"""
    source_id: str
    text: str
    filename: str
    chunk_index: int
    relevance_score: float
    chunk_type: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "relevance_score": self.relevance_score,
            "chunk_type": self.chunk_type
        }


@dataclass
class RAGResponse:
    """Phase 3.3: RAG response with source citations"""
    answer: str
    sources: List[CitedSource] = field(default_factory=list)
    confidence: float = 0.0
    query: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "confidence": self.confidence,
            "query": self.query,
            "citation_count": len(self.sources)
        }


class ChromaDBClient:
	def __init__(self, persist_directory="./chroma_db"):
		# Updated for modern ChromaDB API (v0.4+)
		# Use PersistentClient for file-based persistence
		try:
			self.client = chromadb.PersistentClient(
				path=persist_directory,
				settings=Settings(
					anonymized_telemetry=False,
					allow_reset=True
				)
			)
		except Exception as e:
			# Fallback to ephemeral client if persistence fails
			logger.warning(f"PersistentClient failed, using EphemeralClient: {e}")
			self.client = chromadb.EphemeralClient()
		self.collection_name = "nexus_documents"
		self.collection = self.client.get_or_create_collection(self.collection_name)
		
		# Phase 3.2: Hybrid search configuration
		self.vector_weight = 0.7
		self.keyword_weight = 0.3
		self.stopwords = {
			'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 
			'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
			'this', 'that', 'with', 'from', 'they', 'will', 'would', 'there',
			'their', 'what', 'about', 'which', 'when', 'make', 'like', 'into',
			'just', 'over', 'such', 'than', 'then', 'some', 'could', 'them'
		}

	def add_document(self, doc_id, text, embedding=None, metadata=None):
		# embedding: Optional precomputed embedding vector
		self.collection.add(
			documents=[text],
			ids=[doc_id],
			embeddings=[embedding] if embedding is not None else None,
			metadatas=[metadata] if metadata else None
		)

	def query(self, query_text, n_results=5, embedding=None):
		"""Query the collection with better error handling"""
		try:
			# If embedding is provided, use it; else, rely on Chroma's internal embedding
			return self.collection.query(
				query_texts=[query_text],
				n_results=n_results,
				query_embeddings=[embedding] if embedding is not None else None
			)
		except Exception as e:
			# Return a mock result structure that indicates no documents found
			return {
				'documents': [[]],
				'metadatas': [[]],
				'ids': [[]],
				'distances': [[]],
				'error': f"Query failed: {str(e)}"
			}

	def hybrid_query(self, query_text: str, n_results: int = 5, embedding: Optional[List[float]] = None) -> Dict[str, Any]:
		"""
		Phase 3.2: Hybrid Search - Combine vector similarity with keyword matching.
		
		This approach improves RAG retrieval by:
		1. Getting more candidates via vector search
		2. Re-ranking using keyword overlap
		3. Combining scores with configurable weights
		
		Args:
			query_text: The search query
			n_results: Number of results to return
			embedding: Optional precomputed query embedding
			
		Returns:
			Dictionary with documents, scores, metadatas, and ids
		"""
		try:
			# Vector search (get more candidates for re-ranking)
			vector_results = self.collection.query(
				query_texts=[query_text],
				n_results=min(n_results * 3, 20),  # Get 3x for re-ranking, max 20
				query_embeddings=[embedding] if embedding is not None else None
			)
			
			# Handle empty results
			if not vector_results.get('documents') or not vector_results['documents'][0]:
				return {
					'documents': [[]],
					'scores': [[]],
					'metadatas': [[]],
					'ids': [[]]
				}
			
			# Extract keywords from query
			keywords = self._extract_keywords(query_text)
			
			# Re-rank by combining vector similarity and keyword overlap
			scored_results = []
			documents = vector_results['documents'][0]
			distances = vector_results.get('distances', [[0] * len(documents)])[0]
			metadatas = vector_results.get('metadatas', [[{}] * len(documents)])[0]
			ids = vector_results.get('ids', [[''] * len(documents)])[0]
			
			for i, doc in enumerate(documents):
				# Convert distance to similarity score (lower distance = higher similarity)
				# ChromaDB uses L2 distance, so we convert it
				vector_score = 1.0 / (1 + distances[i]) if distances[i] >= 0 else 0.5
				
				# Calculate keyword overlap score
				keyword_score = self._keyword_overlap(doc, keywords)
				
				# Combined score with configurable weights
				combined_score = (
					self.vector_weight * vector_score + 
					self.keyword_weight * keyword_score
				)
				
				scored_results.append({
					'document': doc,
					'score': combined_score,
					'vector_score': vector_score,
					'keyword_score': keyword_score,
					'metadata': metadatas[i] if i < len(metadatas) else {},
					'id': ids[i] if i < len(ids) else ''
				})
			
			# Sort by combined score (descending)
			scored_results.sort(key=lambda x: x['score'], reverse=True)
			
			# Take top n_results
			top_results = scored_results[:n_results]
			
			return {
				'documents': [[r['document'] for r in top_results]],
				'scores': [[r['score'] for r in top_results]],
				'metadatas': [[r['metadata'] for r in top_results]],
				'ids': [[r['id'] for r in top_results]],
				'debug': {
					'total_candidates': len(documents),
					'keywords_used': list(keywords),
					'vector_weight': self.vector_weight,
					'keyword_weight': self.keyword_weight
				}
			}
			
		except Exception as e:
			logger.error(f"Hybrid query failed: {e}")
			# Fallback to standard query
			return self.query(query_text, n_results, embedding)

	def _extract_keywords(self, text: str) -> Set[str]:
		"""
		Phase 3.2: Extract significant keywords from text.
		Filters out stopwords and short words for better matching.
		"""
		# Find all words with 3+ characters
		words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
		
		# Remove stopwords and return as set
		return set(words) - self.stopwords

	def _keyword_overlap(self, document: str, keywords: Set[str]) -> float:
		"""
		Phase 3.2: Calculate keyword overlap score between document and query keywords.
		
		Returns a score between 0 and 1 based on:
		- What fraction of query keywords appear in the document
		- Bonus for multiple occurrences
		"""
		if not keywords:
			return 0.0
		
		doc_lower = document.lower()
		doc_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', doc_lower))
		
		# Basic overlap
		overlap = keywords & doc_words
		base_score = len(overlap) / len(keywords)
		
		# Bonus for multiple occurrences (up to 0.2 extra)
		occurrence_bonus = 0
		for keyword in overlap:
			count = doc_lower.count(keyword)
			if count > 1:
				occurrence_bonus += min(0.05, (count - 1) * 0.01)
		
		return min(1.0, base_score + occurrence_bonus)

	def query_with_citations(
		self, 
		query_text: str, 
		n_results: int = 5,
		use_hybrid: bool = True
	) -> List[CitedSource]:
		"""
		Phase 3.3: Query with citation tracking.
		Returns structured citation objects for source attribution.
		
		Args:
			query_text: The search query
			n_results: Number of results to return
			use_hybrid: Whether to use hybrid search (recommended)
			
		Returns:
			List of CitedSource objects with full provenance
		"""
		# Use hybrid or standard query
		if use_hybrid:
			results = self.hybrid_query(query_text, n_results)
		else:
			results = self.query(query_text, n_results)
		
		citations = []
		
		if results.get('documents') and results['documents'][0]:
			documents = results['documents'][0]
			metadatas = results.get('metadatas', [[{}] * len(documents)])[0]
			ids = results.get('ids', [[''] * len(documents)])[0]
			scores = results.get('scores', [[0.5] * len(documents)])[0]
			
			for i, doc in enumerate(documents):
				metadata = metadatas[i] if i < len(metadatas) else {}
				
				citation = CitedSource(
					source_id=ids[i] if i < len(ids) else f"source_{i+1}",
					text=doc,
					filename=metadata.get('filename', 'unknown'),
					chunk_index=metadata.get('chunk_index', i),
					relevance_score=scores[i] if i < len(scores) else 0.5,
					chunk_type=metadata.get('chunk_type', 'unknown')
				)
				citations.append(citation)
		
		return citations

	def format_context_with_citations(self, citations: List[CitedSource]) -> str:
		"""
		Phase 3.3: Format retrieved context with citation markers for LLM.
		
		Returns formatted context string with [Source N] markers that the LLM
		can reference in its response for proper attribution.
		"""
		if not citations:
			return ""
		
		formatted_parts = []
		for i, citation in enumerate(citations, 1):
			source_marker = f"[Source {i}]"
			formatted_parts.append(
				f"{source_marker} (from {citation.filename}, relevance: {citation.relevance_score:.2f}):\n"
				f"{citation.text}\n"
			)
		
		return "\n".join(formatted_parts)

	def list_collections(self):
		return self.client.list_collections()

def chunk_text(text, chunk_size=500, overlap=50):
	"""Chunk text into overlapping segments for embedding."""
	words = text.split()
	chunks = []
	i = 0
	while i < len(words):
		chunk = words[i:i+chunk_size]
		chunks.append(' '.join(chunk))
		i += chunk_size - overlap
	return chunks


def embed_text(text, model="nomic-embed-text"):
	"""
	Get embedding vector for text using Ollama's embedding API.
	Returns a list of floats (embedding) or None on failure.
	"""
	url = "http://localhost:11434/api/embeddings"
	payload = {"model": model, "prompt": text}
	try:
		response = requests.post(url, json=payload, timeout=60)
		response.raise_for_status()
		data = response.json()
		if "embedding" in data:
			return data["embedding"]
	except Exception as e:
		logger.debug(f"Embedding failed: {e}")
	return None

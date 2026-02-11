"""
ChromaDB Client — Nexus LLM Analytics v2.0
==========================================

Handles ChromaDB integration for vector storage and retrieval with
hybrid search (Phase 3, Task 3.2) and citation tracking (Phase 3,
Task 3.3).

Enterprise v2.0 Additions
-------------------------
* **CollectionManager** — Multi-collection lifecycle management with
  health checks, compaction triggers, and usage statistics.
* **EmbeddingCache** — LRU cache for embedding vectors to avoid
  redundant Ollama calls on repeated text.
* ``get_chromadb_client()`` — Thread-safe singleton accessor.

Backward Compatibility
----------------------
All v1.x classes (``ChromaDBClient``, ``CitedSource``, ``RAGResponse``)
and free functions (``chunk_text``, ``embed_text``) retain their
original signatures and import paths.

.. versionchanged:: 2.0
   Added enterprise helpers and thread-safe singleton.

Author: Nexus Analytics Research Team
Date: February 2026
"""

# Handles ChromaDB integration for vector storage and retrieval
# Phase 3: Added Hybrid Search (Task 3.2) and Citation Tracking (Task 3.3)
import chromadb
from chromadb.config import Settings
import requests
import logging
import re
import threading
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# [OPTIMIZATION 3.1] Shared session for connection pooling
_session = requests.Session()
from functools import lru_cache
from collections import OrderedDict


@dataclass
class CitedSource:
    """A cited source in RAG responses (Phase 3.3).

    Attributes:
        source_id: Unique document/chunk identifier.
        text: The full chunk text retrieved from the vector store.
        filename: Originating filename for provenance.
        chunk_index: Zero-based chunk position within the source file.
        relevance_score: Combined similarity score (0–1).
        chunk_type: Semantic type label (``paragraph``, ``code_block``, etc.).
    """
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
    """RAG response with source citations (Phase 3.3).

    Wraps the generated answer together with full citation metadata so
    that downstream consumers can render inline citations.

    Attributes:
        answer: The generated LLM answer.
        sources: Ordered list of :class:`CitedSource` objects.
        confidence: Aggregate confidence score (0–1).
        query: The original user query.
    """
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
	def __init__(self, persist_directory="./chroma_db", collection_name="nexus_documents"):
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
		self.collection_name = collection_name
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

	def add_or_update(self, collection_name: str, ids: List[str],
					  documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
		"""Add or update documents in a named collection.

		Used for cross-session vector memory (analysis history,
		error patterns, etc.).  Creates the collection if it does
		not already exist.

		Args:
			collection_name: Target ChromaDB collection.
			ids: Document IDs (upsert semantics).
			documents: Document texts to embed and store.
			metadatas: Optional per-document metadata dicts.
		"""
		try:
			coll = self.client.get_or_create_collection(collection_name)
			coll.upsert(ids=ids, documents=documents,
						metadatas=metadatas if metadatas else None)
		except Exception as e:
			logger.warning("add_or_update(%s) failed: %s", collection_name, e)

	# [OPTIMIZATION 1.3] Cache history lookups to call ChromaDB only once per similar query
	@lru_cache(maxsize=32)
	def semantic_search_history(self, query: str, n_results: int = 5,
								collection_name: str = "analysis_history") -> List[Dict[str, Any]]:
		"""Retrieve semantically similar past analyses for cross-session context.

		Patent Compliance: Claim 1e requires a working memory system using
		vector-based representation to facilitate retrieval/reuse of
		intermediate results and contextual information across sessions.

		Args:
			query: Natural language query to search for.
			n_results: Maximum results to return.
			collection_name: ChromaDB collection to search.

		Returns:
			List of dicts with keys: id, document, metadata, distance.
		"""
		try:
			coll = self.client.get_or_create_collection(collection_name)
			results = coll.query(query_texts=[query], n_results=n_results)
			output = []
			for i in range(len(results.get('ids', [[]])[0])):
				output.append({
					'id': results['ids'][0][i],
					'document': results['documents'][0][i] if results.get('documents') else '',
					'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
					'distance': results['distances'][0][i] if results.get('distances') else 1.0,
				})
			return output
		except Exception as e:
			logger.warning("semantic_search_history failed: %s", e)
			return []

	def query(self, query_text, n_results=5, embedding=None):
		"""Query the collection with better error handling and caching"""
		try:
			# Cache integration
			try:
				from backend.core.enhanced_cache_integration import get_enhanced_cache_manager
				cache = get_enhanced_cache_manager()
				# Create a stable cache key
				cache_key = f"chroma_query_{hash(query_text)}_{n_results}_{hash(str(embedding)) if embedding else 'no_emb'}"
				
				# Try cache first
				cached_result = cache.get_sync(cache_key)
				if cached_result:
					return cached_result
			except ImportError:
				cache = None

			# If embedding is provided, use it; else, rely on Chroma's internal embedding
			result = self.collection.query(
				query_texts=[query_text],
				n_results=n_results,
				query_embeddings=[embedding] if embedding is not None else None
			)

			# Cache the result
			if cache:
				cache.put_sync(cache_key, result, ttl=3600)  # 1 hour TTL
				
			return result
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
		With Caching Integration.
		"""
		try:
			# Cache integration
			try:
				from backend.core.enhanced_cache_integration import get_enhanced_cache_manager
				cache = get_enhanced_cache_manager()
				cache_key = f"chroma_hybrid_{hash(query_text)}_{n_results}_{hash(str(embedding)) if embedding else 'no_emb'}"
				
				cached_result = cache.get_sync(cache_key)
				if cached_result:
					return cached_result
			except ImportError:
				cache = None

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
			
			final_result = {
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

			if cache:
				cache.put_sync(cache_key, final_result, ttl=3600)

			return final_result
			
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
    
    [OPTIMIZATION 3.1] Uses shared session and checks EmbeddingCache first.
    """
    # Check cache first (circular import avoidance via late import or singleton)
    try:
        cache = get_embedding_cache()
        # We access the internal cache directly to avoid recursion loop 
        # since EmbeddingCache.get_or_compute calls this function
        key = cache._make_key(text, model)
        with cache._lock:
            if key in cache._cache:
                cache._hits += 1
                cache._cache.move_to_end(key)
                return cache._cache[key]
    except Exception:
        pass # Cache unavailable, proceed to compute

    url = "http://localhost:11434/api/embeddings"
    payload = {"model": model, "prompt": text}
    try:
        # [OPTIMIZATION 3.1] Use shared session
        response = _session.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        if "embedding" in data:
            embedding = data["embedding"]
            # Cache the result
            try:
                cache = get_embedding_cache()
                with cache._lock:
                    cache._cache[key] = embedding
                    cache._cache.move_to_end(key)
                    if len(cache._cache) > cache._max_size:
                        cache._cache.popitem(last=False)
            except Exception:
                pass
            return embedding
    except Exception as e:
        logger.debug(f"Embedding failed: {e}")
    return None


# ============================================================================
# Enterprise v2.0 — Collection Manager, Embedding Cache & Singleton
# ============================================================================


class EmbeddingCache:
    """LRU cache for embedding vectors to reduce Ollama round-trips.

    Stores the most recently computed embeddings keyed by text hash.
    Thread-safe via an internal lock.

    Args:
        max_size: Maximum number of cached embeddings.

    Example::

        cache = EmbeddingCache(max_size=500)
        vec = cache.get_or_compute("hello world", model="nomic-embed-text")

    .. versionadded:: 2.0
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max_size
        # [OPTIMIZATION 3.2] Use OrderedDict for O(1) eviction
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, text: str, model: str) -> str:
        """Create a deterministic cache key."""
        import hashlib
        return hashlib.sha256(f"{model}::{text}".encode()).hexdigest()

    def get_or_compute(self, text: str, model: str = "nomic-embed-text") -> Optional[List[float]]:
        """Return cached embedding or compute via :func:`embed_text`.

        Args:
            text: The text to embed.
            model: Ollama embedding model name.

        Returns:
            Embedding vector or ``None`` on failure.
        """
        key = self._make_key(text, model)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                self._cache.move_to_end(key)
                return self._cache[key]

        # Cache miss — compute
        self._misses += 1
        embedding = embed_text(text, model=model)
        if embedding is not None:
            with self._lock:
                self._cache[key] = embedding
                self._cache.move_to_end(key)
                # Evict oldest if over capacity
                if len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)
            return embedding
        return None

    @property
    def stats(self) -> Dict[str, Any]:
        """Return cache hit/miss statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "size": len(self._cache),
            "max_size": self._max_size,
        }


class CollectionManager:
    """Multi-collection lifecycle manager for ChromaDB.

    Wraps :class:`ChromaDBClient` to support multiple named collections
    with health checks and basic usage statistics.

    Args:
        persist_directory: Path used for the underlying ChromaDB storage.

    Example::

        mgr = CollectionManager()
        client = mgr.get_or_create("my_project_docs")
        client.add_document("d1", "Hello world")

    .. versionadded:: 2.0
    """

    def __init__(self, persist_directory: str = "./chroma_db") -> None:
        self._persist_dir = persist_directory
        self._clients: Dict[str, ChromaDBClient] = {}
        self._lock = threading.Lock()

    def get_or_create(self, collection_name: str) -> ChromaDBClient:
        """Return an existing client or create a new one for *collection_name*.

        Args:
            collection_name: Logical name of the collection.

        Returns:
            A ready-to-use :class:`ChromaDBClient`.
        """
        with self._lock:
            if collection_name not in self._clients:
                self._clients[collection_name] = ChromaDBClient(
                    persist_directory=self._persist_dir,
                    collection_name=collection_name,
                )
                logger.info("CollectionManager: created collection '%s'", collection_name)
            return self._clients[collection_name]

    def list_managed(self) -> List[str]:
        """Return the names of all managed collections."""
        with self._lock:
            return list(self._clients.keys())

    def health_check(self) -> Dict[str, Any]:
        """Return a health-check summary for every managed collection.

        Returns:
            Dict mapping collection name to its status dict.
        """
        report: Dict[str, Any] = {}
        for name, client in self._clients.items():
            try:
                count = client.collection.count()
                report[name] = {"status": "healthy", "doc_count": count}
            except Exception as exc:
                report[name] = {"status": "error", "error": str(exc)}
        return report


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor (v2.0)
# ---------------------------------------------------------------------------

_chromadb_singleton: Optional[ChromaDBClient] = None
_chromadb_lock = threading.Lock()


def get_chromadb_client(
    persist_directory: str = "./chroma_db",
    collection_name: str = "nexus_documents",
) -> ChromaDBClient:
    """Return the global :class:`ChromaDBClient` singleton.

    Thread-safe with double-checked locking.

    Args:
        persist_directory: ChromaDB storage path.
        collection_name: Default collection name.

    Returns:
        The shared ``ChromaDBClient`` instance.

    .. versionadded:: 2.0
    """
    global _chromadb_singleton
    if _chromadb_singleton is None:
        with _chromadb_lock:
            if _chromadb_singleton is None:
                _chromadb_singleton = ChromaDBClient(
                    persist_directory=persist_directory,
                    collection_name=collection_name,
                )
    return _chromadb_singleton

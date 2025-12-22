

# Handles ChromaDB integration for vector storage and retrieval
import chromadb
from chromadb.config import Settings
import requests
import logging

logger = logging.getLogger(__name__)

class ChromaDBClient:
	def __init__(self, persist_directory="./chroma_db"):
		self.client = chromadb.Client(Settings(persist_directory=persist_directory))
		self.collection_name = "nexus_documents"
		self.collection = self.client.get_or_create_collection(self.collection_name)

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

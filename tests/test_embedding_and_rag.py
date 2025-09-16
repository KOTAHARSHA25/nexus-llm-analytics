# Test Ollama embedding and end-to-end RAG pipeline
import pytest
from backend.core.chromadb_client import embed_text, ChromaDBClient, chunk_text
from backend.agents.rag_agent import RAGAgent

def test_embed_text_returns_vector():
    text = "This is a test for embedding."
    embedding = embed_text(text)
    assert embedding is not None, "Embedding should not be None."
    assert isinstance(embedding, list), "Embedding should be a list."
    assert all(isinstance(x, float) for x in embedding), "Embedding should be a list of floats."
    assert len(embedding) > 0, "Embedding vector should not be empty."


def test_rag_end_to_end_embedding_and_retrieval():
    # Setup: Add a document with real embedding
    chroma = ChromaDBClient(persist_directory="./test_chroma_db")
    text = "Ollama embedding RAG test document. This chunk is for end-to-end retrieval."
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    for i, chunk in enumerate(chunks):
        emb = embed_text(chunk)
        chroma.add_document(doc_id=f"ragembed_{i}", text=chunk, embedding=emb)
    # Test RAGAgent retrieval uses embedding
    agent = RAGAgent(persist_directory="./test_chroma_db")
    results = agent.retrieve("end-to-end retrieval", n_results=2)
    print("RAGAgent Results:", results)
    assert len(results) > 0 and any("retrieval" in r["text"] for r in results)
    print("RAG end-to-end embedding and retrieval test passed.")

if __name__ == "__main__":
    pytest.main([__file__])

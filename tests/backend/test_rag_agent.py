# Test RAGAgent retrieval from ChromaDB

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.agents.rag_agent import RAGAgent
from backend.core.chromadb_client import ChromaDBClient, chunk_text

def test_rag_agent_retrieve():
    # Setup: Add some chunks to ChromaDB
    chroma = ChromaDBClient(persist_directory="./test_chroma_db")
    text = "RAG Agent test document. This chunk is for retrieval testing."
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    for i, chunk in enumerate(chunks):
        chroma.add_document(doc_id=f"ragtest_{i}", text=chunk)
    # Test RAGAgent retrieval
    agent = RAGAgent(persist_directory="./test_chroma_db")
    results = agent.retrieve("retrieval testing", n_results=2)
    print("RAGAgent Results:", results)
    assert len(results) > 0 and any("retrieval" in r["text"] for r in results)
    print("RAGAgent retrieval test passed.")

if __name__ == "__main__":
    test_rag_agent_retrieve()

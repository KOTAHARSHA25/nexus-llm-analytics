# Test ControllerAgent routing to RAGAgent for unstructured data
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.agents.controller_agent import ControllerAgent
from backend.core.chromadb_client import ChromaDBClient, chunk_text

def test_controller_rag_routing():
    # Setup: Add some chunks to ChromaDB
    chroma = ChromaDBClient(persist_directory="./test_chroma_db")
    text = "Controller RAG routing test. This chunk is for PDF/TXT retrieval."
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    for i, chunk in enumerate(chunks):
        chroma.add_document(doc_id=f"ctrlrag_{i}", text=chunk)
    # Test ControllerAgent routing for PDF file
    controller = ControllerAgent()
    results = controller.handle_query("retrieval", filename="test.pdf", n_results=2)
    print("Controller RAG Results:", results)
    assert isinstance(results, list) and any("retrieval" in r["text"] for r in results)
    print("ControllerAgent RAG routing test passed.")

if __name__ == "__main__":
    test_controller_rag_routing()

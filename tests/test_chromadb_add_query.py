# Test ChromaDB add and query functionality

# Allow running as a script by fixing sys.path
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.core.chromadb_client import ChromaDBClient, chunk_text

def test_chromadb_add_and_query():
    client = ChromaDBClient(persist_directory="./test_chroma_db")
    text = "This is a test document for ChromaDB. It contains several words for testing retrieval."
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    # Add each chunk as a separate document
    for i, chunk in enumerate(chunks):
        client.add_document(doc_id=f"test_{i}", text=chunk)
    # Query for a word in the text
    results = client.query(query_text="retrieval", n_results=2)
    print("Query Results:", results)
    assert 'documents' in results and len(results['documents'][0]) > 0
    print("ChromaDB add and query test passed.")

if __name__ == "__main__":
    test_chromadb_add_and_query()

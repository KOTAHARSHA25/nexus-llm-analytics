# Test ChromaDB client initialization and collection creation
from backend.core.chromadb_client import ChromaDBClient

def test_chromadb_client_init():
    client = ChromaDBClient(persist_directory="./test_chroma_db")
    assert client.collection_name == "nexus_documents"
    assert client.collection is not None
    print("ChromaDB client initialized and collection created successfully.")

if __name__ == "__main__":
    test_chromadb_client_init()

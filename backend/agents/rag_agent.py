# CrewAI RAG Agent: Retrieves and summarizes unstructured data from ChromaDB


from backend.core.chromadb_client import ChromaDBClient

class RAGAgent:
    """Retrieval-Augmented Generation agent for unstructured data."""
    def __init__(self, persist_directory="./test_chroma_db"):
        self.chroma = ChromaDBClient(persist_directory=persist_directory)

    def retrieve(self, query, n_results=3):
        """Retrieve top matching chunks from ChromaDB for a given query."""
        results = self.chroma.query(query_text=query, n_results=n_results)
        # Format results for downstream use
        docs = results.get('documents', [[]])[0]
        ids = results.get('ids', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0] if 'metadatas' in results else [{}]*len(docs)
        return [
            {"id": id_, "text": doc, "metadata": meta}
            for id_, doc, meta in zip(ids, docs, metadatas)
        ]

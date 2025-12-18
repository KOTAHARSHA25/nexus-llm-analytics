# CrewAI RAG Agent: Retrieves and summarizes unstructured data from ChromaDB



from backend.core.chromadb_client import ChromaDBClient
from backend.core.llm_client import LLMClient

class RAGAgent:
    """Retrieval-Augmented Generation agent for unstructured data."""
    def __init__(self, persist_directory="./test_chroma_db", llm_client=None):
        self.chroma = ChromaDBClient(persist_directory=persist_directory)
        self.llm_client = llm_client or LLMClient()

    def retrieve(self, query, n_results=3):
        """Retrieve top matching chunks from ChromaDB for a given query, then use LLM to summarize."""
        results = self.chroma.query(query_text=query, n_results=n_results)
        docs = results.get('documents', [[]])[0]
        ids = results.get('ids', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0] if 'metadatas' in results else [{}]*len(docs)
        # Use LLM to summarize the retrieved docs
        context = "\n\n".join(docs)
        prompt = f"Given the following context from a document, answer the user's query: '{query}'.\n\nContext:\n{context}"
        llm_result = self.llm_client.generate_primary(prompt)
        summary = llm_result.get("response")
        return {
            "query": query,
            "summary": summary,
            "chunks": [
                {"id": id_, "text": doc, "metadata": meta}
                for id_, doc, meta in zip(ids, docs, metadatas)
            ]
        }

import pytest
from unittest.mock import MagicMock, patch
from src.backend.core.chromadb_client import (
    ChromaDBClient, CitedSource, RAGResponse, chunk_text, embed_text
)

@pytest.fixture
def mock_chroma():
    with patch('src.backend.core.chromadb_client.chromadb') as mock:
        mock.PersistentClient.return_value.get_or_create_collection.return_value = MagicMock()
        yield mock

@pytest.fixture
def client(mock_chroma):
    return ChromaDBClient()

def test_init_fallback(mock_chroma):
    # Simulate PersistentClient failure
    mock_chroma.PersistentClient.side_effect = Exception("Fail")
    client = ChromaDBClient()
    mock_chroma.EphemeralClient.assert_called()

def test_add_document(client):
    client.collection.add = MagicMock()
    client.add_document("id1", "text", [0.1], {"meta": "data"})
    client.collection.add.assert_called_once()

def test_keyword_logic(client):
    text = "The quick brown fox jumps over the lazy dog."
    client.stopwords = {"the", "over"}
    keywords = client._extract_keywords(text)
    assert "quick" in keywords
    assert "brown" in keywords
    assert "the" not in keywords

    score = client._keyword_overlap("quick brown fox", {"quick", "fox", "missing"})
    # 2 matches out of 3 keywords = 0.66 + bonus
    assert score > 0.0

def test_hybrid_query(client):
    # Mock vector results
    client.collection.query.return_value = {
        'documents': [["doc1", "doc2"]],
        'distances': [[0.1, 0.5]], # doc1 closer
        'metadatas': [[{"id": 1}, {"id": 2}]],
        'ids': [["1", "2"]]
    }
    
    # Query with keywords that favor doc2
    # doc2 has "keyword"
    result = client.hybrid_query("keyword query")
    
    assert 'documents' in result
    assert len(result['documents'][0]) > 0

def test_query_with_citations(client):
    client.hybrid_query = MagicMock(return_value={
        'documents': [["doc1 text"]],
        'metadatas': [[{"filename": "f1.txt"}]],
        'ids': [["id1"]],
        'scores': [[0.9]]
    })
    
    citations = client.query_with_citations("query")
    assert len(citations) == 1
    assert citations[0].filename == "f1.txt"
    assert citations[0].text == "doc1 text"

def test_format_citations(client):
    c = CitedSource("id", "text content", "f1", 0, 0.9)
    formatted = client.format_context_with_citations([c])
    assert "[Source 1]" in formatted
    assert "f1" in formatted
    assert "text content" in formatted

def test_chunk_text():
    text = "word " * 100
    chunks = chunk_text(text, chunk_size=10, overlap=0)
    assert len(chunks) > 0

def test_embed_text():
    with patch('src.backend.core.chromadb_client.requests.post') as mock_post:
        mock_post.return_value.json.return_value = {"embedding": [0.1, 0.2]}
        mock_post.return_value.raise_for_status = MagicMock()
        
        emb = embed_text("text")
        assert emb == [0.1, 0.2]

        # Fail case
        mock_post.side_effect = Exception("Fail")
        assert embed_text("text") is None

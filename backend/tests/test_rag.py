import pytest
from app import rag
from app.core import token_subject, issue_token
from fastapi import HTTPException
from .conftest import account


class Vector:
    def __init__(self, data):
        self.data = data

    def tolist(self):
        return self.data


class FakeEmbeddings:
    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            return Vector([1.0, 0.0] if "alpha" in texts else [0.0, 1.0])
        return Vector([[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts])


def test_vector_filter_and_document_filter(tmp_path, monkeypatch):
    import chromadb
    col = chromadb.PersistentClient(path=str(tmp_path)).get_or_create_collection("private_documents")
    monkeypatch.setattr(rag, "collection", lambda: col)
    monkeypatch.setattr(rag, "embedding_model", lambda: FakeEmbeddings())
    rag.add_chunks("a", "a-doc", "same.txt", [("alpha private A", 1)])
    rag.add_chunks("b", "b-doc", "same.txt", [("alpha private B", 1)])
    assert [item["metadata"]["document_id"] for item in rag.search("alpha", "a", ["a-doc"], rerank=False)] == ["a-doc"]
    assert rag.search("alpha", "a", ["b-doc"], rerank=False) == []
    rag.remove_chunks("a", "a-doc")
    assert rag.search("alpha", "a", ["a-doc"], rerank=False) == []


def test_expired_token(monkeypatch):
    monkeypatch.setattr("app.core.JWT_TTL", -1)
    with pytest.raises(HTTPException):
        token_subject(issue_token("a"))


def test_empty_context_and_history_not_duplicated(client):
    a = account(client, "userone")
    chat = client.post("/chats", json={"title": "Empty"}, headers=a).json()["id"]
    result = client.post("/rag/query", json={"question": "Anything?", "chat_id": chat}, headers=a)
    assert result.status_code == 200
    assert result.json()["sources"] == []
    messages = client.get("/chats/" + chat + "/messages", headers=a).json()
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[-1]["sources"] == []


def test_sources_preserve_metadata():
    item = {"text": "A private chunk", "distance": 0.2, "rerank_score": 0.9,
            "metadata": {"document_id": "d", "filename": "notes.pdf", "page": 2, "chunk_index": 3}}
    assert rag.sources([item]) == [{"document_id": "d", "filename": "notes.pdf", "page": 2,
        "chunk_index": 3, "preview": "A private chunk", "distance": 0.2, "rerank_score": 0.9}]


def test_unavailable_provider_without_key(monkeypatch):
    import asyncio
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="QWEN_API_KEY"):
        asyncio.run(rag.qwen_answer([{"role": "user", "content": "Question"}]))


def test_reranking_changes_order(monkeypatch):
    class Collection:
        def query(self, **kwargs):
            assert kwargs["where"]["$and"][0] == {"owner_id": "a"}
            return {"documents": [["first", "second"]],
                    "metadatas": [[{"owner_id": "a", "document_id": "d", "filename": "one.txt", "page": 0, "chunk_index": 0},
                                   {"owner_id": "a", "document_id": "d", "filename": "one.txt", "page": 0, "chunk_index": 1}]],
                    "distances": [[0.1, 0.2]]}

    class Reranker:
        def predict(self, pairs):
            return [0.1, 0.9]

    monkeypatch.setattr(rag, "collection", lambda: Collection())
    monkeypatch.setattr(rag, "embedding_model", lambda: FakeEmbeddings())
    monkeypatch.setattr(rag, "reranker", lambda: Reranker())
    assert [item["text"] for item in rag.search("alpha", "a", ["d"], rerank=False)] == ["first", "second"]
    assert [item["text"] for item in rag.search("alpha", "a", ["d"], rerank=True)] == ["second", "first"]


def test_api_uses_real_owner_filtered_chroma(client, monkeypatch, tmp_path):
    import chromadb
    collection = chromadb.PersistentClient(path=str(tmp_path / "vectors")).get_or_create_collection("private_documents")
    monkeypatch.setattr(rag, "collection", lambda: collection)
    monkeypatch.setattr(rag, "embedding_model", lambda: FakeEmbeddings())
    a = account(client, "vectoralice")
    b = account(client, "vectorbob")
    doc_a = client.post("/documents", headers=a, files={"file": ("same.txt", b"alpha private A")})
    doc_b = client.post("/documents", headers=b, files={"file": ("same.txt", b"beta private B")})
    assert doc_a.status_code == doc_b.status_code == 201
    id_a, id_b = doc_a.json()["id"], doc_b.json()["id"]
    response_a = client.post("/rag/search", json={"question": "alpha", "rerank": False}, headers=a)
    response_b = client.post("/rag/search", json={"question": "alpha", "rerank": False}, headers=b)
    assert [s["document_id"] for s in response_a.json()["sources"]] == [id_a]
    assert [s["document_id"] for s in response_b.json()["sources"]] == [id_b]
    assert client.post("/rag/search", json={"question": "alpha", "document_ids": [id_b]}, headers=a).status_code == 404
    assert client.delete("/documents/" + id_a, headers=a).status_code == 204
    assert client.post("/rag/search", json={"question": "alpha"}, headers=a).json()["sources"] == []
    assert [s["document_id"] for s in client.post("/rag/search", json={"question": "alpha"}, headers=b).json()["sources"]] == [id_b]

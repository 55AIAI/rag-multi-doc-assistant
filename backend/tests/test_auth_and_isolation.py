from .conftest import account
from app import rag
from app.core import issue_token, token_subject, password_hash, verify_password
from fastapi import HTTPException
import pytest


def test_password_and_tokens():
    stored = password_hash("strong password")
    assert stored != "strong password"
    assert verify_password("strong password", stored)
    assert not verify_password("wrong password", stored)
    assert token_subject(issue_token("user-id")) == "user-id"
    with pytest.raises(HTTPException):
        token_subject("not-a-token")


def test_registration_login_and_resource_isolation(client):
    a = account(client, "alice")
    b = account(client, "bob")
    assert client.post("/auth/login", json={"email": "alice@example.test", "password": "wrong"}).status_code == 401
    assert client.get("/auth/me").status_code == 401
    assert client.get("/auth/me", headers={"Authorization": "Bearer forged"}).status_code == 401
    assert client.get("/auth/me", headers=a).json()["username"] == "alice"
    chat = client.post("/chats", json={"title": "Private"}, headers=a).json()
    assert client.get("/chats", headers=b).json() == []
    assert client.get("/chats/" + chat["id"], headers=b).status_code == 404
    assert client.get("/chats/" + chat["id"] + "/messages", headers=b).status_code == 404
    assert client.delete("/chats/" + chat["id"], headers=b).status_code == 404
    assert client.post("/chats/" + chat["id"] + "/messages", json={"content": "secret"}, headers=b).status_code == 404
    assert client.post("/chats/" + chat["id"] + "/messages", json={"content": "hello"}, headers=a).status_code == 201
    assert len(client.get("/chats/" + chat["id"] + "/messages", headers=a).json()) == 1


def test_upload_filter_delete(client, monkeypatch):
    a = account(client, "alice")
    b = account(client, "bob")
    stored = {}
    monkeypatch.setattr(rag, "add_chunks", lambda owner, did, name, entries: stored.update({did: owner}))
    monkeypatch.setattr(rag, "remove_chunks", lambda owner, did: stored.pop(did) if stored.get(did) == owner else None)

    def fake_search(question, owner, ids, k, rerank, top_k):
        return [dict(text="Alice's answer", metadata={"owner_id": owner, "document_id": did,
                "filename": "notes.txt", "page": 0, "chunk_index": 0}, distance=0.1)
                for did in ids if stored.get(did) == owner]

    monkeypatch.setattr(rag, "search", fake_search)
    assert client.post("/documents", headers=a, files={"file": ("bad.exe", b"bad")}).status_code == 415
    doc = client.post("/documents", headers=a, files={"file": ("notes.txt", b"Alice's answer is 42.")})
    assert doc.status_code == 201, doc.text
    did = doc.json()["id"]
    assert stored[did]
    assert client.get("/documents", headers=b).json() == []
    assert client.get("/documents/" + did, headers=b).status_code == 404
    assert client.delete("/documents/" + did, headers=b).status_code == 404
    assert client.post("/rag/search", json={"question": "answer", "document_ids": [did]}, headers=b).status_code == 404
    assert client.post("/rag/search", json={"question": "answer"}, headers=b).json()["sources"] == []
    result = client.post("/rag/search", json={"question": "answer"}, headers=a).json()
    assert result["sources"][0]["document_id"] == did
    assert client.delete("/documents/" + did, headers=a).status_code == 204
    assert did not in stored
    assert client.post("/rag/search", json={"question": "answer"}, headers=a).json()["sources"] == []


def test_bad_files_and_size_limit(client, monkeypatch):
    a = account(client, "fileuser")
    assert client.post("/documents", headers=a, files={"file": ("fake.pdf", b"not a PDF")}).status_code == 422
    assert client.post("/documents", headers=a, files={"file": ("fake.docx", b"not a ZIP")}).status_code == 422
    assert client.post("/documents", headers=a, files={"file": ("bad.txt", b"\xff\xfe")}).status_code == 422
    monkeypatch.setattr("app.main.MAX_UPLOAD_BYTES", 4)
    assert client.post("/documents", headers=a, files={"file": ("big.txt", b"more than 4")}).status_code == 413
    assert client.get("/documents", headers=a).json() == []


def test_stream_persists_once_and_provider_failure(client, monkeypatch):
    a = account(client, "streamuser")
    chat = client.post("/chats", json={"title": "Stream"}, headers=a).json()["id"]
    empty = client.post("/rag/query/stream", json={"question": "Missing?", "chat_id": chat}, headers=a)
    assert empty.status_code == 200
    assert "event: done" in empty.text
    assert [m["role"] for m in client.get("/chats/" + chat + "/messages", headers=a).json()] == ["user", "assistant"]

    monkeypatch.setattr(rag, "add_chunks", lambda *args: None)
    monkeypatch.setattr(rag, "remove_chunks", lambda *args: None)
    doc = client.post("/documents", headers=a, files={"file": ("a.txt", b"hello world")})
    assert doc.status_code == 201
    did = doc.json()["id"]
    item = {"text": "hello world", "metadata": {"document_id": did, "filename": "a.txt", "page": 0,
            "chunk_index": 0}, "distance": 0.1}
    monkeypatch.setattr(rag, "search", lambda *args: [item])

    async def unavailable(messages):
        raise RuntimeError("provider unavailable")
        yield ""  # The async generator raises before yielding.

    monkeypatch.setattr(rag, "qwen_stream", unavailable)
    failed = client.post("/rag/query/stream", json={"question": "What?", "chat_id": chat}, headers=a)
    assert "event: error" in failed.text
    assert len(client.get("/chats/" + chat + "/messages", headers=a).json()) == 2

    async def successful(messages):
        yield "hello "
        yield "world"

    monkeypatch.setattr(rag, "qwen_stream", successful)
    streamed = client.post("/rag/query/stream", json={"question": "What?", "chat_id": chat}, headers=a)
    assert "event: sources" in streamed.text and "event: done" in streamed.text
    messages = client.get("/chats/" + chat + "/messages", headers=a).json()
    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[-1]["content"] == "hello world"
    assert messages[-1]["sources"][0]["document_id"] == did


def test_partial_index_failure_is_not_searchable(client, monkeypatch):
    a = account(client, "failureuser")
    vectors = set()

    def fail_after_write(owner, did, filename, entries):
        vectors.add(did)
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(rag, "add_chunks", fail_after_write)
    monkeypatch.setattr(rag, "remove_chunks", lambda owner, did: vectors.discard(did))
    response = client.post("/documents", headers=a, files={"file": ("notes.txt", b"Some nonempty text")})
    assert response.status_code == 503
    documents = client.get("/documents", headers=a).json()
    assert len(documents) == 1 and documents[0]["status"] == "failed"
    assert vectors == set()
    assert client.post("/rag/search", json={"question": "text"}, headers=a).json()["sources"] == []
    assert client.delete("/documents/" + documents[0]["id"], headers=a).status_code == 204

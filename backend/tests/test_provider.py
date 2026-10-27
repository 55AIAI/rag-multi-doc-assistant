import asyncio
import httpx
import pytest
from app import rag
from .conftest import account


def test_provider_error_does_not_persist_chat(client, monkeypatch):
    a = account(client, "provideruser")
    chat = client.post("/chats", json={"title": "Provider"}, headers=a).json()["id"]
    monkeypatch.setattr(rag, "add_chunks", lambda *args: None)
    monkeypatch.setattr(rag, "search", lambda question, owner, ids, k, rerank, top_k: [{
        "text": "The answer is 42.", "metadata": {"document_id": ids[0], "filename": "answer.txt",
        "page": 0, "chunk_index": 0}, "distance": 0.1}])
    doc = client.post("/documents", files={"file": ("answer.txt", b"The answer is 42.")}, headers=a)
    assert doc.status_code == 201

    async def failed(messages):
        raise RuntimeError("LLM provider unavailable")

    monkeypatch.setattr(rag, "qwen_answer", failed)
    result = client.post("/rag/query", json={"question": "What is the answer?", "chat_id": chat}, headers=a)
    assert result.status_code == 503
    assert client.get("/chats/" + chat + "/messages", headers=a).json() == []

    received = []
    async def working(messages):
        received.append(messages)
        return "It is 42 [1]."

    monkeypatch.setattr(rag, "qwen_answer", working)
    result = client.post("/rag/query", json={"question": "What is the answer?", "chat_id": chat}, headers=a)
    assert result.status_code == 200
    assert result.json()["sources"][0]["document_id"] == doc.json()["id"]
    assert len([message for message in received[0] if "What is the answer?" in message["content"]]) == 1
    assert [m["role"] for m in client.get("/chats/" + chat + "/messages", headers=a).json()] == ["user", "assistant"]


def test_qwen_bounded_retries(monkeypatch):
    original_client = httpx.AsyncClient
    attempts = []

    def handler(request):
        attempts.append(request)
        if len(attempts) < 3:
            return httpx.Response(429)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Answer [1]"}}]})

    monkeypatch.setattr(rag.httpx, "AsyncClient", lambda **kwargs: original_client(
        transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setenv("QWEN_API_KEY", "test-only")

    async def no_wait(seconds):
        pass

    monkeypatch.setattr(rag.asyncio, "sleep", no_wait)
    answer = asyncio.run(rag.qwen_answer([{"role": "user", "content": "Question"}]))
    assert answer == "Answer [1]"
    assert len(attempts) == 3


def test_qwen_timeout_is_bounded(monkeypatch):
    original_client = httpx.AsyncClient
    attempts = []

    def handler(request):
        attempts.append(request)
        raise httpx.ConnectTimeout("timeout")

    monkeypatch.setattr(rag.httpx, "AsyncClient", lambda **kwargs: original_client(
        transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setenv("QWEN_API_KEY", "test-only")

    async def no_wait(seconds):
        pass

    monkeypatch.setattr(rag.asyncio, "sleep", no_wait)
    with pytest.raises(RuntimeError, match="unavailable"):
        asyncio.run(rag.qwen_answer([{"role": "user", "content": "Question"}]))
    assert len(attempts) == 3


def test_stream_retries_before_first_token(monkeypatch):
    original_client = httpx.AsyncClient
    attempts = []

    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429)
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n')

    monkeypatch.setattr(rag.httpx, "AsyncClient", lambda **kwargs: original_client(
        transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setenv("QWEN_API_KEY", "test-only")

    async def no_wait(seconds):
        pass

    async def collect():
        return [token async for token in rag.qwen_stream([{"role": "user", "content": "Question"}])]

    monkeypatch.setattr(rag.asyncio, "sleep", no_wait)
    assert asyncio.run(collect()) == ["Hello"]
    assert len(attempts) == 2

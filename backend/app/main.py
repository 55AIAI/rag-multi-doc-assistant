"""HTTP boundary. Every private resource is scoped by the validated token."""
import asyncio
import json
import re
import time
import uuid
import zipfile
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import rag
from .core import DATA_DIR, MAX_UPLOAD_BYTES, SessionLocal, current_user, db_session, issue_token, password_hash, verify_password
from .models import Chat, Document, Message, User
from .schemas import Answer, ChatPublic, DocumentPublic, MessagePublic, SearchResult, Token, UserPublic

app = FastAPI(title="Private RAG Assistant", version="1.0.0")


class Registration(BaseModel):
    username: str = Field(min_length=3, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    email: str = Field(min_length=5, max_length=255)
    password: str = Field(min_length=12, max_length=128)


class Login(BaseModel):
    email: str
    password: str


class ChatInput(BaseModel):
    title: str = Field(min_length=1, max_length=150)


class MessageInput(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


class QueryInput(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    chat_id: str | None = None
    document_ids: list[str] | None = None
    k: int = Field(default=10, ge=1, le=30)
    top_k: int = Field(default=3, ge=1, le=10)
    rerank: bool = True


def user_view(user):
    return {"id": user.id, "username": user.username, "email": user.email}


def document_view(doc):
    return {"id": doc.id, "filename": doc.original_filename, "content_type": doc.content_type,
            "file_size": doc.file_size, "status": doc.status, "chunk_count": doc.chunk_count,
            "created_at": doc.created_at}


def chat_view(chat):
    return {"id": chat.id, "title": chat.title, "created_at": chat.created_at}


def message_view(message):
    return {"id": message.id, "role": message.role, "content": message.content,
            "sources": message.source_metadata, "created_at": message.created_at}


def owned(db: Session, model, resource_id: str, owner_id: str):
    result = db.scalar(select(model).where(model.id == resource_id, model.owner_id == owner_id))
    if result is None:
        raise HTTPException(404, "Resource not found")
    return result


@app.post("/auth/register", status_code=201, response_model=UserPublic)
def register(body: Registration, db: Session = Depends(db_session)):
    email = body.email.strip().lower()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        raise HTTPException(422, "Invalid email")
    user = User(username=body.username, email=email, password_hash=password_hash(body.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(409, "Username or email already registered") from exc
    return user_view(user)


@app.post("/auth/login", response_model=Token)
def login(body: Login, db: Session = Depends(db_session)):
    user = db.scalar(select(User).where(User.email == body.email.strip().lower()))
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    return {"access_token": issue_token(user.id), "token_type": "bearer"}


@app.get("/auth/me", response_model=UserPublic)
def me(user: User = Depends(current_user)):
    return user_view(user)


@app.post("/documents", status_code=201, response_model=DocumentPublic)
async def upload(file: UploadFile = File(...), user: User = Depends(current_user), db: Session = Depends(db_session)):
    filename = (file.filename or "").replace("\\", "/").split("/")[-1]
    if not filename or len(filename) > 255 or any(ord(char) < 32 for char in filename):
        raise HTTPException(400, "Invalid filename")
    extension = Path(filename).suffix.lower()
    if extension not in (".pdf", ".docx", ".txt", ".csv"):
        raise HTTPException(415, "Unsupported document format")
    directory = DATA_DIR / "files"
    directory.mkdir(parents=True, exist_ok=True)
    document_id = str(uuid.uuid4())
    path = directory / (document_id + extension)
    size = 0
    try:
        with path.open("xb") as output:
            while block := await file.read(1024 * 1024):
                size += len(block)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "File exceeds size limit")
                output.write(block)
        if size == 0:
            raise HTTPException(422, "Empty file")
        if extension == ".docx" and not zipfile.is_zipfile(path):
            raise HTTPException(422, "Invalid DOCX")
        pages = await asyncio.to_thread(rag.extract, path, extension)
        entries = rag.chunks(pages)
        if not entries:
            raise HTTPException(422, "Document contains no text")
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(422, "Cannot extract text from document") from exc
    doc = Document(id=document_id, owner_id=user.id, original_filename=filename,
                   content_type=file.content_type or "application/octet-stream",
                   file_size=size, storage_path=str(path), status="processing")
    db.add(doc)
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        path.unlink(missing_ok=True)
        raise HTTPException(503, "Document metadata unavailable") from exc
    try:
        await asyncio.to_thread(rag.add_chunks, user.id, document_id, filename, entries)
        doc.chunk_count = len(entries)
        doc.status = "ready"
        db.commit()
    except Exception as exc:
        db.rollback()
        try:
            await asyncio.to_thread(rag.remove_chunks, user.id, document_id)
        except Exception:
            pass  # Failed records are excluded from retrieval; deletion can retry cleanup.
        try:
            persisted = db.get(Document, document_id)
            if persisted:
                persisted.status = "failed"
                db.commit()
        except Exception:
            db.rollback()  # The processing record remains excluded from retrieval.
        raise HTTPException(503, "Document processing failed; delete or retry upload") from exc
    return document_view(doc)


@app.get("/documents", response_model=list[DocumentPublic])
def list_documents(user: User = Depends(current_user), db: Session = Depends(db_session)):
    query = select(Document).where(Document.owner_id == user.id).order_by(Document.created_at.desc())
    return [document_view(doc) for doc in db.scalars(query)]


@app.get("/documents/{document_id}", response_model=DocumentPublic)
def get_document(document_id: str, user: User = Depends(current_user), db: Session = Depends(db_session)):
    return document_view(owned(db, Document, document_id, user.id))


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str, user: User = Depends(current_user), db: Session = Depends(db_session)):
    doc = owned(db, Document, document_id, user.id)
    doc.status = "deleting"
    db.commit()
    try:
        rag.remove_chunks(user.id, doc.id)
        Path(doc.storage_path).unlink(missing_ok=True)
    except Exception as exc:
        raise HTTPException(503, "Deletion incomplete; retry this request") from exc
    try:
        db.delete(doc)
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(503, "Deletion incomplete; retry this request") from exc


@app.post("/chats", status_code=201, response_model=ChatPublic)
def create_chat(body: ChatInput, user: User = Depends(current_user), db: Session = Depends(db_session)):
    chat = Chat(owner_id=user.id, title=body.title)
    db.add(chat)
    db.commit()
    return chat_view(chat)


@app.get("/chats", response_model=list[ChatPublic])
def list_chats(user: User = Depends(current_user), db: Session = Depends(db_session)):
    query = select(Chat).where(Chat.owner_id == user.id).order_by(Chat.created_at.desc())
    return [chat_view(chat) for chat in db.scalars(query)]


@app.get("/chats/{chat_id}", response_model=ChatPublic)
def get_chat(chat_id: str, user: User = Depends(current_user), db: Session = Depends(db_session)):
    return chat_view(owned(db, Chat, chat_id, user.id))


@app.delete("/chats/{chat_id}", status_code=204)
def delete_chat(chat_id: str, user: User = Depends(current_user), db: Session = Depends(db_session)):
    db.delete(owned(db, Chat, chat_id, user.id))
    db.commit()


@app.get("/chats/{chat_id}/messages", response_model=list[MessagePublic])
def list_messages(chat_id: str, limit: int = Query(100, ge=1, le=200), offset: int = Query(0, ge=0),
                  user: User = Depends(current_user), db: Session = Depends(db_session)):
    owned(db, Chat, chat_id, user.id)
    query = (select(Message).where(Message.chat_id == chat_id)
             .order_by(Message.created_at.desc(), Message.id.desc()).offset(offset).limit(limit))
    return [message_view(message) for message in reversed(list(db.scalars(query)))]


@app.post("/chats/{chat_id}/messages", status_code=201, response_model=MessagePublic)
def append_message(chat_id: str, body: MessageInput, user: User = Depends(current_user), db: Session = Depends(db_session)):
    chat = owned(db, Chat, chat_id, user.id)
    message = Message(chat_id=chat.id, role="user", content=body.content)
    db.add(message)
    db.commit()
    return message_view(message)


def query_context(body: QueryInput, owner_id: str, db: Session):
    chat = owned(db, Chat, body.chat_id, owner_id) if body.chat_id else None
    if body.document_ids is not None:
        ids = list(dict.fromkeys(body.document_ids))
        matched = set(db.scalars(select(Document.id).where(
            Document.owner_id == owner_id, Document.status == "ready", Document.id.in_(ids))))
        if set(ids) != matched:
            raise HTTPException(404, "Document not found")
    else:
        ids = list(db.scalars(select(Document.id).where(
            Document.owner_id == owner_id, Document.status == "ready")))
    recent = (list(db.scalars(select(Message).where(Message.chat_id == chat.id)
              .order_by(Message.created_at.desc(), Message.id.desc()).limit(6))) if chat else [])
    history = [{"role": message.role, "content": message.content} for message in reversed(recent)]
    try:
        items = rag.search(body.question, owner_id, ids, body.k, body.rerank, body.top_k) if ids else []
    except Exception as exc:
        raise HTTPException(503, "Retrieval unavailable") from exc
    return items, history


def save_exchange(db: Session, chat_id: str | None, owner_id: str, question: str, answer: str, references: list):
    if not chat_id:
        return
    owned(db, Chat, chat_id, owner_id)
    db.add_all([Message(chat_id=chat_id, role="user", content=question),
                Message(chat_id=chat_id, role="assistant", content=answer, source_metadata=references)])
    db.commit()


@app.post("/rag/search", response_model=SearchResult)
def search_only(body: QueryInput, user: User = Depends(current_user), db: Session = Depends(db_session)):
    started = time.perf_counter()
    items, _ = query_context(body, user.id, db)
    return {"sources": rag.sources(items), "retrieval_ms": round((time.perf_counter() - started) * 1000, 2)}


@app.post("/rag/query", response_model=Answer)
async def query(body: QueryInput, user: User = Depends(current_user), db: Session = Depends(db_session)):
    items, history = query_context(body, user.id, db)
    references = rag.sources(items)
    if not items:
        answer = "I don't know based on the provided documents."
    else:
        try:
            answer = await rag.qwen_answer(rag.llm_messages(body.question, items, history))
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
    save_exchange(db, body.chat_id, user.id, body.question, answer, references)
    return {"answer": answer, "sources": references}


@app.post("/rag/query/stream")
async def stream_query(body: QueryInput, request: Request, user: User = Depends(current_user),
                       db: Session = Depends(db_session)):
    items, history = query_context(body, user.id, db)
    references = rag.sources(items)
    messages = rag.llm_messages(body.question, items, history) if items else []
    owner_id = user.id

    def event(kind, data):
        return "event: " + kind + "\ndata: " + json.dumps(data) + "\n\n"

    async def events():
        yield event("sources", references)
        if not items:
            answer = "I don't know based on the provided documents."
            yield event("token", answer)
        else:
            answer = ""
            try:
                async for token in rag.qwen_stream(messages):
                    if await request.is_disconnected():
                        return
                    answer += token
                    yield event("token", token)
            except (RuntimeError, httpx.HTTPError):
                yield event("error", "LLM provider unavailable")
                return
        if await request.is_disconnected():
            return
        with SessionLocal() as session:
            save_exchange(session, body.chat_id, owner_id, body.question, answer, references)
        yield event("done", {"saved": bool(body.chat_id)})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready")
def ready(db: Session = Depends(db_session)):
    try:
        db.execute(text("SELECT 1"))
        rag.collection().count()
    except Exception as exc:
        raise HTTPException(503, "Dependencies unavailable") from exc
    return {"status": "ready"}

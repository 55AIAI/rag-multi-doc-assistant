"""Public API response contracts."""
from datetime import datetime
from pydantic import BaseModel


class UserPublic(BaseModel):
    id: str
    username: str
    email: str


class Token(BaseModel):
    access_token: str
    token_type: str


class DocumentPublic(BaseModel):
    id: str
    filename: str
    content_type: str
    file_size: int
    status: str
    chunk_count: int
    created_at: datetime


class ChatPublic(BaseModel):
    id: str
    title: str
    created_at: datetime


class Source(BaseModel):
    document_id: str
    filename: str
    page: int | None
    chunk_index: int
    preview: str
    distance: float | None = None
    rerank_score: float | None = None


class MessagePublic(BaseModel):
    id: str
    role: str
    content: str
    sources: list[Source]
    created_at: datetime


class Answer(BaseModel):
    answer: str
    sources: list[Source]


class SearchResult(BaseModel):
    sources: list[Source]
    retrieval_ms: float

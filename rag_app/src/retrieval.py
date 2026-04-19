import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config.settings import (
    CHROMA_PATH, EMBEDDING_MODEL,
    RERANKER_MODEL, RETRIEVAL_K, RERANKER_TOP_K,
)
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Singletons — initialised once per process
_embeddings = None
_db         = None
_reranker   = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_db() -> Chroma:
    global _embeddings, _db
    if _db is None:
        if not Path(CHROMA_PATH).exists():
            raise FileNotFoundError(
                f"No vector database at '{CHROMA_PATH}'. Upload a document first."
            )
        logger.info("Loading embedding model and Chroma DB...")
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _db = Chroma(persist_directory=CHROMA_PATH, embedding_function=_embeddings)
    return _db


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker model: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def reset_db() -> None:
    """Invalidate DB singleton — call after ingestion so next query reloads it."""
    global _db
    _db = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    k: int = RETRIEVAL_K,
    selected_sources: list[str] = None,
    rerank: bool = True,
    top_k_after_rerank: int = RERANKER_TOP_K,
) -> list[Document]:
    """
    Retrieve relevant chunks for *query*.

    Pipeline:
        1. Chroma similarity search → k candidates (with optional source filter)
        2. CrossEncoder reranker   → sorted by relevance score
        3. Return top_k_after_rerank final docs

    Args:
        query:               User question.
        k:                   Candidate pool size from Chroma.
        selected_sources:    Raw source values to filter on.
                             None/[] = all docs.
                             1 value = exact match.
                             2+ values = $in filter.
        rerank:              Whether to run the CrossEncoder reranker.
        top_k_after_rerank:  Final number of docs returned after reranking.
    """
    db = _get_db()

    # Build Chroma filter
    chroma_filter = None
    if selected_sources:
        if len(selected_sources) == 1:
            chroma_filter = {"source": selected_sources[0]}
        else:
            chroma_filter = {"source": {"$in": selected_sources}}

    search_kwargs: dict = {"k": k}
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter

    candidates = db.similarity_search(query, **search_kwargs)

    if not candidates:
        return []

    if not rerank or len(candidates) <= 1:
        return candidates[:top_k_after_rerank]

    # Rerank: score each (query, chunk) pair with CrossEncoder
    reranker  = _get_reranker()
    pairs     = [(query, doc.page_content) for doc in candidates]
    scores    = reranker.predict(pairs)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k_after_rerank]]


def list_documents() -> dict:
    """Return {raw_source: chunk_count} as stored in Chroma."""
    db = _get_db()
    result = db._collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in result.get("metadatas", []):
        raw = meta.get("source", "")
        if raw:
            counts[raw] = counts.get(raw, 0) + 1
    return counts


def get_document_sample(source: str, k: int = 3) -> list[Document]:
    """Return k chunks for *source* via direct metadata query (no embedding search)."""
    db = _get_db()
    result = db._collection.get(
        where={"source": source},
        limit=k,
        include=["documents", "metadatas"],
    )
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(result.get("documents", []), result.get("metadatas", []))
    ]


def get_all_chunks(source: str = None) -> list[Document]:
    """
    Return ALL chunks — optionally filtered by source.
    Used by the Chunks viewer tab.

    Note: chromadb 0.4.x does NOT accept "ids" in the include list.
    IDs are always returned as a top-level key regardless of include.
    """
    db = _get_db()
    kwargs: dict = {"include": ["documents", "metadatas"]}
    if source:
        kwargs["where"] = {"source": source}
    result = db._collection.get(**kwargs)

    # ids are returned at top level, not inside include
    ids       = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    chunks = []
    for doc_id, text, meta in zip(ids, documents, metadatas):
        if not text:
            continue
        doc = Document(page_content=text, metadata=dict(meta or {}))
        doc.metadata["_id"] = doc_id
        chunks.append(doc)
    return chunks

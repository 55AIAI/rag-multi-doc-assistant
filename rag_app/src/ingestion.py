import sys
import logging
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config.settings import CHROMA_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model...")
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def load_document(file_path: Path, file_name: str) -> list:
    """Load file with the correct loader and stamp normalised metadata."""
    ext = file_path.suffix.lower()
    loaders = {
        ".pdf":  lambda: PyPDFLoader(str(file_path)),
        ".docx": lambda: Docx2txtLoader(str(file_path)),
        ".txt":  lambda: TextLoader(str(file_path), encoding="utf-8"),
        ".csv":  lambda: CSVLoader(str(file_path)),
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported file type '{ext}' for: {file_name}")

    docs = loaders[ext]().load()
    if not docs:
        raise ValueError(f"No content extracted from: {file_name}")

    for doc in docs:
        doc.metadata["source"]    = file_name
        doc.metadata["file_type"] = ext.lstrip(".")

    return docs


def split_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("Document splitting produced no chunks.")

    # Stamp chunk-level metadata after splitting
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"]   = str(uuid.uuid4())
        chunk.metadata["chunk_size"] = len(chunk.page_content)
        chunk.metadata["chunk_index"] = i

    return chunks


def store_in_chroma(chunks: list) -> None:
    """Add chunks to an existing Chroma DB, or create one if absent."""
    embeddings = _get_embeddings()
    db_path = Path(CHROMA_PATH)

    if db_path.exists() and any(db_path.iterdir()):
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        db.add_documents(chunks)
    else:
        db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_PATH)

    db.persist()


def ingest_file(file_path: Path, file_name: str) -> int:
    """Full pipeline for one file. Returns number of chunks stored."""
    logger.info(f"Ingesting '{file_name}'...")
    docs   = load_document(file_path, file_name)
    chunks = split_documents(docs)
    store_in_chroma(chunks)
    logger.info(f"Stored {len(chunks)} chunks for '{file_name}'.")
    return len(chunks)


def ingest_uploaded_file(uploaded_file) -> int:
    """Accept a Streamlit UploadedFile, persist to temp, ingest, then clean up."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)
    try:
        return ingest_file(tmp_path, uploaded_file.name)
    finally:
        tmp_path.unlink(missing_ok=True)

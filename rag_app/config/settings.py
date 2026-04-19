from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

# Paths
PDF_PATH = BASE_DIR / "data" / "data.pdf"
CHROMA_PATH = str(BASE_DIR / "chroma_db")
CHATS_PATH = str(BASE_DIR / "chats.json")

# Embedding
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Reranker
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_K = 3          # final chunks passed to LLM after reranking
RETRIEVAL_K = 10            # candidates fetched from Chroma before reranking

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# LLM
LLM_MODEL = "qwen-plus"
LLM_TEMPERATURE = 0.2
LLM_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
LLM_TIMEOUT = 60
LLM_RETRIES = 3

# Supported upload formats (Streamlit type list — no dot)
SUPPORTED_FORMATS = ["pdf", "docx", "txt", "csv"]

# API keys
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

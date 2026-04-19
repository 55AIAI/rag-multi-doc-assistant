import sys
import time
import logging
import requests
from pathlib import Path
from typing import Generator

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    QWEN_API_KEY, LLM_MODEL, LLM_TEMPERATURE,
    LLM_API_URL, LLM_TIMEOUT, LLM_RETRIES,
)
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a strict document Q&A assistant. Follow these rules without exception:

1. Answer ONLY using the provided context. Never use outside knowledge.
2. If the answer is not in the context, say exactly: "I don't know based on the provided documents."
3. Do not guess, infer, or extrapolate beyond what the context explicitly states.
4. Always cite which document your answer comes from.

Format every answer exactly as follows:
**Answer:** <one-sentence direct answer>

**Explanation:** <2-3 sentences with supporting detail from the context>

**Key Points:**
- <point 1>
- <point 2>
- <point 3 if applicable>\
"""


def _build_payload(query: str, docs: list, history: list, stream: bool) -> tuple[dict, dict]:
    """Build request headers and JSON payload."""
    context_parts = []
    for doc in docs:
        raw_source = doc.metadata.get("source", "unknown")
        source = Path(raw_source).name
        context_parts.append(f"[Source: {source}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    user_message = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
        "stream": stream,
    }
    return headers, payload


def generate_answer(query: str, docs: list, history: list = None) -> str:
    """Blocking call — returns the full answer string."""
    if not QWEN_API_KEY:
        raise EnvironmentError("QWEN_API_KEY is not set. Check your .env file.")

    headers, payload = _build_payload(query, docs, history or [], stream=False)

    last_error = None
    for attempt in range(1, LLM_RETRIES + 1):
        try:
            response = requests.post(
                LLM_API_URL, headers=headers, json=payload, timeout=LLM_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            last_error = f"Timed out (attempt {attempt}/{LLM_RETRIES})"
            logger.warning(last_error)

        except requests.exceptions.HTTPError as e:
            if response.status_code < 500:
                raise RuntimeError(f"API client error {response.status_code}: {e}") from e
            last_error = f"Server error {response.status_code} (attempt {attempt}/{LLM_RETRIES})"
            logger.warning(last_error)

        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"Unexpected response structure: {response.text[:200]}"
            ) from e

        except requests.exceptions.RequestException as e:
            last_error = f"Request failed (attempt {attempt}/{LLM_RETRIES}): {e}"
            logger.warning(last_error)

        if attempt < LLM_RETRIES:
            wait = 2 ** (attempt - 1)
            logger.info(f"Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Failed after {LLM_RETRIES} attempts. Last error: {last_error}")


def stream_answer(query: str, docs: list, history: list = None) -> Generator[str, None, None]:
    """
    Streaming call — yields text tokens as they arrive from the API.
    Use with Streamlit's st.write_stream() or a manual placeholder loop.
    """
    if not QWEN_API_KEY:
        raise EnvironmentError("QWEN_API_KEY is not set. Check your .env file.")

    headers, payload = _build_payload(query, docs, history or [], stream=True)

    with requests.post(
        LLM_API_URL, headers=headers, json=payload,
        timeout=LLM_TIMEOUT, stream=True
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8")
            if text.startswith("data: "):
                text = text[6:]
            if text.strip() == "[DONE]":
                break
            try:
                import json
                chunk = json.loads(text)
                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    yield token
            except (KeyError, IndexError, ValueError):
                continue

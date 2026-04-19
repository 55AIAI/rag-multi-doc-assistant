import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from src.ingestion import ingest_uploaded_file
from src.retrieval import retrieve, list_documents, get_document_sample, get_all_chunks, reset_db
from src.llm import stream_answer
from config.settings import SUPPORTED_FORMATS, CHATS_PATH

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Persistent chat storage helpers
# ---------------------------------------------------------------------------
CHATS_FILE = Path(CHATS_PATH)


def _load_chats() -> dict:
    """Load chats from JSON. Structure: {username: {chat_name: [messages]}}"""
    if CHATS_FILE.exists():
        try:
            return json.loads(CHATS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_chats(data: dict) -> None:
    CHATS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------
if "chats_data" not in st.session_state:
    st.session_state.chats_data = _load_chats()   # {user: {chat_name: [msgs]}}
if "username" not in st.session_state:
    st.session_state.username = ""
if "active_chat" not in st.session_state:
    st.session_state.active_chat = ""
if "selected_docs" not in st.session_state:
    st.session_state.selected_docs = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _user_chats() -> dict:
    """Return chat dict for the current user, creating it if missing."""
    u = st.session_state.username
    if u not in st.session_state.chats_data:
        st.session_state.chats_data[u] = {}
    return st.session_state.chats_data[u]


def _active_messages() -> list:
    return _user_chats().get(st.session_state.active_chat, [])


def _append_message(role: str, content: str) -> None:
    chat = st.session_state.active_chat
    _user_chats().setdefault(chat, []).append({"role": role, "content": content})
    _save_chats(st.session_state.chats_data)


def _build_doc_maps() -> tuple[dict, dict]:
    """Return (display→raw, display→count). Normalises full paths to basenames."""
    try:
        raw_map = list_documents()
    except Exception:
        raw_map = {}
    display_to_raw: dict[str, str] = {}
    display_counts: dict[str, int] = {}
    for raw, count in raw_map.items():
        display = Path(raw).name
        display_to_raw[display] = raw
        display_counts[display] = display_counts.get(display, 0) + count
    return display_to_raw, display_counts


# ---------------------------------------------------------------------------
# ── Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 RAG Assistant")
    st.divider()

    # ── User login ──────────────────────────────────────────────────────────
    st.subheader("User")
    username_input = st.text_input(
        "Your name:", value=st.session_state.username, placeholder="e.g. Alice"
    )
    if username_input != st.session_state.username:
        st.session_state.username = username_input.strip()
        st.session_state.active_chat = ""
        st.rerun()

    if not st.session_state.username:
        st.info("Enter your name to start.")
        st.stop()

    st.caption(f"Logged in as **{st.session_state.username}**")
    st.divider()

    # ── Chat management ─────────────────────────────────────────────────────
    st.subheader("Chats")
    user_chats = _user_chats()
    chat_names = sorted(user_chats.keys())

    # New chat
    new_chat_name = st.text_input("New chat name:", placeholder="e.g. Project Q&A")
    if st.button("Create Chat", use_container_width=True):
        name = new_chat_name.strip()
        if name and name not in user_chats:
            user_chats[name] = []
            _save_chats(st.session_state.chats_data)
            st.session_state.active_chat = name
            st.rerun()
        elif not name:
            st.warning("Enter a chat name.")
        else:
            st.warning("Chat already exists.")

    # Chat selector
    if chat_names:
        active_idx = chat_names.index(st.session_state.active_chat) \
            if st.session_state.active_chat in chat_names else 0
        selected_chat = st.selectbox(
            "Switch chat:", chat_names, index=active_idx, label_visibility="collapsed"
        )
        if selected_chat != st.session_state.active_chat:
            st.session_state.active_chat = selected_chat
            st.rerun()

        if st.button("Clear Chat", use_container_width=True):
            user_chats[st.session_state.active_chat] = []
            _save_chats(st.session_state.chats_data)
            st.rerun()
    else:
        st.caption("No chats yet — create one above.")

    st.divider()

    # ── Upload ──────────────────────────────────────────────────────────────
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "upload", type=SUPPORTED_FORMATS, accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        if st.button("Process Files", use_container_width=True, type="primary"):
            for uf in uploaded_files:
                with st.spinner(f"Processing {uf.name}…"):
                    try:
                        n = ingest_uploaded_file(uf)
                        st.success(f"**{uf.name}** → {n} chunks")
                    except Exception as e:
                        st.error(f"**{uf.name}**: {e}")
            reset_db()  # invalidate singleton once after all files are ingested
            st.rerun()

    st.divider()

    # ── Document filter ─────────────────────────────────────────────────────
    st.subheader("Document Filter")
    display_to_raw, display_counts = _build_doc_maps()
    all_display = sorted(display_to_raw.keys())

    if not all_display:
        st.caption("No documents indexed yet.")
    else:
        valid_prev = [d for d in st.session_state.selected_docs if d in all_display]
        selected = st.multiselect(
            "Scope (blank = all):",
            options=all_display,
            default=valid_prev,
            placeholder="All Documents",
            label_visibility="collapsed",
        )
        st.session_state.selected_docs = selected
        if selected:
            st.caption(f"Scope: **{len(selected)}** doc(s)")
        else:
            st.caption("Scope: **All Documents**")


# ---------------------------------------------------------------------------
# Guard: must have an active chat to use the tabs
# ---------------------------------------------------------------------------
if not st.session_state.active_chat:
    st.info("Create or select a chat from the sidebar to get started.")
    st.stop()


# ---------------------------------------------------------------------------
# ── Main tabs
# ---------------------------------------------------------------------------
tab_chat, tab_db, tab_chunks = st.tabs(["💬  Chat", "🗄️  Database", "🔍  Chunks"])


# ===========================================================================
# TAB 1 — Chat
# ===========================================================================
with tab_chat:
    selected   = st.session_state.selected_docs
    scope_label = ", ".join(selected) if selected else "All Documents"
    st.subheader(
        f"**{st.session_state.active_chat}**  ·  {st.session_state.username}  ·  {scope_label}",
        divider="gray",
    )

    messages = _active_messages()

    # Render history
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if user_input := st.chat_input("Ask a question about your documents…"):
        _append_message("user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        # Translate display names → raw sources for Chroma filter
        if selected:
            selected_sources = [display_to_raw[d] for d in selected if d in display_to_raw]
        else:
            selected_sources = None

        with st.chat_message("assistant"):

            # Step 1 — Retrieve + Rerank
            with st.spinner("Retrieving and reranking…"):
                try:
                    docs = retrieve(user_input, selected_sources=selected_sources)
                except FileNotFoundError as e:
                    st.error(str(e))
                    st.stop()
                except Exception as e:
                    st.error(f"Retrieval error: {e}")
                    st.stop()

            # Step 2 — Stream answer
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in messages[-6:]
            ]

            answer_placeholder = st.empty()
            full_answer = ""

            try:
                with st.spinner("Generating answer…"):
                    for token in stream_answer(user_input, docs, history=history):
                        full_answer += token
                        answer_placeholder.markdown(full_answer + "▌")
                answer_placeholder.markdown(full_answer)
            except Exception as e:
                st.error(f"LLM error: {e}")
                st.stop()

            # Source citations
            with st.expander(f"Sources ({len(docs)} chunks retrieved)"):
                for i, doc in enumerate(docs):
                    raw_src    = doc.metadata.get("source", "unknown")
                    src        = Path(raw_src).name
                    page       = doc.metadata.get("page", "")
                    file_type  = doc.metadata.get("file_type", "")
                    chunk_id   = doc.metadata.get("chunk_id", "")
                    parts = [f"**{src}**"]
                    if page != "":
                        parts.append(f"page {page}")
                    if file_type:
                        parts.append(file_type.upper())
                    st.caption("  ·  ".join(parts))
                    if chunk_id:
                        st.caption(f"chunk_id: `{chunk_id}`")
                    st.text(doc.page_content[:400])
                    if i < len(docs) - 1:
                        st.divider()

        _append_message("assistant", full_answer)


# ===========================================================================
# TAB 2 — Database Viewer
# ===========================================================================
with tab_db:
    st.subheader("Indexed Documents", divider="gray")
    display_to_raw, display_counts = _build_doc_maps()

    if not display_counts:
        st.info("No documents indexed yet. Upload files via the sidebar.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Documents",    len(display_counts))
        c2.metric("Total chunks", sum(display_counts.values()))
        st.divider()

        for display_name, chunk_count in sorted(display_counts.items()):
            raw_source = display_to_raw[display_name]
            file_type  = Path(display_name).suffix.lstrip(".").upper()
            with st.expander(f"**{display_name}**  ·  {file_type}  ·  {chunk_count} chunks"):
                try:
                    samples = get_document_sample(raw_source, k=3)
                    if samples:
                        for s in samples:
                            st.text(s.page_content[:300])
                            st.divider()
                    else:
                        st.caption("No preview available.")
                except Exception as e:
                    st.caption(f"Preview unavailable: {e}")


# ===========================================================================
# TAB 3 — Chunk Viewer
# ===========================================================================
with tab_chunks:
    st.subheader("Chunk Explorer", divider="gray")
    display_to_raw, display_counts = _build_doc_maps()

    if not display_counts:
        st.info("No documents indexed yet.")
    else:
        # Filter selector
        chunk_filter_options = ["All Documents"] + sorted(display_to_raw.keys())
        chunk_filter = st.selectbox(
            "Filter by document:", chunk_filter_options, key="chunk_filter"
        )

        if st.button("Load Chunks", use_container_width=True):
            raw_filter = None if chunk_filter == "All Documents" else display_to_raw.get(chunk_filter)
            with st.spinner("Loading chunks…"):
                try:
                    chunks = get_all_chunks(source=raw_filter)
                    st.session_state["loaded_chunks"] = chunks
                except Exception as e:
                    st.error(f"Error loading chunks: {e}")

        chunks = st.session_state.get("loaded_chunks", [])
        if chunks:
            st.caption(f"{len(chunks)} chunks loaded")
            st.divider()
            for chunk in chunks:
                src       = Path(chunk.metadata.get("source", "unknown")).name
                chunk_id  = chunk.metadata.get("chunk_id",  chunk.metadata.get("_id", "—"))
                chunk_sz  = chunk.metadata.get("chunk_size", len(chunk.page_content))
                page      = chunk.metadata.get("page", "")
                file_type = chunk.metadata.get("file_type", "")

                header_parts = [f"**{src}**"]
                if page != "":
                    header_parts.append(f"page {page}")
                if file_type:
                    header_parts.append(file_type.upper())

                with st.container(border=True):
                    col_a, col_b = st.columns([3, 1])
                    col_a.markdown("  ·  ".join(header_parts))
                    col_b.caption(f"{chunk_sz} chars")
                    st.caption(f"ID: `{chunk_id}`")
                    st.text(chunk.page_content)

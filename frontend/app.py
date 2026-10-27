"""Streamlit client. All private data is fetched through the authenticated API."""
import json
import os

import httpx
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Private RAG Assistant", page_icon="📚", layout="wide")


def request(method, path, **kwargs):
    headers = {"Authorization": "Bearer " + st.session_state.token} if st.session_state.get("token") else {}
    try:
        with httpx.Client(timeout=180) as client:
            response = client.request(method, API + path, headers=headers, **kwargs)
        if response.status_code >= 400:
            raise ValueError(str(response.json().get("detail", "Request failed")))
        return response.json() if response.content else None
    except httpx.RequestError as exc:
        raise ValueError("Backend is unavailable") from exc


if not st.session_state.get("token"):
    st.title("📚 Private RAG Assistant")
    login_tab, register_tab = st.tabs(["Sign in", "Create account"])
    with login_tab, st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            try:
                st.session_state.token = request("POST", "/auth/login", json={"email": email, "password": password})["access_token"]
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    with register_tab, st.form("register"):
        username = st.text_input("Username")
        email_new = st.text_input("Email", key="register_email")
        password_new = st.text_input("Password (at least 12 characters)", type="password", key="register_password")
        if st.form_submit_button("Create account"):
            try:
                request("POST", "/auth/register", json={"username": username, "email": email_new, "password": password_new})
                st.success("Account created. Sign in to continue.")
            except ValueError as exc:
                st.error(str(exc))
    st.stop()

try:
    profile = request("GET", "/auth/me")
except ValueError:
    st.session_state.clear()
    st.rerun()

with st.sidebar:
    st.title("📚 Documents")
    st.caption("Signed in as " + profile["username"])
    if st.button("Sign out"):
        st.session_state.clear()
        st.rerun()
    st.divider()
    upload = st.file_uploader("Upload PDF, DOCX, TXT or CSV", type=["pdf", "docx", "txt", "csv"])
    if upload and st.button("Process document", type="primary"):
        try:
            result = request("POST", "/documents", files={"file": (upload.name, upload.getvalue(), upload.type)})
            st.success(f"Indexed {result['chunk_count']} chunks")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    try:
        documents = request("GET", "/documents")
        ready = [doc for doc in documents if doc["status"] == "ready"]
        selected = st.multiselect("Search in", [doc["id"] for doc in ready],
            format_func=lambda did: next(doc["filename"] + " (" + did[:8] + ")" for doc in ready if doc["id"] == did))
    except ValueError as exc:
        st.error(str(exc))
        documents, selected = [], []
    st.divider()
    st.subheader("Chats")
    with st.form("new_chat"):
        title = st.text_input("New chat title")
        if st.form_submit_button("Create chat"):
            try:
                st.session_state.chat_id = request("POST", "/chats", json={"title": title})["id"]
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    try:
        chats = request("GET", "/chats")
        ids = [chat["id"] for chat in chats]
        if ids:
            if st.session_state.get("chat_id") not in ids:
                st.session_state.chat_id = ids[0]
            st.session_state.chat_id = st.selectbox("Chat", ids, index=ids.index(st.session_state.chat_id),
                format_func=lambda cid: next(chat["title"] for chat in chats if chat["id"] == cid))
            if st.button("Delete chat"):
                request("DELETE", "/chats/" + st.session_state.chat_id)
                st.session_state.pop("chat_id", None)
                st.rerun()
    except ValueError as exc:
        st.error(str(exc))

st.title("Document assistant")
chat_tab, library_tab = st.tabs(["Chat", "Document library"])
with library_tab:
    if not documents:
        st.info("Upload a document to start.")
    for document in documents:
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown("**" + document["filename"] + "**")
            left.caption(f"{document['status']} · {document['file_size']} bytes · {document['chunk_count']} chunks")
            if right.button("Delete", key="delete_" + document["id"]):
                try:
                    request("DELETE", "/documents/" + document["id"])
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
with chat_tab:
    chat_id = st.session_state.get("chat_id")
    if not chat_id:
        st.info("Create a chat in the sidebar.")
    else:
        try:
            history = request("GET", "/chats/" + chat_id + "/messages")
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        for message in history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["sources"]:
                    with st.expander("Sources"):
                        for source in message["sources"]:
                            st.caption(f"{source['filename']} · page {source['page'] or 'n/a'} · chunk {source['chunk_index']}")
                            st.text(source["preview"])
        if question := st.chat_input("Ask about your documents"):
            with st.chat_message("user"):
                st.markdown(question)
            payload = {"question": question, "chat_id": chat_id, "document_ids": selected or None}
            with st.chat_message("assistant"):
                answer_box = st.empty()
                answer = ""
                references = []
                error = None
                try:
                    with httpx.Client(timeout=180) as client, client.stream(
                            "POST", API + "/rag/query/stream", json=payload,
                            headers={"Authorization": "Bearer " + st.session_state.token}) as response:
                        response.raise_for_status()
                        kind = ""
                        for line in response.iter_lines():
                            if line.startswith("event: "):
                                kind = line[7:]
                            elif line.startswith("data: "):
                                data = json.loads(line[6:])
                                if kind == "sources":
                                    references = data
                                elif kind == "token":
                                    answer += data
                                    answer_box.markdown(answer + "▌")
                                elif kind == "error":
                                    error = data
                    answer_box.markdown(answer)
                    if error:
                        st.error(error)
                    elif references:
                        with st.expander("Sources"):
                            for source in references:
                                st.caption(f"{source['filename']} · page {source['page'] or 'n/a'}")
                                st.text(source["preview"])
                except (httpx.HTTPError, ValueError) as exc:
                    st.error(str(exc))

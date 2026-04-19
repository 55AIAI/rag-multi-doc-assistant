# 📄 RAG Multi-Document Assistant

A production-ready Retrieval-Augmented Generation (RAG) application that allows users to chat with multiple documents using semantic search and LLM-powered answers.

---

## 🚀 Features

* 📂 Multi-file upload (PDF, DOCX, TXT, CSV)
* 🔍 Semantic search with ChromaDB
* 🧠 Local embeddings (HuggingFace)
* ⚡ Reranking for high retrieval accuracy
* 💬 Multi-chat system with persistent storage
* 🎯 Document filtering (query specific files)
* 📊 Database viewer (see indexed documents)
* 🔍 Chunk explorer (debug and inspect chunks)
* 🤖 LLM answers powered by Qwen API
* 🧾 Source citations for transparency
* 🔄 Streaming responses (ChatGPT-like UX)

---

## 🏗 Architecture

```
User Query
   ↓
Embedding Search (ChromaDB)
   ↓
Reranker (CrossEncoder)
   ↓
Top-K Relevant Chunks
   ↓
LLM (Qwen API)
   ↓
Answer + Sources
```

---

## 📁 Project Structure

```
rag_app/
│
├── app/                # Streamlit UI
├── src/
│   ├── ingestion.py   # Document processing
│   ├── retrieval.py   # Retrieval + reranking
│   ├── llm.py         # LLM interaction
│
├── chroma_db/         # Vector database (ignored in git)
├── data/              # Uploaded documents
├── chats.json         # Chat history (ignored)
├── .env               # API keys (ignored)
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Clone repo

```bash
git clone https://github.com/YOUR_USERNAME/rag-multi-doc-assistant.git
cd rag-multi-doc-assistant
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create `.env` file:

```env
QWEN_API_KEY=your_api_key_here
```

---

## ▶️ Run the app

```bash
streamlit run app/app.py
```

---

## 🧠 How It Works

1. Documents are uploaded and split into chunks
2. Chunks are embedded using HuggingFace models
3. Stored in ChromaDB
4. Query is embedded and matched via similarity search
5. Results are reranked using CrossEncoder
6. Top chunks are sent to Qwen LLM
7. Answer is generated with source grounding

---

## 🔍 Example Use Cases

* Document Q&A
* Research assistant
* Knowledge base search
* Multi-document comparison

---

## 📈 Future Improvements

* Authentication system
* Cloud deployment
* Hybrid search (BM25 + embeddings)
* Evaluation metrics for RAG quality

---

## 🧑‍💻 Author

Built by Bektas Mukhambetov

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!

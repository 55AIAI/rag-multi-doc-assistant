# Architecture

The backend is a modular monolith. FastAPI handles authentication and access checks, SQLAlchemy stores users/documents/chats/messages, a local Chroma collection stores vectors, and a file volume stores originals. Streamlit calls the API with a bearer token, never the database or vector index. Alembic creates the SQL schema.

The upload request validates the bounded file, extracts pages or text, splits chunks, commits a processing record, upserts vectors with deterministic IDs and marks the record ready. Failed indexing marks the record failed and attempts vector cleanup. Search includes ready document IDs owned by the token subject, then repeats the owner filter inside Chroma. Delete first marks deleting, removes vectors and file, then removes metadata; a failed delete can be retried.

Qwen receives only retrieved context plus the question. Both normal and SSE query routes share retrieval and ownership checks. A chat exchange is saved once after the complete answer. No LLM call occurs for empty retrieval. Client disconnects during SSE do not persist partial answers.

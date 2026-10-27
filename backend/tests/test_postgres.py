"""Run with TEST_POSTGRES_URL pointing to a disposable rag_test database."""
import os
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session
from app.core import Base, password_hash
from app.models import Chat, Document, Message, User


@pytest.mark.integration
def test_postgres_models_and_cascade():
    url = os.getenv("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL is not set")
    if make_url(url).database != "rag_test":
        pytest.fail("Integration tests require the disposable rag_test database")
    engine = create_engine(url)
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as db:
            user = User(username="integration_user", email="integration@example.test", password_hash=password_hash("long-test-password"))
            db.add(user)
            db.flush()
            doc = Document(owner_id=user.id, original_filename="test.txt", content_type="text/plain",
                           file_size=4, storage_path="/tmp/test.txt", status="ready")
            chat = Chat(owner_id=user.id, title="Integration")
            db.add_all([doc, chat])
            db.flush()
            db.add(Message(chat_id=chat.id, role="user", content="Hello"))
            db.commit()
            assert db.scalar(select(Document).where(Document.owner_id == user.id)).id == doc.id
            assert len(chat.messages) == 1
            db.delete(chat)
            db.commit()
            assert db.scalars(select(Message).where(Message.chat_id == chat.id)).all() == []
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()

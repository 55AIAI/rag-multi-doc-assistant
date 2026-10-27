import os
os.environ["DATABASE_URL"] = "sqlite+pysqlite://"
os.environ["JWT_SECRET"] = "test-only-signing-key-with-more-than-32-bytes"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.core import Base, db_session
from app.main import app
from app import models  # noqa: F401


@pytest.fixture
def client(monkeypatch, tmp_path):
    engine = create_engine("sqlite+pysqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    maker = sessionmaker(engine, expire_on_commit=False)

    def override():
        with maker() as session:
            yield session

    app.dependency_overrides[db_session] = override
    monkeypatch.setattr("app.main.DATA_DIR", tmp_path)
    monkeypatch.setattr("app.main.SessionLocal", maker)
    with TestClient(app) as api:
        yield api
    app.dependency_overrides.clear()
    engine.dispose()


def account(client, username):
    email = username + "@example.test"
    password = "a-long-test-password"
    assert client.post("/auth/register", json={"username": username, "email": email, "password": password}).status_code == 201
    token = client.post("/auth/login", json={"email": email, "password": password}).json()["access_token"]
    return {"Authorization": "Bearer " + token}

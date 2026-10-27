"""Database and authentication primitives."""
import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL must be configured")
DATA_DIR = Path(os.getenv("DATA_DIR", "./runtime")).resolve()
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
JWT_SECRET = os.getenv("JWT_SECRET", "")
if len(JWT_SECRET.encode()) < 32:
    raise RuntimeError("JWT_SECRET must contain at least 32 bytes")
JWT_TTL = int(os.getenv("JWT_TTL_SECONDS", "3600"))
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def db_session():
    with SessionLocal() as session:
        yield session


def password_hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return "scrypt$16384$8$1$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, expected = stored.split("$")
        if scheme != "scrypt" or (n, r, p) != ("16384", "8", "1"):
            return False
        digest = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=2**14, r=8, p=1)
        return hmac.compare_digest(digest, base64.b64decode(expected))
    except (ValueError, TypeError, binascii.Error):
        return False


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _key() -> bytes:
    if len(JWT_SECRET.encode()) < 32:
        raise RuntimeError("JWT_SECRET must contain at least 32 bytes")
    return JWT_SECRET.encode()


def issue_token(user_id: str) -> str:
    header = _b64(b'{"alg":"HS256","typ":"JWT"}')
    payload = _b64(json.dumps({"sub": user_id, "iss": "rag-assistant", "exp": int(time.time()) + JWT_TTL}).encode())
    signing_input = header + "." + payload
    signature = _b64(hmac.new(_key(), signing_input.encode(), hashlib.sha256).digest())
    return signing_input + "." + signature


def token_subject(token: str) -> str:
    try:
        header, payload, signature = token.split(".")
        expected = hmac.new(_key(), (header + "." + payload).encode(), hashlib.sha256).digest()
        claims = json.loads(_decode(payload))
        metadata = json.loads(_decode(header))
        if (metadata != {"alg": "HS256", "typ": "JWT"} or
                not hmac.compare_digest(expected, _decode(signature)) or
                claims.get("iss") != "rag-assistant" or
                type(claims.get("exp")) is not int or claims["exp"] <= time.time() or
                not isinstance(claims.get("sub"), str)):
            raise ValueError("invalid token")
        return claims["sub"]
    except (ValueError, TypeError, KeyError, UnicodeDecodeError, binascii.Error) as exc:
        raise HTTPException(401, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"}) from exc


bearer = HTTPBearer(auto_error=False)


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer), db=Depends(db_session)):
    from .models import User
    if credentials is None:
        raise HTTPException(401, "Authentication required", headers={"WWW-Authenticate": "Bearer"})
    user = db.get(User, token_subject(credentials.credentials))
    if user is None:
        raise HTTPException(401, "Invalid or expired token")
    return user

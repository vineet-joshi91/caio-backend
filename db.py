# db.py
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text, ForeignKey, text
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session, relationship

# ---------- connection & fallback ----------
def _normalize_db_url(url: str) -> str:
    return url.replace("postgres://", "postgresql://", 1) if url.startswith("postgres://") else url

PRIMARY_URL = _normalize_db_url(os.getenv("DATABASE_URL", "").strip())
FALLBACK_TO_SQLITE = os.getenv("DB_FALLBACK_TO_SQLITE", "1").lower() in ("1", "true", "yes")
PG_CONNECT_TIMEOUT = int(os.getenv("PG_CONNECT_TIMEOUT", "5"))
SQLITE_URL = "sqlite:///./caio.db"

def _make_engine(url: str):
    url = _normalize_db_url(url)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {"connect_timeout": PG_CONNECT_TIMEOUT}
    return create_engine(url, pool_pre_ping=True, pool_recycle=300, future=True, connect_args=connect_args)

engine = _make_engine(PRIMARY_URL or SQLITE_URL)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()

# ---------- models ----------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=True)
    is_admin = Column(Boolean, default=False, nullable=False)
    is_paid = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, nullable=True)

class UsageLog(Base):
    __tablename__ = "usage_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    endpoint = Column(String, default="analyze", index=True)
    status = Column(String, default="ok")
    meta = Column(String, default="")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    title = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True, nullable=False)
    role = Column(String(16), index=True)  # "user" | "assistant" | "system"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    session = relationship("ChatSession", back_populates="messages")

# ---------- init & session ----------
def init_db():
    global engine, SessionLocal
    target = PRIMARY_URL or SQLITE_URL
    try:
        probe = _make_engine(target)
        with probe.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine = probe
        SessionLocal.remove()
        SessionLocal.configure(bind=engine)
    except Exception:
        if not FALLBACK_TO_SQLITE or target.startswith("sqlite"):
            raise
        engine = _make_engine(SQLITE_URL)
        SessionLocal.remove()
        SessionLocal.configure(bind=engine)
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

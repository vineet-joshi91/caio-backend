# db.py
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session

# --- DB URL normalizer (Render/Heroku compatibility) -------------------------
def _normalize_db_url(url: str) -> str:
    # Heroku-style URLs use postgres://; SQLAlchemy expects postgresql://
    return url.replace("postgres://", "postgresql://", 1) if url.startswith("postgres://") else url

DATABASE_URL = _normalize_db_url(os.getenv("DATABASE_URL", "")) or "sqlite:///./caio.db"

# SQLite needs this flag for multi-threaded FastAPI usage
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = scoped_session(sessionmaker(bind=engine, autocommit=False, autoflush=False))
Base = declarative_base()

# --- Models -------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=True)  # kept nullable for legacy imports
    is_admin = Column(Boolean, default=False, nullable=False)
    is_paid = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, nullable=True)

class UsageLog(Base):
    __tablename__ = "usage_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    endpoint = Column(String, default="")
    tokens_used = Column(Integer, default=0)
    status = Column(String, default="success")  # success/error/etc.

# --- Helpers ------------------------------------------------------------------
def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

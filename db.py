# db.py
import os
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session

def _normalize_db_url(url: str) -> str:
    return url.replace("postgres://", "postgresql://", 1) if url.startswith("postgres://") else url

DATABASE_URL = _normalize_db_url(os.getenv("DATABASE_URL", "")) or "sqlite:///./caio.db"

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)

SessionLocal = scoped_session(sessionmaker(bind=engine, autocommit=False, autoflush=False))
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(320), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=True)
    is_admin = Column(Boolean, default=False, nullable=False)
    is_paid = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, nullable=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)

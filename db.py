# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 17:57:01 2025

@author: Vineet
"""

# db.py

import os
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# --- CONFIG ---

# Get your Render DATABASE_URL from env (never hardcode in repo)
POSTGRES_URL = os.environ.get("DATABASE_URL", "postgresql://caio_db_prod_yhi3_user:4uHW7iQgNPKoRXWYZtlvOU99I7Sr2M7H@dpg-d28u383uibrs73dvkc4g-a/caio_db_prod_yhi3")

# For SSL support with Render Postgres (uncomment below if you get SSL errors)
# engine = create_engine(POSTGRES_URL, connect_args={"sslmode": "require"})
engine = create_engine(POSTGRES_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- MODELS ---

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    stripe_customer_id = Column(String, nullable=True)
    is_paid = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class UsageLog(Base):
    __tablename__ = "usage_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    tokens_used = Column(Integer, default=0)
    endpoint = Column(String, default="")
    status = Column(String, default="success")  # success/error/etc.

# --- DB INIT/HELPERS ---

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

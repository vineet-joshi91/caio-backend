# db.py
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 17:57:01 2025
@author: Vineet
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker

# --- CONFIG ---

POSTGRES_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://caio_db_prod_yhi3_user:4uHW7iQgNPKoRXWYZtlvOU99I7Sr2M7H@dpg-d28u383uibrs73dvkc4g-a/caio_db_prod_yhi3",
)

engine = create_engine(POSTGRES_URL, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
Base = declarative_base()

# --- MODELS ---

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    is_admin = Column(Boolean, nullable=False, server_default=text("false"))
    is_active = Column(Boolean, nullable=False, server_default=text("true"))
    is_paid  = Column(Boolean, nullable=False, server_default=text("false"))

    stripe_customer_id = Column(String, nullable=True)

    # NEW — Razorpay Subscriptions tracking
    subscription_id = Column(String, nullable=True)   # e.g., "sub_********"
    plan_status     = Column(String, nullable=True)   # "active" | "created" | "cancelled" | etc.

    created_at = Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))

class UsageLog(Base):
    __tablename__ = "usage_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    tokens_used = Column(Integer, default=0)
    endpoint = Column(String, default="")
    status = Column(String, default="success")

# --- OPTIONAL: capture subscription cancellation reasons ---

class CancellationReason(Base):
    __tablename__ = "cancellation_reasons"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)                 # users.id
    subscription_id = Column(String, nullable=False)          # your User.subscription_id (or local sub id)
    category = Column(String, nullable=False)                 # e.g. 'price', 'value', 'missing_feature', ...
    detail = Column(String, nullable=True)                    # free text
    created_at = Column(DateTime, default=datetime.utcnow)

# --- DB INIT/HELPERS ---

def init_db():
    """Create missing tables (does not add new columns to existing tables)."""
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

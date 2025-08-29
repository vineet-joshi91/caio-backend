# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 15:53:39 2025

@author: Vineet
"""

from sqlalchemy import Column, String, Boolean, DateTime
from sqlalchemy.sql import func
from database import Base  # adjust import if different

class User(Base):
    __tablename__ = "users"

    email = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=True)
    organisation = Column(String, nullable=True)
    is_admin = Column(Boolean, default=False)
    is_paid = Column(Boolean, default=False)

    # New subscription fields
    subscription_id = Column(String, nullable=True)   # Razorpay subscription id
    plan_status     = Column(String, nullable=True)   # "active" | "cancelled" | "trialing" etc.
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

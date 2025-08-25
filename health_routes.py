# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 16:40:57 2025

@author: Vineet
"""

# health_routes.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from db import get_db

router = APIRouter(prefix="/api")

@router.get("/health")
def health():
    # super fast: proves the app is mounted
    return {"ok": True}

@router.get("/ready")
def ready(db: Session = Depends(get_db)):
    # warms DB pool; returns fast after first cold start
    try:
        db.execute(text("SELECT 1"))
        return {"ok": True, "db": "up"}
    except Exception as e:
        # don’t expose internal traces
        return {"ok": False, "db": "down"}

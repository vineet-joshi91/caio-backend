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
def health():  # app is up
    return {"ok": True}

@router.get("/ready")
def ready(db: Session = Depends(get_db)):  # app + DB are up
    try:
        db.execute(text("SELECT 1"))
        return {"ok": True, "db": "up"}
    except Exception:
        return {"ok": False, "db": "down"}

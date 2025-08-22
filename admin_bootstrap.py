# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 18:17:16 2025

@author: Vineet
"""

# admin_bootstrap.py
import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db import get_db, User

router = APIRouter(prefix="/api/admin_bootstrap", tags=["admin"])

BOOT_TOKEN = os.getenv("ADMIN_SETUP_TOKEN")  # set this in Render → Environment

@router.post("/grant")
def grant_admin(email: str, token: str, db: Session = Depends(get_db)):
    if not BOOT_TOKEN or token != BOOT_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    u = db.query(User).filter(User.email == email.lower()).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    if hasattr(u, "is_admin"):
        u.is_admin = True
    db.add(u); db.commit()
    return {"ok": True, "email": u.email, "is_admin": bool(getattr(u, "is_admin", False))}

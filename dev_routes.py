# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 17:08:32 2025

@author: Vineet
"""

# dev_routes.py
from fastapi import APIRouter, HTTPException, Form
import os

router = APIRouter()

BOOTSTRAP_SECRET = os.getenv("BOOTSTRAP_ADMIN_SECRET")  # e.g., "caio-bootstrap-2025-secret"
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")                  # e.g., "vineetpjoshi.71@gmail.com"

def _enabled() -> bool:
    return bool(BOOTSTRAP_SECRET and ADMIN_EMAIL)

@router.post("/force-login")
def force_login(secret: str = Form(...)):
    """
    One-time admin bootstrap.
    Returns an admin JWT for ADMIN_EMAIL *only if* BOOTSTRAP_ADMIN_SECRET matches.
    Disable by removing BOOTSTRAP_ADMIN_SECRET from env and redeploy.
    """
    if not _enabled():
        raise HTTPException(status_code=403, detail="Bootstrap disabled")
    if secret != BOOTSTRAP_SECRET:
        raise HTTPException(status_code=401, detail="Invalid secret")

    try:
        from auth import create_access_token  # import late to avoid boot errors
    except Exception:
        raise HTTPException(status_code=500, detail="Auth not available")

    token = create_access_token(sub=ADMIN_EMAIL)  # type: ignore
    return {"access_token": token, "token_type": "bearer"}

@router.get("/status")
def status():
    return {"bootstrap_enabled": _enabled()}

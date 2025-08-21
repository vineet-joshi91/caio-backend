# main.py  (TOP OF FILE → paste this block over your current header)
import os
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from db import get_db, User
from auth import (
    create_access_token,
    verify_password,
    get_password_hash,
    get_current_user,
)

# --- create app FIRST ---
logger = logging.getLogger("caio")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="CAIO Backend", version="0.1.0")

from signup_routes import router as signup_router
app.include_router(signup_router)


# --- CORS ---
def _origins():
    raw = os.getenv("ALLOWED_ORIGINS")
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return [
        "https://caio-frontend.vercel.app",
        "https://caioai.netlify.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

@app.options("/{path:path}")
def cors_preflight(path: str):
    return JSONResponse({"ok": True})

# --- health ---
@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}

# --- import routers AFTER app exists ---
from routes_public_config import router as public_cfg_router
app.include_router(public_cfg_router, tags=["public"])

# optional routers; keep these guards if some files aren’t deployed yet
try:
    from payment_routes import router as payments_router
    app.include_router(payments_router, prefix="/api/payments", tags=["payments"])
    logger.info("Payments router loaded at /api/payments")
except Exception as e:
    logger.warning(f"Payments router NOT loaded: {e}")

try:
    from contact_routes import router as contact_router
    app.include_router(contact_router)  # exposes POST /api/contact
    logger.info("Contact router loaded at /api/contact")
except Exception as e:
    logger.warning(f"Contact router NOT loaded: {e}")

# --- the rest of your file continues (login/profile/analyze endpoints, etc.) ---

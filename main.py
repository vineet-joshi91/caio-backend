# main.py
import os
import logging
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
from datetime import datetime

from db import get_db, User
from auth import (
    create_access_token,
    verify_password,
    get_password_hash,
    get_current_user,
)

# -----------------------------------------------------------------------------
# App & logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("caio")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="CAIO Backend", version="0.1.0")

# -----------------------------------------------------------------------------
# CORS
# -----------------------------------------------------------------------------
def _origins():
    raw = os.getenv("ALLOWED_ORIGINS")
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    # sensible defaults for your stack
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

# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}

# -----------------------------------------------------------------------------
# Auth: login (auto‑provision demo users; admin via ADMIN_EMAILS)
# -----------------------------------------------------------------------------
ADMIN_EMAILS = {
    e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "vineetpjoshi.71@gmail.com").split(",") if e.strip()
}

@app.post("/api/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    email = form.username.strip().lower()
    password = form.password

    user = db.query(User).filter(User.email == email).first()

    if user:
        # existing user → verify password
        if not verify_password(password, user.hashed_password):
            raise HTTPException(status_code=400, detail="Incorrect email or password")
    else:
        # new user → create Demo account
        user = User(
            email=email,
            hashed_password=get_password_hash(password),
            is_admin=(email in ADMIN_EMAILS),
            is_paid=False,
            created_at=getattr(User, "created_at", None) and datetime.utcnow() or None,  # set if column exists
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    token = create_access_token(sub=user.email)
    return {
        "access_token": token,
        "token_type": "bearer",
        "email": user.email,
        "is_admin": bool(user.is_admin),
        "is_paid": bool(user.is_paid),
    }

# -----------------------------------------------------------------------------
# Profile
# -----------------------------------------------------------------------------
@app.get("/api/profile")
def profile(current_user: User = Depends(get_current_user)):
    return {
        "email": current_user.email,
        "is_admin": bool(current_user.is_admin),
        "is_paid": bool(current_user.is_paid),
        "created_at": getattr(current_user, "created_at", None),
    }

# -----------------------------------------------------------------------------
# Analyze (minimal stub: Demo returns sample; Pro returns ‘no credits’ placeholder)
# Keep this until your real pipeline is wired. Frontend already copes with 402.
# -----------------------------------------------------------------------------
# main.py (only the analyze endpoint, rest stays same)

@app.post("/api/analyze")
async def analyze(
    request: Request,
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
):
    text = (text or "").strip()

    if not current_user.is_paid:
        # Demo mode → return sample result
        return JSONResponse(
            {
                "status": "demo",
                "title": "Demo Mode Result",
                "summary": "This is a sample analysis. Upgrade to Pro for full insights.",
                "tip": "Upload a business document or upgrade your plan to unlock advanced engines."
            },
            status_code=200,
        )

    # Pro but no credits → friendly error
    return JSONResponse(
        {
            "status": "error",
            "title": "Analysis Unavailable",
            "message": "You’ve run out of credits or the AI engine is unavailable right now.",
            "action": "Please add credits or try again later.",
        },
        status_code=402,
    )
# -----------------------------------------------------------------------------
# Mount sub‑routers
# -----------------------------------------------------------------------------
try:
    from payment_routes import router as payments_router  # your file name is payment_routes.py
    app.include_router(payments_router, prefix="/api/payments", tags=["payments"])
    logger.info("Payments router loaded at /api/payments")
except Exception as e:
    logger.warning(f"Payments router NOT loaded: {e}")

try:
    from admin_routes import router as admin_router
    app.include_router(admin_router, prefix="/api/admin", tags=["admin"])
    logger.info("Admin router loaded at /api/admin")
except Exception as e:
    logger.warning(f"Admin router NOT loaded: {e}")

try:
    from dev_routes import router as dev_router
    app.include_router(dev_router, prefix="/api/dev", tags=["dev"])
    logger.info("Dev router loaded at /api/dev")
except Exception as e:
    logger.warning(f"Dev router NOT loaded: {e}")

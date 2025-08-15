# -*- coding: utf-8 -*-
"""
CAIO SaaS API – main.py (Render/Production-ready)
"""

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import timedelta
from typing import Optional
import os
import uvicorn

from db import get_db, init_db, User, UsageLog
from auth import (
    create_user,
    authenticate_user,
    create_access_token,
    decode_access_token,
    get_user_by_email,
    is_admin,
    is_paid,
)

# --- SAFE ADMIN ROUTER INCLUDE (add this in main.py after `app = FastAPI()` and CORS setup) ---
# This will never crash the app if the file is missing or has an import error.
# It logs a warning instead, so your current API keeps running.

import logging
logger = logging.getLogger("uvicorn.error")  # Render/uvicorn logger

try:
    from admin_routes import router as admin_router  # the file we created
    app.include_router(admin_router, prefix="/api/admin", tags=["admin"])
    logger.info("Admin router loaded at /api/admin")
except Exception as e:
    logger.warning(f"Admin router NOT loaded: {e}")
# --- END SAFE ADMIN ROUTER INCLUDE ---

# ---------- CORS ----------

def _parse_allowed_origins() -> list[str]:
    """
    Read ALLOWED_ORIGINS from env as a comma-separated list.
    Falls back to sensible defaults for local dev + Vercel.
    """
    raw = os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost,http://localhost:3000,http://localhost:8501,https://caio-frontend.vercel.app",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]

origins = _parse_allowed_origins()

app = FastAPI(
    title="CAIO SaaS API",
    description="Backend API for CAIO SaaS App",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use leading slash (helps OpenAPI doc references)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")

# ---------- Startup ----------

@app.on_event("startup")
def startup():
    try:
        init_db()  # Ensure DB tables exist on start
        print("DB initialized successfully.")
    except Exception as e:
        print(f"[startup] Database initialization failed: {e}")

# ---------- Helpers ----------

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    payload = decode_access_token(token)
    if not payload or "email" not in payload:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    user = get_user_by_email(db, payload["email"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def log_usage(db: Session, user: User, tokens: int, endpoint: str, status: str = "success"):
    log = UsageLog(
        user_id=user.id,
        tokens_used=tokens,
        endpoint=endpoint,
        status=status,
    )
    db.add(log)
    db.commit()

# ---------- Routes ----------

@app.get("/")
def root_health_check():
    """Root endpoint for platform health checks."""
    return {"status": "ok", "message": "CAIO backend is running."}

@app.get("/api/health")
def api_health_check():
    return {"status": "ok"}

@app.post("/api/signup")
async def signup(
    request: Request,
    email: Optional[str] = None,
    password: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Accepts either:
      - query params:  POST /api/signup?email=..&password=..
      - JSON body:     { "email": "...", "password": "..." }
    """
    if not email or not password:
        try:
            body = await request.json()
            email = email or body.get("email")
            password = password or body.get("password")
        except Exception:
            pass

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    email_l = email.strip().lower()
    if get_user_by_email(db, email_l):
        raise HTTPException(status_code=400, detail="User already exists")

    admin_env = os.environ.get("ADMIN_EMAIL", "vineetpjoshi.71@gmail.com").strip().lower()
    is_admin_flag = (email_l == admin_env)

    _ = create_user(db, email_l, password, is_admin=is_admin_flag)
    return {"message": "Signup successful. Please log in."}

@app.post("/api/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # OAuth2PasswordRequestForm uses 'username' field for email by spec
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token = create_access_token(
        data={
            "user_id": user.id,
            "email": user.email,
            "is_admin": user.is_admin,
            "is_paid": user.is_paid,
        },
        expires_delta=timedelta(days=1),
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/profile")
def get_profile(current_user: User = Depends(get_current_user)):
    return {
        "email": current_user.email,
        "is_admin": current_user.is_admin,
        "is_paid": current_user.is_paid,
        "created_at": str(current_user.created_at),
    }

@app.post("/api/analyze")
async def analyze(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not (current_user.is_paid or current_user.is_admin):
        raise HTTPException(status_code=402, detail="Payment required for full analysis access.")

    data = await request.json()
    text = (data.get("text") or "").strip()
    which_brains = data.get("brains", ["CFO", "COO", "CMO", "CHRO"])

    if not text:
        raise HTTPException(status_code=400, detail="Please provide some text to analyze.")

    # Lazy import to keep import-time light
    from brains import analyze_cfo, analyze_cmo, analyze_coo, analyze_chro

    responses = {}
    tokens_used = 0

    if "CFO" in which_brains:
        result = analyze_cfo(text)
        responses["CFO"] = result
        tokens_used += len(result) // 4
    if "COO" in which_brains:
        result = analyze_coo(text)
        responses["COO"] = result
        tokens_used += len(result) // 4
    if "CMO" in which_brains:
        result = analyze_cmo(text)
        responses["CMO"] = result
        tokens_used += len(result) // 4
    if "CHRO" in which_brains:
        result = analyze_chro(text)
        responses["CHRO"] = result
        tokens_used += len(result) // 4

    log_usage(db, current_user, tokens=tokens_used, endpoint="analyze")
    return {"insights": responses, "tokens_used": tokens_used}

@app.get("/api/admin")
def admin_dashboard(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin access required")

    user_count = db.query(User).count()
    paid_users = db.query(User).filter(User.is_paid.is_(True)).count()
    usage_logs = db.query(UsageLog).count()

    # Echo the server's configured admin email for sanity checks
    admin_email_server = os.environ.get("ADMIN_EMAIL", current_user.email)

    return {
        "users": user_count,
        "paid_users": paid_users,
        "usage_logs": usage_logs,
        "admin_email": admin_email_server,
    }

# ---------- Entrypoint ----------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )
    
    


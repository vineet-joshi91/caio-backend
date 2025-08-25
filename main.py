# main.py
import os
import logging
from typing import Optional
from datetime import datetime

from fastapi import (
    FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
)
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

# ---------------------------------------------------------------------
# App & logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("caio")

app = FastAPI(title="CAIO Backend", version="0.1.0")

# ---------------------------------------------------------------------
# CORS (Netlify + Vercel + local dev)
# ---------------------------------------------------------------------
ALLOWED_ORIGINS_DEFAULT = [
    "https://caio-frontend.vercel.app",
    "https://caioai.netlify.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
extra = os.getenv("ALLOWED_ORIGINS", "")
if extra:
    ALLOWED_ORIGINS_DEFAULT += [o.strip() for o in extra.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS_DEFAULT,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# Let browsers succeed on preflight even if a specific method/path isn't declared.
@app.options("/{path:path}")
def cors_preflight(path: str):
    return JSONResponse({"ok": True})

# ---------------------------------------------------------------------
# Health/Ready
# ---------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}

@app.get("/api/ready")
def ready():
    return {"ready": True, "time": datetime.utcnow().isoformat() + "Z"}

# ---------------------------------------------------------------------
# Auth: /api/login and /api/profile
# ---------------------------------------------------------------------
ADMIN_EMAILS = {
    e.strip().lower()
    for e in os.getenv("ADMIN_EMAILS", "vineetpjoshi.71@gmail.com").split(",")
    if e.strip()
}

@app.post("/api/login")
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """
    Password login. If user doesn't exist yet, create them (create-or-login behavior).
    Frontend posts 'username' (email) and 'password' as x-www-form-urlencoded.
    """
    email = (form.username or "").strip().lower()
    password = form.password or ""

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")

    user = db.query(User).filter(User.email == email).first()
    if user:
        if not verify_password(password, user.hashed_password):
            raise HTTPException(status_code=400, detail="Incorrect email or password")
    else:
        # Create on first login to keep current behavior
        user = User(
            email=email,
            hashed_password=get_password_hash(password),
            is_admin=(email in ADMIN_EMAILS) if hasattr(User, "is_admin") else False,
            is_paid=False if hasattr(User, "is_paid") else False,
            created_at=datetime.utcnow() if hasattr(User, "created_at") else None,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    token = create_access_token(sub=user.email)
    return {
        "access_token": token,
        "token_type": "bearer",
        "email": user.email,
        "is_admin": bool(getattr(user, "is_admin", False)),
        "is_paid": bool(getattr(user, "is_paid", False)),
    }

@app.get("/api/profile")
def profile(current_user: User = Depends(get_current_user)):
    """
    Basic profile for the frontend to check auth state.
    """
    data = {
        "email": current_user.email,
        "is_admin": bool(getattr(current_user, "is_admin", False)),
        "is_paid": bool(getattr(current_user, "is_paid", False)),
    }
    for k in ("name", "organisation", "company", "created_at"):
        if hasattr(current_user, k):
            data[k] = getattr(current_user, k)
    return data

# ---------------------------------------------------------------------
# Demo Analyze stub (kept as-is; adjust when engines are wired)
# ---------------------------------------------------------------------
@app.post("/api/analyze")
async def analyze(
    request: Request,
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
):
    text = (text or "").strip()

    if not getattr(current_user, "is_paid", False):
        return JSONResponse(
            {
                "status": "demo",
                "title": "Demo Mode Result",
                "summary": "This is a sample analysis. Upgrade to Pro for full insights.",
                "tip": "Upload a business document or upgrade your plan to unlock advanced engines.",
            },
            status_code=200,
        )

    # Placeholder until Pro engines/credits are wired
    return JSONResponse(
        {
            "status": "error",
            "title": "Analysis Unavailable",
            "message": "You’ve run out of credits or the AI engine is unavailable right now.",
            "action": "Please add credits or try again later.",
        },
        status_code=402,
    )

# ---------------------------------------------------------------------
# Routers (mounted safely so import errors never break boot)
# ---------------------------------------------------------------------
try:
    from routes_public_config import router as public_cfg_router
    app.include_router(public_cfg_router)  # exposes /api/public-config
    logger.info("Loaded routes_public_config")
except Exception as e:
    logger.warning(f"routes_public_config not loaded: {e}")

try:
    from payment_routes import router as payments_router
    # payment_routes already uses prefix="/api/payments" internally; don't add one here.
    app.include_router(payments_router)
    logger.info("Loaded /api/payments/*")
except Exception as e:
    logger.warning(f"payment_routes not loaded: {e}")

try:
    from contact_routes import router as contact_router
    app.include_router(contact_router)  # /api/contact
    logger.info("Loaded /api/contact")
except Exception as e:
    logger.warning(f"contact_routes not loaded: {e}")

try:
    from admin_bootstrap import router as admin_bootstrap_router
    app.include_router(admin_bootstrap_router)  # /api/admin_bootstrap/grant
    logger.info("Loaded /api/admin_bootstrap")
except Exception as e:
    logger.warning(f"admin_bootstrap not loaded: {e}")

try:
    from maintenance_routes import router as maintenance_router
    app.include_router(maintenance_router)  # /api/admin/maintenance/upgrade-db
    logger.info("Loaded /api/admin/maintenance")
except Exception as e:
    logger.warning(f"maintenance_routes not loaded: {e}")

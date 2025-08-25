# main.py
import os
import logging
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from health_routes import router as health_router

from db import get_db, User
from auth import (
    create_access_token,
    verify_password,
    get_password_hash,
    get_current_user,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("caio")

from admin_metrics_routes import router as admin_metrics_router

# -----------------------------------------------------------------------------
# App FIRST
# -----------------------------------------------------------------------------
app = FastAPI(title="CAIO Backend", version="0.1.0")

# -----------------------------------------------------------------------------
# CORS
# -----------------------------------------------------------------------------
DEFAULT_ORIGINS = [
    "https://caio-frontend.vercel.app",
    "https://caioai.netlify.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
extra = os.getenv("ALLOWED_ORIGINS", "")
if extra:
    DEFAULT_ORIGINS += [o.strip() for o in extra.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEFAULT_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

app.include_router(admin_metrics_router)

@app.options("/{path:path}")
def cors_preflight(path: str):
    # Helps browsers pass preflight even if a specific route isn’t defined
    return JSONResponse({"ok": True})

# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}

app.include_router(health_router)

# -----------------------------------------------------------------------------
# Auth: /api/login and /api/profile
# -----------------------------------------------------------------------------
@app.post("/api/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    Password login. If user doesn't exist yet, create them (create-or-login behavior).
    Frontend posts 'username' (email) and 'password' as x-www-form-urlencoded.
    """
    email = (form.username or "").strip().lower()
    password = form.password or ""

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        # Create on first login to keep current behavior
        user = User(email=email, hashed_password=get_password_hash(password))
        # Optional flags if your model has these columns
        if hasattr(user, "is_admin"):
            user.is_admin = False
        if hasattr(user, "is_paid"):
            user.is_paid = False
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        if not verify_password(password, user.hashed_password):
            raise HTTPException(status_code=400, detail="Incorrect email or password")

    token = create_access_token(sub=user.email)
    return {"access_token": token, "token_type": "bearer"}

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
    # include optional fields if present
    for k in ("name", "organisation", "company", "created_at"):
        if hasattr(current_user, k):
            data[k] = getattr(current_user, k)
    return data

# -----------------------------------------------------------------------------
# Routers (after app exists; guard optional ones so boot never crashes)
# -----------------------------------------------------------------------------
try:
    from routes_public_config import router as public_cfg_router
    app.include_router(public_cfg_router, tags=["public"])
    logger.info("Loaded routes_public_config")
except Exception as e:
    logger.warning(f"routes_public_config not loaded: {e}")

try:
    from signup_routes import router as signup_router
    app.include_router(signup_router)
    logger.info("Loaded /api/signup")
except Exception as e:
    logger.warning(f"signup_routes not loaded: {e}")

try:
    from payment_routes import router as payments_router
    # your payment_routes has no prefix → mount under /api/payments here
    app.include_router(payments_router, prefix="/api/payments", tags=["payments"])
    logger.info("Loaded /api/payments/*")
except Exception as e:
    logger.warning(f"payment_routes not loaded: {e}")

try:
    from contact_routes import router as contact_router
    app.include_router(contact_router)
    logger.info("Loaded /api/contact")
except Exception as e:
    logger.warning(f"contact_routes not loaded: {e}")

try:
    from admin_bootstrap import router as admin_bootstrap_router
    app.include_router(admin_bootstrap_router)
except Exception as e:
    logger.warning(f"admin_bootstrap not loaded: {e}")

try:
    from maintenance_routes import router as maintenance_router
    app.include_router(maintenance_router)
except Exception as e:
    logger.warning(f"maintenance_routes not loaded: {e}")

# -----------------------------------------------------------------------------
# (Keep any other existing endpoints below; no changes needed)
# -----------------------------------------------------------------------------

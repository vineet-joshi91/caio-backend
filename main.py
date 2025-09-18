# -*- coding: utf-8 -*-
# main.py — CAIO Backend (resilient startup, stable routes, pricing endpoints)

import os
import re
import json
import asyncio
import logging
import traceback
from io import BytesIO
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI, Depends, HTTPException, UploadFile, File, Form, Request, Query
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from pydantic import BaseModel

# ---- Project modules (must exist in your repo) ----
from db import get_db, User, init_db, UsageLog, ChatSession, ChatMessage
from auth import create_access_token, verify_password, get_password_hash, get_current_user

# Optional routers (don’t fail boot if they’re absent)
try:
    from routers import chat_router   # type: ignore
except Exception:
    chat_router = None  # noqa: E305
try:
    from routers import admin_router  # type: ignore
except Exception:
    admin_router = None  # noqa: E305

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("caio")

# --------------------------------------------------------------------------------------
# Lifespan: resilient startup with diagnostics (appears in /api/ready)
# --------------------------------------------------------------------------------------
DB_READY = False
STARTUP_OK = False
STARTUP_ERROR = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Keep startup resilient: warm DB with retries; never hard-crash the process.
    Any exception is captured and exposed via /api/ready for quick diagnosis.
    """
    global DB_READY, STARTUP_OK, STARTUP_ERROR
    tries = int(os.getenv("DB_WARMUP_TRIES", "20"))
    delay = float(os.getenv("DB_WARMUP_DELAY", "1.5"))
    try:
        for i in range(tries):
            try:
                init_db()
                DB_READY = True
                break
            except Exception as e:
                logger.warning("init_db attempt %s/%s failed: %s", i + 1, tries, e)
                await asyncio.sleep(delay)
        STARTUP_OK = True
        STARTUP_ERROR = ""
        yield
    except Exception:
        STARTUP_OK = False
        STARTUP_ERROR = traceback.format_exc()
        logger.error("Startup failed:\n%s", STARTUP_ERROR)
        # still yield so health/ready endpoints respond with details
        yield


app = FastAPI(title="CAIO Backend", version="0.5.0", lifespan=lifespan)

# --------------------------------------------------------------------------------------
# CORS
# --------------------------------------------------------------------------------------
ALLOWED_ORIGINS = [
    "https://caio-frontend.vercel.app",
    "https://caioai.netlify.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
extra = (os.getenv("ALLOWED_ORIGINS") or "").strip()
if extra:
    ALLOWED_ORIGINS += [o.strip() for o in extra.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

@app.options("/{path:path}")
def cors_preflight(path: str):
    return JSONResponse({"ok": True})

# --------------------------------------------------------------------------------------
# Health / Ready
# --------------------------------------------------------------------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "caio-backend", "version": "0.5.0"}

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.5.0"}

@app.get("/api/ready")
def ready():
    return {
        "ready": True,
        "db_ready": DB_READY,
        "startup_ok": STARTUP_OK,
        "startup_error": (STARTUP_ERROR[:4000] if STARTUP_ERROR else ""),
        "time": datetime.utcnow().isoformat() + "Z",
    }

# --------------------------------------------------------------------------------------
# Tiers, testers and limits
# --------------------------------------------------------------------------------------
def _env_set(name: str) -> set:
    raw = os.getenv(name, "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}

ADMIN_EMAILS      = _env_set("ADMIN_EMAIL")
PREMIUM_EMAILS    = _env_set("PREMIUM_EMAILS")
PRO_PLUS_EMAILS   = _env_set("PRO_PLUS_EMAILS")
PRO_TEST_EMAILS   = _env_set("PRO_TEST_EMAILS")

# hardcoded legacy tester id is allowed but optional
PRO_TEST_EMAILS_HARDCODE = {"testpro@123.com"}

def _bool_env(name: str, default: bool = False) -> bool:
    v = (os.getenv(name, "").strip().lower())
    if not v:
        return default
    return v in ("1", "true", "yes", "y", "t")

PRO_TEST_ENABLE        = _bool_env("PRO_TEST_ENABLE", True)
PRO_TEST_AUTO_CREATE   = _bool_env("PRO_TEST_AUTO_CREATE", True)
PRO_TEST_DEFAULT_PW    = os.getenv("PRO_TEST_DEFAULT_PASSWORD", "testpro123")

# “Free/Trial” preview caps (fallbacks to TRIAL_* or small defaults)
FREE_CHAT_MSGS_PER_DAY    = int(os.getenv("FREE_CHAT_MSGS_PER_DAY", os.getenv("TRIAL_MAX_MSGS", "3")))
FREE_CHAT_UPLOADS_PER_DAY = int(os.getenv("FREE_CHAT_UPLOADS_PER_DAY", os.getenv("TRIAL_MAX_MSGS", "3")))

FREE_QUERIES_PER_DAY = int(os.getenv("FREE_QUERIES_PER_DAY", "3"))
FREE_UPLOADS         = int(os.getenv("FREE_UPLOADS", "3"))
FREE_BRAINS          = int(os.getenv("FREE_BRAINS", "2"))

# usage helpers
def _today_bounds_utc() -> Tuple[datetime, datetime]:
    now = datetime.utcnow()
    start = datetime(now.year, now.month, now.day)
    end = start + timedelta(days=1)
    return start, end

def _count_usage(db: Session, user_id: int, endpoint: str) -> int:
    start, end = _today_bounds_utc()
    return db.query(UsageLog).filter(
        UsageLog.user_id == user_id,
        UsageLog.endpoint == endpoint,
        UsageLog.timestamp >= start,
        UsageLog.timestamp < end,
    ).count()

def _log_usage(db: Session, user_id: int, endpoint: str, status: str = "ok", meta: str = ""):
    db.add(UsageLog(user_id=user_id or 0, timestamp=datetime.utcnow(), endpoint=endpoint, status=status, meta=meta))
    db.commit()

def _limit_response(used: int, limit: int, reset_at: Optional[str], plan: str):
    return JSONResponse({
        "status": "error",
        "title": "Daily limit reached",
        "message": f"You've used {used}/{limit} {plan.capitalize()} requests today.",
        "plan": plan,
        "used": used,
        "limit": limit,
        "reset_at": reset_at,
    }, status_code=429)

# ---- tier helpers ----
def _user_tier(u: User) -> str:
    email = (getattr(u, "email", "") or "").lower()
    explicit = getattr(u, "tier", None)
    if isinstance(explicit, str) and explicit in ("admin", "premium", "pro_plus", "pro", "demo"):
        return explicit
    if not email:
        return "demo"
    if email in ADMIN_EMAILS:
        return "admin"
    if email in PRO_PLUS_EMAILS:
        return "pro_plus"
    if email in PREMIUM_EMAILS or getattr(u, "is_paid", False):
        # treat any paid flag or premium email as Pro by default (unless pro_plus/premium above)
        return "pro"
    return "demo"

def _is_admin(u: User) -> bool:
    return _user_tier(u) == "admin"

def apply_tier_overrides(user: User) -> User:
    """
    QA override: allow emails listed in PRO_TEST_EMAILS (or legacy hardcoded)
    to behave as Pro without changing persistent billing state.
    """
    try:
        allow = _env_set("PRO_TEST_EMAILS")
        allow |= PRO_TEST_EMAILS_HARDCODE
        email = (user.email or "").lower()
        if PRO_TEST_ENABLE and email in allow:
            user.tier = "pro"
            user.is_paid = True
    except Exception:
        pass
    return user

# --------------------------------------------------------------------------------------
# Auth — robust JSON or form parsing; tester auto-create; stable token shape
# --------------------------------------------------------------------------------------
def _extract_email_password_from_request(request: Request) -> Tuple[str, str]:
    email = ""
    password = ""
    ctype = (request.headers.get("content-type") or "").lower()
    async def _try_json():
        try:
            data = await request.json()
            return (
                (data.get("email") or data.get("username") or "").strip().lower(),
                data.get("password") or "",
            )
        except Exception:
            return "", ""

    async def _try_form():
        try:
            form = await request.form()
            return (
                (form.get("email") or form.get("username") or "").strip().lower(),
                form.get("password") or "",
            )
        except Exception:
            return "", ""

    return asyncio.get_event_loop().run_until_complete(
        _try_json() if ctype.startswith("application/json") else _try_form()
    ) or ("", "")

@app.post("/api/login")
async def login(request: Request, db: Session = Depends(get_db)):
    """
    Accept BOTH form (application/x-www-form-urlencoded/multipart) and JSON.
    If an allowed tester email signs in and doesn't exist, optionally auto-create.
    Always returns token shape: {"access_token": "...", "token_type": "bearer"}
    """
    email = ""
    password = ""
    ctype = (request.headers.get("content-type") or "").lower()

    # parse body
    try:
        if ctype.startswith("application/json"):
            data = await request.json()
            email = (data.get("email") or data.get("username") or "").strip().lower()
            password = data.get("password") or ""
        else:
            form = await request.form()
            email = (form.get("email") or form.get("username") or "").strip().lower()
            password = form.get("password") or ""
    except Exception:
        # last chance: try JSON again
        try:
            data = await request.json()
            email = (data.get("username") or data.get("email") or "").strip().lower()
            password = data.get("password") or ""
        except Exception:
            pass

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")

    user = db.query(User).filter(User.email == email).first()
    is_tester = PRO_TEST_ENABLE and (email in PRO_TEST_EMAILS or email in PRO_TEST_EMAILS_HARDCODE)

    if not user:
        if is_tester and PRO_TEST_AUTO_CREATE:
            # auto-provision with provided password (or default)
            hashed = get_password_hash(password or PRO_TEST_DEFAULT_PW)
            user = User(
                email=email,
                hashed_password=hashed,
                is_admin=False,
                is_paid=False,
                created_at=datetime.utcnow(),
            )
            db.add(user); db.commit(); db.refresh(user)
        else:
            raise HTTPException(status_code=401, detail="User not found")

    # password check (permit testers using default)
    if not verify_password(password, user.hashed_password):
        if not (is_tester and PRO_TEST_DEFAULT_PW and verify_password(PRO_TEST_DEFAULT_PW, user.hashed_password)):
            raise HTTPException(status_code=401, detail="Incorrect email or password")

    # keep admin/premium flags loosely in sync with env
    changed = False
    if hasattr(user, "is_admin"):
        admin_now = email in ADMIN_EMAILS
        if bool(user.is_admin) != admin_now:
            user.is_admin = admin_now
            changed = True
    if hasattr(user, "is_paid"):
        premium_now = email in PREMIUM_EMAILS
        if premium_now and not bool(user.is_paid):
            user.is_paid = True
            changed = True
    if changed:
        db.commit()

    # non-persistent QA override
    user = apply_tier_overrides(user)

    token = create_access_token(email)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/api/signup")
async def signup(request: Request, db: Session = Depends(get_db)):
    """
    Create a new user account (JSON or form-data).
    Returns: {"ok": True, "user": {"id", "email"}, "access_token", "token_type"}
    """
    try:
        email = ""
        password = ""
        ct = (request.headers.get("content-type") or "").lower()
        if "application/json" in ct:
            data = await request.json()
            email = (data.get("email") or "").strip().lower()
            password = data.get("password") or ""
        else:
            form = await request.form()
            email = ((form.get("email") or "")).strip().lower()
            password = form.get("password") or ""

        if not email or not password:
            raise HTTPException(status_code=422, detail="email and password are required")

        if "@" not in email or "." not in email.split("@")[-1]:
            raise HTTPException(status_code=422, detail="invalid email format")

        hashed = get_password_hash(password)
        user = User(
            email=email,
            hashed_password=hashed,
            is_admin=False,
            is_paid=False,
            created_at=datetime.utcnow(),
        )
        db.add(user)
        db.flush()   # surface constraint violations before commit
        db.commit()
        db.refresh(user)

        access_token = create_access_token(user.email)
        return {
            "ok": True,
            "user": {"id": user.id, "email": user.email},
            "access_token": access_token,
            "token_type": "bearer",
        }

    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="User already exists or invalid data.")
    except HTTPException:
        raise
    except Exception:
        db.rollback()
        logger.error("signup failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Signup failed on the server. Please try again.")

# --------------------------------------------------------------------------------------
# Profile — small helper the frontend expects
# --------------------------------------------------------------------------------------
@app.get("/api/profile")
def profile(current: User = Depends(get_current_user)):
    """
    Return minimal profile the frontend uses; keep key names stable.
    """
    u = apply_tier_overrides(current)
    tier = _user_tier(u)
    return {
        "email": u.email,
        "tier": tier,
        "is_admin": (tier == "admin"),
        "created_at": getattr(u, "created_at", None),
    }

# --------------------------------------------------------------------------------------
# Pricing (shared loader) + endpoints: /api/public-config (stable shape) & /api/pricing
# --------------------------------------------------------------------------------------
def _load_pricing_from_env() -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      {
        "INR": {"symbol":"₹","pro":1999,"pro_plus":3999,"premium":7999},
        "USD": {"symbol":"$","pro":25,"pro_plus":49,"premium":99}
      }
    Prefers PRICING_JSON; falls back to individual env vars.
    """
    raw = (os.getenv("PRICING_JSON") or "").strip()
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "INR" in data and "USD" in data:
                return data
        except Exception:
            logger.warning("Failed to parse PRICING_JSON; falling back to discrete env vars.")
    return {
        "INR": {
            "symbol": "₹",
            "pro": int(os.getenv("PRO_PRICE_INR", "1999") or 1999),
            "pro_plus": int(os.getenv("PRO_PLUS_PRICE_INR", "3999") or 3999),
            "premium": int(os.getenv("PREMIUM_PRICE_INR", "7999") or 7999),
        },
        "USD": {
            "symbol": "$",
            "pro": float(os.getenv("PRO_PRICE_USD", "25") or 25),
            "pro_plus": float(os.getenv("PRO_PLUS_PRICE_USD", "49") or 49),
            "premium": float(os.getenv("PREMIUM_PRICE_USD", "99") or 99),
        },
    }

@app.get("/api/pricing")
def public_pricing():
    data = _load_pricing_from_env()
    return {"ok": True, "pricing": data, "updated_at": datetime.utcnow().isoformat() + "Z"}

@app.get("/api/public-config")
def public_config():
    """
    Keep structure stable for the frontend:
      {
        "currency": "INR"|"USD",
        "plans": {"pro": {"price": int}, "pro_plus": {...}, "premium": {...}},
        "chat_preview": {"enabled": bool, "msgs_per_day": int, "uploads_per_day": int}
      }
    """
    currency = (os.getenv("PAY_DEFAULT_CURRENCY", "INR") or "INR").upper()
    table = _load_pricing_from_env()
    current = table.get(currency) or table.get("USD")

    def _as_int(v: Any) -> int:
        try:
            return int(round(float(v)))
        except Exception:
            return 0

    plans = {
        "pro":      {"price": _as_int(current.get("pro", 0))},
        "pro_plus": {"price": _as_int(current.get("pro_plus", 0))},
        "premium":  {"price": _as_int(current.get("premium", 0))},
    }
    chat_preview = {
        "enabled": True,
        "msgs_per_day": FREE_CHAT_MSGS_PER_DAY,
        "uploads_per_day": FREE_CHAT_UPLOADS_PER_DAY,
    }
    return {"currency": currency, "plans": plans, "chat_preview": chat_preview}

# --------------------------------------------------------------------------------------
# (Chat/Admin routers) — preserved, only mounted if present in the repo
# --------------------------------------------------------------------------------------
if chat_router:
    app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
if admin_router:
    app.include_router(admin_router, prefix="/api/admin", tags=["admin"])

# -*- coding: utf-8 -*-
# main.py (CAIO Backend) — resilient startup + intact routes

import os, json, logging, traceback, re
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
from io import BytesIO
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request, Query, APIRouter
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func
from pydantic import BaseModel

# project modules
from db import get_db, User, init_db, UsageLog, ChatSession, ChatMessage
from auth import create_access_token, verify_password, get_password_hash, get_current_user

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("caio")
DEBUG = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------
# Resilient startup: warm DB on lifespan instead of crashing on import
# ---------------------------------------------------------------------

DB_READY = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global DB_READY
    tries = int(os.getenv("DB_WARMUP_TRIES", "20"))
    delay = float(os.getenv("DB_WARMUP_DELAY", "1.5"))
    for i in range(tries):
        try:
            init_db()
            DB_READY = True
            break
        except Exception as e:
            logger.warning("init_db attempt %s/%s failed: %s", i + 1, tries, e)
            await asyncio.sleep(delay)
    yield

app = FastAPI(title="CAIO Backend", version="0.4.0", lifespan=lifespan)

# ---------------- CORS ----------------
ALLOWED_ORIGINS = [
    "https://caio-frontend.vercel.app",
    "https://caioai.netlify.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
extra = os.getenv("ALLOWED_ORIGINS", "")
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

# ---------------- Health ----------------
@app.get("/")
def root():
    return {"ok": True, "service": "caio-backend", "version": "0.4.0"}

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.4.0"}

@app.get("/api/ready")
def ready():
    return {"ready": True, "db_ready": DB_READY, "time": datetime.utcnow().isoformat() + "Z"}

# ---------------- Tiers & limits ----------------
ADMIN_EMAILS = set(e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip())
def is_admin_email(email: str) -> bool:
    return email.lower() in ADMIN_EMAILS
PREMIUM_EMAILS = {e.strip().lower() for e in os.getenv("PREMIUM_EMAILS", "").split(",") if e.strip()}

FREE_QUERIES_PER_DAY = int(os.getenv("FREE_QUERIES_PER_DAY", "3"))
FREE_UPLOADS         = int(os.getenv("FREE_UPLOADS", "3"))
FREE_BRAINS          = int(os.getenv("FREE_BRAINS", "2"))

FREE_CHAT_MSGS_PER_DAY     = int(os.getenv("FREE_CHAT_MSGS_PER_DAY", "3"))
FREE_CHAT_UPLOADS_PER_DAY  = int(os.getenv("FREE_CHAT_UPLOADS_PER_DAY", "3"))

PRO_PLUS_EMAILS = {e.strip().lower() for e in os.getenv("PRO_PLUS_EMAILS", "").split(",") if e.strip()}

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
    if email in PREMIUM_EMAILS:
        return "premium"
    if getattr(u, "is_paid", False):
        return "pro"
    return "demo"

def _is_admin(u: User) -> bool:
    return _user_tier(u) == "admin"

def _is_premium_or_plus(u: User) -> bool:
    t = _user_tier(u)
    return t in ("admin", "premium", "pro_plus")

def apply_tier_overrides(user):
    allow = {e.strip().lower() for e in os.getenv("PRO_TEST_EMAILS", "").split(",") if e.strip()}
    if user.email.lower() in allow:
        user.tier = "pro"
        user.is_paid = True
    return user

# ---------- usage helpers ----------
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

# ---------------- Auth ----------------
# (login/signup/profile/logout unchanged — kept from your file)
# ---------------- Analyzer ----------------
DEFAULT_BRAINS_ORDER = ["CFO","CHRO","COO","CMO","CPO"]

def _brain_prompt(brief: str, extracted: str, brain: str) -> str:
    role_map = {
        "CFO":  "Chief Financial Officer - unit economics; revenue mix, margins, CCC, runway.",
        "CHRO": "Chief Human Resources Officer - org effectiveness; attrition, engagement.",
        "COO":  "Chief Operating Officer - cost-to-serve & reliability; capacity, throughput, SLA.",
        "CMO":  "Chief Marketing Officer - efficient growth; CAC/LTV, funnel, retention.",
        "CPO":  "Chief People Officer - talent acquisition; pipeline, time-to-hire, QoH.",
    }
    role = role_map.get(brain, "Executive Advisor")
    return f"""
You are {role}.
Return STRICT MARKDOWN ONLY for **{brain}** with this structure:

### Insights
1. <insight>
2. <insight>
3. <insight>

### Recommendations
1. **<headline>**: <ONE sentence action.>
2. **<headline>**: <ONE sentence action.>
3. **<headline>**: <ONE sentence action.>

[BRIEF]
{brief or "(none)"}

[DOCUMENT TEXT]
{(extracted or "")[:120000]}
""".strip()

def _compose_premium_system_prompt() -> str:
    return (
        "You are CAIO - a pragmatic business & ops copilot. "
        "Answer clearly in Markdown. When files are provided, ground your answer on them."
    )

# (chat endpoints, admin, routers unchanged — kept as in your file)

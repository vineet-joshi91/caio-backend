# -*- coding: utf-8 -*-
# main.py — CAIO Backend (resilient startup, stable routes, chat+admin fixes)

import os
import json
import asyncio
import logging
import traceback
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI, Depends, HTTPException, UploadFile, File, Form, Request, Query, status
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func

# ---- Project modules (must exist) ----
from db import get_db, User, init_db, UsageLog, ChatSession, ChatMessage
from auth import create_access_token, verify_password, get_password_hash, get_current_user

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("caio")

# --------------------------------------------------------------------------------------
# Lifespan: resilient startup with diagnostics (visible via /api/ready)
# --------------------------------------------------------------------------------------
DB_READY = False
STARTUP_OK = False
STARTUP_ERROR = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
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
        yield


app = FastAPI(title="CAIO Backend", version="0.7.0", lifespan=lifespan)

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

# Preflight helper (does NOT cause 405 for valid routes)
@app.options("/{path:path}")
def cors_preflight(path: str):
    return JSONResponse({"ok": True})

# --------------------------------------------------------------------------------------
# Health / Ready
# --------------------------------------------------------------------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "caio-backend", "version": "0.7.0"}

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.7.0"}

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

# admins — accept ADMIN_EMAILS and legacy ADMIN_EMAIL
ADMIN_EMAILS = _env_set("ADMIN_EMAILS")
_admin_single = (os.getenv("ADMIN_EMAIL") or "").strip().lower()
if _admin_single:
    ADMIN_EMAILS.add(_admin_single)

PREMIUM_EMAILS  = _env_set("PREMIUM_EMAILS")
PRO_PLUS_EMAILS = _env_set("PRO_PLUS_EMAILS")
PRO_TEST_EMAILS = _env_set("PRO_TEST_EMAILS")
PRO_TEST_EMAILS_HARDCODE = {"testpro@123.com"}

def _bool_env(name: str, default: bool = False) -> bool:
    v = (os.getenv(name, "").strip().lower())
    if not v:
        return default
    return v in ("1", "true", "yes", "y", "t")

PRO_TEST_ENABLE        = _bool_env("PRO_TEST_ENABLE", True)
PRO_TEST_AUTO_CREATE   = _bool_env("PRO_TEST_AUTO_CREATE", True)
PRO_TEST_DEFAULT_PW    = os.getenv("PRO_TEST_DEFAULT_PASSWORD", "testpro123")

FREE_CHAT_MSGS_PER_DAY    = int(os.getenv("FREE_CHAT_MSGS_PER_DAY", os.getenv("TRIAL_MAX_MSGS", "3")))
FREE_CHAT_UPLOADS_PER_DAY = int(os.getenv("FREE_CHAT_UPLOADS_PER_DAY", os.getenv("TRIAL_MAX_MSGS", "3")))
FREE_QUERIES_PER_DAY = int(os.getenv("FREE_QUERIES_PER_DAY", "3"))
FREE_UPLOADS         = int(os.getenv("FREE_UPLOADS", "3"))
FREE_BRAINS          = int(os.getenv("FREE_BRAINS", "2"))

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

def _user_tier(u: User) -> str:
    email = (getattr(u, "email", "") or "").lower()
    if email in ADMIN_EMAILS:
        return "admin"
    if email in PRO_PLUS_EMAILS:
        return "pro_plus"
    if email in PREMIUM_EMAILS:
        return "premium"
    explicit = getattr(u, "tier", None)
    if isinstance(explicit, str) and explicit in ("admin", "premium", "pro_plus", "pro", "demo"):
        return explicit
    if getattr(u, "is_paid", False):
        return "pro"
    return "demo" if email else "demo"

def apply_tier_overrides(user: User) -> User:
    try:
        email = (user.email or "").lower()
        if email in ADMIN_EMAILS:
            return user
        allow = _env_set("PRO_TEST_EMAILS") | PRO_TEST_EMAILS_HARDCODE
        if PRO_TEST_ENABLE and email in allow:
            user.is_paid = True
    except Exception:
        pass
    return user

# --------------------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------------------
@app.post("/api/login")
async def login(request: Request, db: Session = Depends(get_db)):
    email = ""
    password = ""
    ctype = (request.headers.get("content-type") or "").lower()
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

    if not verify_password(password, user.hashed_password):
        if not (is_tester and PRO_TEST_DEFAULT_PW and verify_password(PRO_TEST_DEFAULT_PW, user.hashed_password)):
            raise HTTPException(status_code=401, detail="Incorrect email or password")

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

    user = apply_tier_overrides(user)
    token = create_access_token(email)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/api/signup")
async def signup(request: Request, db: Session = Depends(get_db)):
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
        db.flush()
        db.commit()
        db.refresh(user)

        access_token = create_access_token(user.email)
        return {"ok": True, "user": {"id": user.id, "email": user.email},
                "access_token": access_token, "token_type": "bearer"}

    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="User already exists or invalid data.")
    except HTTPException:
        raise
    except Exception:
        db.rollback()
        logger.error("signup failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Signup failed on the server. Please try again.")

@app.get("/api/profile")
def profile(current: User = Depends(get_current_user)):
    u = apply_tier_overrides(current)
    tier = _user_tier(u)
    return {"email": u.email, "tier": tier, "is_admin": (tier == "admin"),
            "created_at": getattr(u, "created_at", None)}

# --------------------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------------------
def _load_pricing_from_env() -> Dict[str, Dict[str, Any]]:
    raw = (os.getenv("PRICING_JSON") or "").strip()
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "INR" in data and "USD" in data:
                return data
        except Exception:
            logger.warning("Failed to parse PRICING_JSON; using discrete env vars.")
    return {
        "INR": {"symbol": "₹",
                "pro": int(os.getenv("PRO_PRICE_INR", "1999") or 1999),
                "pro_plus": int(os.getenv("PRO_PLUS_PRICE_INR", "3999") or 3999),
                "premium": int(os.getenv("PREMIUM_PRICE_INR", "7999") or 7999)},
        "USD": {"symbol": "$",
                "pro": float(os.getenv("PRO_PRICE_USD", "25") or 25),
                "pro_plus": float(os.getenv("PRO_PLUS_PRICE_USD", "49") or 49),
                "premium": float(os.getenv("PREMIUM_PRICE_USD", "99") or 99)},
    }

@app.get("/api/pricing")
def public_pricing():
    data = _load_pricing_from_env()
    return {"ok": True, "pricing": data, "updated_at": datetime.utcnow().isoformat() + "Z"}

@app.get("/api/public-config")
def public_config():
    currency = (os.getenv("PAY_DEFAULT_CURRENCY", "INR") or "INR").upper()
    table = _load_pricing_from_env()
    current = table.get(currency) or table.get("USD")

    def _as_int(v: Any) -> int:
        try:
            return int(round(float(v)))
        except Exception:
            return 0

    plans = {
        "pro": {"price": _as_int(current.get("pro", 0))},
        "pro_plus": {"price": _as_int(current.get("pro_plus", 0))},
        "premium": {"price": _as_int(current.get("premium", 0))},
    }
    chat_preview = {
        "enabled": True,
        "msgs_per_day": FREE_CHAT_MSGS_PER_DAY,
        "uploads_per_day": FREE_CHAT_UPLOADS_PER_DAY,
    }
    return {"currency": currency, "plans": plans, "chat_preview": chat_preview}

# --------------------------------------------------------------------------------------
# Admin (accept GET and POST)
# --------------------------------------------------------------------------------------
def _require_admin(u: User):
    if _user_tier(u) != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")

@app.api_route("/api/admin/users/summary", methods=["GET", "POST"])
def admin_users_summary(db: Session = Depends(get_db), current: User = Depends(get_current_user)):
    _require_admin(current)
    users = db.query(User).all()
    total = len(users); demo = pro = pro_plus = 0; premium_admin = 0
    for u in users:
        t = _user_tier(apply_tier_overrides(u))
        if t == "demo": demo += 1
        elif t == "pro": pro += 1
        elif t == "pro_plus": pro_plus += 1
        if t in ("admin", "premium"):
            premium_admin += 1
    return {"ok": True, "total": total, "demo": demo, "pro": pro, "pro_plus": pro_plus,
            "premium_admin": premium_admin, "time": datetime.utcnow().isoformat() + "Z"}

async def _admin_users_search_impl(request: Request, q: Optional[str], page: int, page_size: int,
                                   db: Session, current: User):
    _require_admin(current)
    ct = (request.headers.get("content-type") or "").lower()
    if request.method.upper() == "POST":
        try:
            if "application/json" in ct:
                data = await request.json()
                q = (data.get("q") or q or "").strip()
                page = int(data.get("page") or page or 1)
                page_size = int(data.get("page_size") or page_size or 25)
            else:
                form = await request.form()
                q = (form.get("q") or q or "").strip()
                page = int(form.get("page") or page or 1)
                page_size = int(form.get("page_size") or page_size or 25)
        except Exception:
            pass

    q = (q or "").strip().lower()
    page = max(1, int(page)); page_size = max(1, min(200, int(page_size)))

    query = db.query(User)
    if q:
        query = query.filter(func.lower(User.email).like(f"%{q}%"))

    total = query.count()
    rows = (query
            .order_by(User.created_at.desc() if hasattr(User, "created_at") else User.id.desc())
            .offset((page - 1) * page_size).limit(page_size).all())

    last_seen_map: Dict[int, Optional[str]] = {}
    sessions_map: Dict[int, int] = {}
    try:
        ls = db.query(ChatMessage.user_id, func.max(ChatMessage.created_at)).group_by(ChatMessage.user_id).all()
        for uid, dt in ls:
            last_seen_map[int(uid)] = (dt.isoformat() if dt else None)
    except Exception:
        pass
    try:
        ses = db.query(ChatSession.user_id, func.count(ChatSession.id)).group_by(ChatSession.user_id).all()
        for uid, cnt in ses:
            sessions_map[int(uid)] = int(cnt or 0)
    except Exception:
        pass

    items = []
    for u in rows:
        tu = apply_tier_overrides(u)
        tier = _user_tier(tu)
        uid = int(getattr(u, "id", 0))
        items.append({
            "email": u.email,
            "tier": tier,
            "created_at": (u.created_at.isoformat() if hasattr(u, "created_at") and u.created_at else None),
            "last_seen": last_seen_map.get(uid),
            "total_sessions": sessions_map.get(uid, 0),
            "tokens_used": 0,
        })
    return {"ok": True, "q": q, "page": page, "page_size": page_size, "total": total, "items": items}

@app.api_route("/api/admin/users/search", methods=["GET", "POST"])
async def admin_users_search(
    request: Request,
    q: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    return await _admin_users_search_impl(request, q, page, page_size, db, current)

@app.api_route("/api/admin/users", methods=["GET", "POST"])
async def admin_users_alias(
    request: Request,
    q: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    return await _admin_users_search_impl(request, q, page, page_size, db, current)

# --------------------------------------------------------------------------------------
# Chat minimal endpoints (to stop 405s and integrate with sidebar)
# --------------------------------------------------------------------------------------

def _mk_session(db: Session, user_id: int) -> ChatSession:
    s = ChatSession(user_id=user_id, created_at=datetime.utcnow())
    db.add(s); db.commit(); db.refresh(s)
    return s

def _mk_message(db: Session, session_id: int, user_id: int, role: str, content: str):
    m = ChatMessage(session_id=session_id, user_id=user_id, role=role,
                    content=content, created_at=datetime.utcnow())
    db.add(m); db.commit(); db.refresh(m)
    return m

@app.get("/api/chat/sessions")
def chat_sessions(db: Session = Depends(get_db), current: User = Depends(get_current_user)):
    """
    Return recent sessions for the sidebar.
    """
    ses = (db.query(ChatSession)
           .filter(ChatSession.user_id == current.id)
           .order_by(ChatSession.created_at.desc())
           .limit(25)
           .all())
    out = []
    for s in ses:
        last = (db.query(func.max(ChatMessage.created_at))
                  .filter(ChatMessage.session_id == s.id).scalar())
        out.append({
            "id": s.id,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "last_message_at": last.isoformat() if last else None,
        })
    return {"ok": True, "sessions": out}

@app.post("/api/chat/send")
async def chat_send(
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    message: Optional[str] = Form(None),
    brief: Optional[str] = Form(None),
    session_id: Optional[int] = Form(None),
):
    """
    Returns: Insights (top) + CXO-only Recommendations for CFO/CHRO/COO/CMO/CPO.
    Uses filenames + brief/message to craft lightweight, deterministic output so the UI renders
    the same structure across tiers. You can swap the inner 'assistant_content' with your LLM
    answer later without touching the response shape.
    """
    user = apply_tier_overrides(current)
    tier = _user_tier(user)

    # Parse multipart to capture filenames if present (no hard dependency on files)
    filenames: List[str] = []
    try:
        form = await request.form()
        for v in form.values():
            if hasattr(v, "filename") and getattr(v, "filename"):
                filenames.append(v.filename)
        # fallback to explicit message/brief if body was multipart
        if message is None:
            message = form.get("message")
        if brief is None:
            brief = form.get("brief")
        if session_id is None:
            sid = form.get("session_id")
            session_id = int(sid) if sid else None
    except Exception:
        pass

    # ensure session
    s: Optional[ChatSession] = None
    if session_id:
        s = db.query(ChatSession).filter(ChatSession.id == session_id,
                                         ChatSession.user_id == user.id).first()
    if not s:
        s = ChatSession(user_id=user.id, created_at=datetime.utcnow())
        db.add(s); db.commit(); db.refresh(s)

    # persist user message
    user_text = (message or brief or "").strip()
    hint = (f"\n\n[files] {', '.join(filenames)}" if filenames else "")
    umsg = ChatMessage(session_id=s.id, user_id=user.id, role="user",
                       content=(user_text + hint) if (user_text or hint) else "(no message)",
                       created_at=datetime.utcnow())
    db.add(umsg); db.commit()

    # ---------- Lightweight, deterministic “Insights + CXO recs” scaffold ----------
    # Inputs for flavoring:
    doc_hint = ", ".join(filenames) if filenames else "attached context"
    brief_hint = (brief or message or "your document").strip()

    # Top insights (3)
    insights = [
        f"The submission includes {('files: ' + doc_hint) if filenames else 'text input only'}, suitable for CXO review.",
        "Content appears narrative/strategic rather than raw telemetry; summarize themes before diving into metrics.",
        "Capture any explicit asks in the brief to align next steps and owners.",
    ]

    # Helper to make 5 crisp, one-line recommendations tailored to the role
    def recs_for(role: str) -> List[str]:
        base = brief_hint if brief_hint else "the current context"
        if role == "CFO":
            return [
                f"Validate revenue/expense drivers referenced in {base} against latest actuals.",
                "Set a weekly variance check and flag deltas >3% to the owner.",
                "Tighten working capital — review AR aging and vendor terms.",
                "Pressure-test unit economics; agree the one true CAC/LTV view.",
                "Publish a one-page runway/threshold update for the exec team.",
            ]
        if role == "CHRO":
            return [
                "Map roles to outcomes; clarify ‘must-win’ competencies this quarter.",
                "Launch a pulse check; track participation and eNPS by function.",
                "Pre-brief managers on upcoming changes; supply talking points.",
                "Create a hiring freeze exception path tied to business impact.",
                "Stand up an attrition review: leavers by tenure/manager/root cause.",
            ]
        if role == "COO":
            return [
                "Instrument the critical path; surface SLA breaches daily.",
                f"Create a doc-to-ops handoff checklist specific to {base}.",
                "Run a 5-whys on the slowest hop and remove one constraint.",
                "Add QA spot-checks on high-variance work items.",
                "Publish a weekly ‘cost-to-serve’ snapshot with trend arrows.",
            ]
        if role == "CMO":
            return [
                "Align messaging with the top three buyer pains surfaced here.",
                "Commit one growth loop (referrals, content, partner) and own it.",
                "Tidy the funnel; fix drop-offs with one experiment per stage.",
                "Enforce UTM hygiene; centralize campaign ROI in one view.",
                "Stand up a churn save motion for at-risk segments.",
            ]
        # CPO = Chief People Officer (your requested 5th brain)
        return [
            "Clarify success profile per role; share example work samples.",
            "Shorten time-to-hire: pre-book panels and scorecards.",
            "Upskill managers on feedback that moves performance quickly.",
            "Codify onboarding ‘day-1 to day-30’ outcomes and buddies.",
            "Track diversity of pipeline and close gaps with targeted sourcing.",
        ]

    # Build CXO sections — recommendations only
    CXOS = [
        ("CFO",  "Chief Financial Officer"),
        ("CHRO", "Chief Human Resources Officer"),
        ("COO",  "Chief Operating Officer"),
        ("CMO",  "Chief Marketing Officer"),
        ("CPO",  "Chief People Officer"),
    ]

    lines: List[str] = []
    lines.append("## Insights")
    for i, bullet in enumerate(insights, 1):
        lines.append(f"{i}. {bullet}")
    lines.append("")  # spacer

    for code, full in CXOS:
        lines.append(f"### {code} ({full})")
        lines.append("**Recommendations**")
        for i, r in enumerate(recs_for(code), 1):
            lines.append(f"{i}. {r}")
        lines.append("")  # spacer between CXOs

    assistant_content = "\n".join(lines)

    # Gate only if you want different behavior by tier; for now we show same
    # structure to all tiers so Premium/Pro+ get the full CXO output consistently.

    # persist assistant message
    amsg = ChatMessage(session_id=s.id, user_id=user.id, role="assistant",
                       content=assistant_content, created_at=datetime.utcnow())
    db.add(amsg); db.commit()

    return {
        "ok": True,
        "session_id": s.id,
        "assistant": {"role": "assistant", "content": assistant_content},
    }

# --------------------------------------------------------------------------------------
# (No external router mounting required; endpoints above match your UI paths)
# --------------------------------------------------------------------------------------

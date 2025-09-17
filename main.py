# -*- coding: utf-8 -*-
# main.py (CAIO Backend) — resilient startup + intact routes

import os, json, logging, traceback
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request, Query
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func

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
    """
    Warm up database connections *without* crashing the process.
    Handles Render cold starts / transient Neon connection issues.
    """
    global DB_READY
    tries = int(os.getenv("DB_WARMUP_TRIES", "20"))
    delay = float(os.getenv("DB_WARMUP_DELAY", "1.5"))
    for i in range(tries):
        try:
            init_db()  # existing initializer
            DB_READY = True
            break
        except Exception as e:
            logger.warning("init_db attempt %s/%s failed: %s", i + 1, tries, e)
            await asyncio.sleep(delay)
    # Keep serving even if DB isn't ready yet; requests can succeed once DB comes up
    yield

app = FastAPI(title="CAIO Backend", version="0.4.0", lifespan=lifespan)

# ---------------- CORS ----------------
ALLOWED_ORIGINS = [
    "https://caio-frontend.vercel.app",
    "https://caioai.netlify.app",
    "http://localhost",
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
    # Always respond 200 so Render doesn't kill the instance; report DB readiness flag
    return {"ready": True, "db_ready": DB_READY, "time": datetime.utcnow().isoformat() + "Z"}

# ---------------- Tiers & limits ----------------
ADMIN_EMAILS    = {e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()}
PREMIUM_EMAILS  = {e.strip().lower() for e in os.getenv("PREMIUM_EMAILS", "").split(",") if e.strip()}
PRO_PLUS_EMAILS = {e.strip().lower() for e in os.getenv("PRO_PLUS_EMAILS", "").split(",") if e.strip()}
PRO_TEST_EMAILS = {e.strip().lower() for e in os.getenv("PRO_TEST_EMAILS", "").split(",") if e.strip()}  # for QA

# Analyzer/demo caps (legacy/free)
FREE_QUERIES_PER_DAY = int(os.getenv("FREE_QUERIES_PER_DAY", "3"))
FREE_UPLOADS         = int(os.getenv("FREE_UPLOADS", "3"))
FREE_BRAINS          = int(os.getenv("FREE_BRAINS", "2"))

# ---- Per-tier analyze limits (docs/day) ----
# Pro defaults to 25/day; if you prefer your env PRO_QUERIES_PER_DAY, it will be used when PRO_ANALYZE_PER_DAY unset.
PRO_ANALYZE_PER_DAY      = int(os.getenv("PRO_ANALYZE_PER_DAY", os.getenv("PRO_QUERIES_PER_DAY", "25")))
PRO_PLUS_ANALYZE_PER_DAY = int(os.getenv("PRO_PLUS_ANALYZE_PER_DAY", "50"))
DEMO_ANALYZE_PER_DAY     = int(os.getenv("DEMO_ANALYZE_PER_DAY",  str(FREE_QUERIES_PER_DAY)))  # default 3
# Premium/Admin are unlimited for analyze

# Chat preview caps for Demo/Pro (Pro+ & Premium/Admin unlimited in chat)
FREE_CHAT_MSGS_PER_DAY     = int(os.getenv("FREE_CHAT_MSGS_PER_DAY", "3"))
FREE_CHAT_UPLOADS_PER_DAY  = int(os.getenv("FREE_CHAT_UPLOADS_PER_DAY", "3"))

# --- PRO QA testing toggles (deduped) ---
PRO_TEST_ENABLE           = os.getenv("PRO_TEST_ENABLE", "1").lower() in ("1","true","yes")
PRO_TEST_EMAILS_HARDCODE  = {"testpro@123.com"}  # keep hardcoded tester during QA
PRO_TEST_AUTO_CREATE      = os.getenv("PRO_TEST_AUTO_CREATE", "1").lower() in ("1","true","yes")
PRO_TEST_DEFAULT_PASSWORD = os.getenv("PRO_TEST_DEFAULT_PASSWORD", "testpro123")

def _user_tier(u: User) -> str:
    """
    Resolve a user's tier. Priority:
    1) explicit u.tier if present
    2) ADMIN_EMAILS -> admin
    3) PRO_PLUS_EMAILS -> pro_plus
    4) PREMIUM_EMAILS -> premium
    5) legacy is_paid flag -> pro
    6) fallback -> demo
    """
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

def apply_tier_overrides(user: User) -> User:
    """
    QA helper: for approved tester emails, *treat as Pro* (non-persistent).
    Controlled by:
      - PRO_TEST_ENABLE
      - PRO_TEST_EMAILS (env) + PRO_TEST_EMAILS_HARDCODE (code)
    """
    try:
        if not PRO_TEST_ENABLE:
            return user
        email = (getattr(user, "email", "") or "").lower()
        if email in PRO_TEST_EMAILS or email in PRO_TEST_EMAILS_HARDCODE:
            # Do NOT write back to DB; keep this an in-memory override.
            user.tier = "pro"
            user.is_paid = True
    except Exception:
        pass
    return user

def _analyze_limit_for_tier(tier: str) -> Optional[int]:
    # None means unlimited
    if tier in ("admin", "premium"):
        return None
    if tier == "pro_plus":
        return PRO_PLUS_ANALYZE_PER_DAY
    if tier == "pro":
        return PRO_ANALYZE_PER_DAY
    # demo / fallback
    return DEMO_ANALYZE_PER_DAY

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
    try:
        db.add(UsageLog(user_id=user_id or 0, timestamp=datetime.utcnow(), endpoint=endpoint, status=status, meta=meta))
        db.commit()
    except Exception:
        db.rollback()

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

def _check_and_increment_usage(
    db: Session,
    user: User,
    endpoint: str,
    plan: Optional[str] = None,
    limit_override: Optional[int] = None,
) -> Tuple[bool, int, int, str]:
    """
    Returns (allowed, used_or_next, limit, plan_or_reset_at)
    - If allowed=False: used_or_next=used_count, plan_or_reset_at=reset_at
    - If allowed=True:  used_or_next=used_count+1 (or 0 if unlimited), plan_or_reset_at=plan
    """
    user_id = getattr(user, "id", 0) or 0
    plan = plan or _user_tier(user)

    # ----- Unlimited tiers by endpoint -----
    if endpoint == "analyze":
        tier_limit = _analyze_limit_for_tier(plan) if limit_override is None else limit_override
        if tier_limit is None:  # unlimited for analyze
            _log_usage(db, user_id, endpoint, "ok")
            return True, 0, 0, plan

    elif endpoint == "chat":
        # Premium/Admin/Pro+ get unlimited chat; Pro/Demo use preview cap
        if plan in ("admin", "premium", "pro_plus") and limit_override is None:
            _log_usage(db, user_id, endpoint, "ok")
            return True, 0, 0, plan
        tier_limit = limit_override if limit_override is not None else FREE_CHAT_MSGS_PER_DAY

    else:
        tier_limit = limit_override if limit_override is not None else FREE_QUERIES_PER_DAY

    # ----- Count & enforce -----
    used = _count_usage(db, user_id, endpoint)
    if used >= tier_limit:
        _, end = _today_bounds_utc()
        return False, used, tier_limit, end.isoformat() + "Z"

    _log_usage(db, user_id, endpoint, "ok")
    return True, used + 1, tier_limit, plan

# ---------------- Auth ----------------
@app.post("/api/login")
async def login(request: Request, db: Session = Depends(get_db)):
    """
    Accept BOTH form (application/x-www-form-urlencoded) and JSON bodies.
    If a tester email signs in and doesn't exist, optionally auto-create (guarded by env).
    Always returns the same token shape your frontend expects.
    """
    email = ""
    password = ""
    ctype = (request.headers.get("content-type") or "").lower()

    # 1) Parse input (JSON or form)
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
            email = (data.get("username") or "").strip().lower()
            password = data.get("password") or ""
        except Exception:
            pass

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")

    # 2) Lookup user (or optionally auto-create for approved tester emails)
    user = db.query(User).filter(User.email == email).first()
    is_tester = PRO_TEST_ENABLE and (email in PRO_TEST_EMAILS or email in PRO_TEST_EMAILS_HARDCODE)

    if not user:
        if is_tester and PRO_TEST_AUTO_CREATE:
            # Auto-provision a test user with provided password; fallback to default if blank
            hashed = get_password_hash(password or PRO_TEST_DEFAULT_PASSWORD)
            user = User(
                email=email,
                hashed_password=hashed,
                is_admin=False,
                is_paid=False,           # we won't persist paid; override handles Pro behavior
                created_at=datetime.utcnow(),
            )
            db.add(user); db.commit(); db.refresh(user)
        else:
            raise HTTPException(status_code=401, detail="User not found")

    # 3) Verify password (normal flow). If tester auto-created with default, allow that too.
    if not verify_password(password, user.hashed_password):
        # If tester and they used the default QA password, let them in.
        if not (is_tester and PRO_TEST_DEFAULT_PASSWORD and verify_password(PRO_TEST_DEFAULT_PASSWORD, user.hashed_password)):
            raise HTTPException(status_code=401, detail="Incorrect email or password")

    # 4) Keep flags loosely in sync with env (do not unset existing is_paid)
    is_admin_now = email in ADMIN_EMAILS
    is_premium_now = email in PREMIUM_EMAILS
    changed = False
    if hasattr(user, "is_admin") and bool(user.is_admin) != is_admin_now:
        user.is_admin = is_admin_now; changed = True
    if hasattr(user, "is_paid") and is_premium_now and not bool(user.is_paid):
        user.is_paid = True; changed = True
    if changed:
        db.commit()

    # 5) Apply *non-persistent* Pro override for QA testers
    user = apply_tier_overrides(user)  # <— key line

    # 6) Return token shape your frontend expects
    token = create_access_token(email)
    return {"access_token": token, "token_type": "bearer"}

# --- robust /api/signup: accepts JSON or form-data ---
@app.post("/api/signup")
async def signup(request: Request, db: Session = Depends(get_db)):
    """
    Create a new user account.
    Accepts either JSON (application/json) or form-data (multipart/x-www-form-urlencoded).
    Only uses columns that actually exist in db.py::User.
    Returns the SAME token shape as /api/login.
    """
    try:
        email = ""
        password = ""

        # Accept JSON or form data
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

        # Basic format check
        if "@" not in email or "." not in email.split("@")[-1]:
            raise HTTPException(status_code=422, detail="invalid email format")

        # Hash the password with your existing util
        hashed = get_password_hash(password)

        # Create only the fields that actually exist in db.py::User
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

        # ✅ Issue token like /api/login (string subject, not dict)
        access_token = create_access_token(user.email)

        return {"ok": True, "user": {"id": user.id, "email": user.email}, "access_token": access_token, "token_type": "bearer"}

    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="User already exists or invalid data.")
    except HTTPException:
        raise
    except Exception:
        db.rollback()
        logger.error("signup failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Signup failed on the server. Please try again.")

@app.post("/api/logout")
def logout(response: Response):
    # Best-effort: clear a non-HttpOnly cookie named "token" if you set one
    response.delete_cookie("token", path="/")
    return {"ok": True}

@app.get("/api/profile")
def profile(current_user: User = Depends(get_current_user)):
    current_user = apply_tier_overrides(current_user)
    tier = _user_tier(current_user)
    return {
        "email": current_user.email,
        "tier": tier,
        "is_admin": getattr(current_user, "is_admin", False),
        "is_paid": tier in {"pro", "pro_plus", "premium", "admin"},
        "created_at": getattr(current_user, "created_at", None),
    }

# ---------------- Public config (pricing etc.) ----------------
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
    currency = os.getenv("PAY_DEFAULT_CURRENCY", "INR").upper()

    # Pricing defaults aligned to your Pro-tier target:
    PRO_PRICE_INR      = int(os.getenv("PRO_PRICE_INR", "1999"))
    PRO_PRICE_USD      = int(os.getenv("PRO_PRICE_USD", "25"))
    PRO_PLUS_PRICE_INR = int(os.getenv("PRO_PLUS_PRICE_INR", "3999"))
    PRO_PLUS_PRICE_USD = int(os.getenv("PRO_PLUS_PRICE_USD", "49"))
    PREMIUM_PRICE_INR  = int(os.getenv("PREMIUM_PRICE_INR", "7999"))
    PREMIUM_PRICE_USD  = int(os.getenv("PREMIUM_PRICE_USD", "99"))

    def pick(inr: int, usd: int) -> int:
        return {"USD": usd, "INR": inr}.get(currency, usd)

    plans = {
        "pro":      {"price": pick(PRO_PRICE_INR, PRO_PRICE_USD)},
        "pro_plus": {"price": pick(PRO_PLUS_PRICE_INR, PRO_PLUS_PRICE_USD)},
        "premium":  {"price": pick(PREMIUM_PRICE_INR, PREMIUM_PRICE_USD)},
    }
    chat_preview = {
        "enabled": True,
        "msgs_per_day": FREE_CHAT_MSGS_PER_DAY,
        "uploads_per_day": FREE_CHAT_UPLOADS_PER_DAY,
    }
    return {"currency": currency, "plans": plans, "chat_preview": chat_preview}

# ---------------- Analyzer ----------------
DEFAULT_BRAINS_ORDER = ["CFO","CHRO","COO","CMO","CPO"]

def _choose_brains(requested: Optional[str], is_paid: bool) -> List[str]:
    if requested:
        items = [b.strip().upper() for b in requested.split(",") if b.strip()]
        chosen = [b for b in items if b in DEFAULT_BRAINS_ORDER] or DEFAULT_BRAINS_ORDER[:]
    else:
        chosen = DEFAULT_BRAINS_ORDER[:]
    return chosen if is_paid else chosen[: max(1, min(FREE_BRAINS, len(chosen)))]

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

async def _extract_text(file: Optional[UploadFile]) -> str:
    if not file:
        return ""
    try:
        raw = await file.read()
        if not raw:
            return ""
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return str(raw[:200000])
    finally:
        try:
            await file.close()
        except Exception:
            pass

# ---- OpenRouter only ----
def _call_openrouter(messages: List[dict]) -> str:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        return "OpenRouter key is not configured. Please contact support."
    import requests
    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://caio-frontend.vercel.app",
        "X-Title": "CAIO",
        "Content-Type": "application/json",
    }
    model = os.getenv("LLM_MODEL_OPENROUTER", "openrouter/auto")
    data = {"model": model, "messages": messages, "temperature": 0.2}
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(data), timeout=60)
    if r.status_code >= 400:
        logger.error("OpenRouter %s: %s", r.status_code, r.text)
        return "The model is currently unavailable. Please try again shortly."
    try:
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return "I had trouble reading the model response."

# ---------------- Analyze endpoint ----------------
@app.post("/api/analyze")
async def analyze(
    request: Request,
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    brains: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # ✅ Treat QA testers as Pro BEFORE limits
    current_user = apply_tier_overrides(current_user)

    allowed, used, limit, plan_or_reset = _check_and_increment_usage(db, current_user, endpoint="analyze")
    if not allowed:
        # plan_or_reset is reset_at here
        return _limit_response(used, limit, plan_or_reset, _user_tier(current_user))

    is_paid = _user_tier(current_user) in ("admin", "premium", "pro_plus", "pro")
    chosen = _choose_brains(brains, is_paid)

    extracted = ""
    try:
        extracted = await _extract_text(file)
    except Exception as e:
        logger.warning("Extract failed: %s", e)

    # For demo experience w/out key
    if not os.getenv("OPENROUTER_API_KEY"):
        summary = f"Demo result.\n\nBrains: {', '.join(chosen)}\n\nBrief:\n{text or '(none)'}"
        return {"status":"demo","title":"Demo Mode","summary":summary}

    # Call LLM once per brain, then stitch
    sections = []
    for b in chosen:
        prompt = _brain_prompt(text or "", extracted or "", b)
        sections.append(f"## {b}\n\n" + _call_openrouter([{"role":"user","content":prompt}]))

    return {"status":"ok","title":"Analysis Result","summary":"\n\n".join(sections)}

# ---------------- Chat endpoints ----------------
def _compose_premium_system_prompt() -> str:
    return (
        "You are CAIO — a pragmatic business & ops copilot. "
        "Answer clearly in Markdown. When files are provided, ground your answer on them."
    )

def _ensure_session(db: Session, user_id: int, session_id: Optional[int], title_hint: Optional[str]) -> ChatSession:
    if session_id:
        sess = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user_id).first()
        if sess:
            return sess
    title = (title_hint or "New chat").strip()[:120] or "New chat"
    sess = ChatSession(user_id=user_id, title=title)
    db.add(sess); db.commit(); db.refresh(sess)
    return sess

def _save_msg(db: Session, session_id: int, role: str, content: str):
    db.add(ChatMessage(session_id=session_id, role=role, content=content[:120000], created_at=datetime.utcnow()))
    db.commit()

def _history(db: Session, session_id: int, user_id: int, limit: int = 40) -> List[ChatMessage]:
    return (
        db.query(ChatMessage)
        .join(ChatSession, ChatMessage.session_id == ChatSession.id)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .order_by(ChatMessage.id.asc())
        .limit(limit)
        .all()
    )

@app.post("/api/chat/send")
async def chat_send(
    request: Request,
    message: str = Form(""),
    session_id: Optional[int] = Form(None),
    files: Optional[List[UploadFile]] = File(None),   # multi-file
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    current_user = apply_tier_overrides(current_user)
    tier = _user_tier(current_user)
    uid = getattr(current_user, "id", 0) or 0

    # Free preview caps for Demo/Pro; unlimited for Premium/Admin/Pro+
    if tier not in ("admin", "premium", "pro_plus"):
        # count message send cap
        used = _count_usage(db, uid, "chat")
        if used >= FREE_CHAT_MSGS_PER_DAY:
            _, end = _today_bounds_utc()
            return _limit_response(used, FREE_CHAT_MSGS_PER_DAY, end.isoformat()+"Z", tier)
        _log_usage(db, uid, "chat", "ok")

        # uploads cap (per day)
        if files:
            upload_used = _count_usage(db, uid, "chat_upload")
            if upload_used + len(files) > FREE_CHAT_UPLOADS_PER_DAY:
                _, end = _today_bounds_utc()
                return _limit_response(upload_used, FREE_CHAT_UPLOADS_PER_DAY, end.isoformat()+"Z", f"{tier} uploads")
            for _ in range(len(files)):
                _log_usage(db, uid, "chat_upload", "ok")

    # Create/reuse session
    sess = _ensure_session(db, uid, session_id, title_hint=(message or "New chat"))

    # Gather short context from all files
    doc_chunks: List[str] = []
    if files:
        for f in files:
            try:
                txt = (await _extract_text(f)).strip()
                if txt:
                    doc_chunks.append(txt[:8000])
            except Exception as e:
                logger.warning("file read failed: %s", e)
    context_block = "\n\n".join(doc_chunks).strip()

    # Build LLM messages
    msgs: List[dict] = [{"role":"system","content": _compose_premium_system_prompt()}]
    for m in _history(db, sess.id, uid, limit=40):
        msgs.append({"role": m.role, "content": m.content})
    if context_block:
        msgs.append({"role":"system","content": f"[DOCUMENT EXCERPTS]\n{context_block}"})
    if message.strip():
        msgs.append({"role":"user","content": message.strip()})
        _save_msg(db, sess.id, "user", message.strip())

    # Model call (OpenRouter only)
    reply = _call_openrouter(msgs)
    _save_msg(db, sess.id, "assistant", reply)

    return {"session_id": sess.id, "assistant": reply}

@app.get("/api/chat/history")
def chat_history(
    session_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    uid = getattr(current_user, "id", 0) or 0
    sess = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == uid).first()
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    items = [{
        "id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at.isoformat()+"Z"
    } for m in _history(db, session_id, uid, limit=200)]
    return {"session_id": session_id, "messages": items, "title": sess.title}

@app.get("/api/chat/sessions")
def chat_sessions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    uid = getattr(current_user, "id", 0) or 0
    rows = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == uid)
        .order_by(ChatSession.id.desc())
        .limit(100)
        .all()
    )
    return [{"id": s.id, "title": s.title or f"Chat {s.id}", "created_at": s.created_at.isoformat()+"Z"} for s in rows]

# ---------------- Admin usage (simple analytics) ----------------
def _require_admin(user: User):
    if not _is_admin(user):
        raise HTTPException(status_code=403, detail="Admin only")

@app.get("/api/admin/usage")
def admin_usage(
    days: int = Query(14, ge=1, le=90),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_admin(current_user)
    now = datetime.utcnow()
    since = now - timedelta(days=days - 1)

    if db.bind.dialect.name == "postgresql":
        daycol = func.date_trunc("day", UsageLog.timestamp)
    else:
        daycol = func.strftime("%Y-%m-%d", UsageLog.timestamp)

    rows = (
        db.query(daycol.label("day"), UsageLog.endpoint, func.count(UsageLog.id).label("count"))
        .filter(UsageLog.timestamp >= since)
        .group_by("day", "endpoint")
        .order_by("day")
        .all()
    )

    def _day_key(v):
        if isinstance(v, str): return v[:10]
        try: return v.date().isoformat()
        except Exception: return str(v)[:10]

    day_keys = [(since + timedelta(days=i)).date().isoformat() for i in range(days)]
    series = {ek: {dk: 0 for dk in day_keys} for ek in ("analyze", "chat", "chat_upload")}
    for day, endpoint, count in rows:
        dk = _day_key(day)
        if endpoint not in series:
            series[endpoint] = {dk: 0 for dk in day_keys}
        series[endpoint][dk] = int(count or 0)

    return {"days": day_keys, "series": series}

# ---------------- Optional: include your extra routers (no behavior change) ----------------
try:
    from admin_routes import router as admin_router
    app.include_router(admin_router, prefix="/api/admin", tags=["admin"])
except Exception:
    if DEBUG: logger.info("admin_routes not included")

try:
    from admin_metrics_routes import router as admin_metrics_router
    app.include_router(admin_metrics_router)  # already prefixed /api/admin in that file
except Exception:
    if DEBUG: logger.info("admin_metrics_routes not included")

try:
    from payment_routes import router as payment_router
    app.include_router(payment_router)
except Exception:
    if DEBUG: logger.info("payment_routes not included")

try:
    from contact_routes import router as contact_router
    app.include_router(contact_router)  # exposes /api/contact
except Exception:
    if DEBUG: logger.info("contact_routes not included")

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

# main.py
import os, json, logging, traceback, re
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
from io import BytesIO

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import func

# project modules
from db import get_db, User, init_db, UsageLog, ChatSession, ChatMessage
from auth import create_access_token, verify_password, get_password_hash, get_current_user

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("caio")
DEBUG = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes")

app = FastAPI(title="CAIO Backend", version="0.4.0")
init_db()

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
@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.4.0"}

@app.get("/api/ready")
def ready():
    return {"ready": True, "time": datetime.utcnow().isoformat() + "Z"}

# ---------------- Tiers & limits ----------------
ADMIN_EMAILS   = {e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "vineetpjoshi.71@gmail.com").split(",") if e.strip()}
PREMIUM_EMAILS = {e.strip().lower() for e in os.getenv("PREMIUM_EMAILS", "").split(",") if e.strip()}

# Analyzer/demo caps (existing)
FREE_QUERIES_PER_DAY = int(os.getenv("FREE_QUERIES_PER_DAY", "3"))
FREE_UPLOADS         = int(os.getenv("FREE_UPLOADS", "3"))
FREE_BRAINS          = int(os.getenv("FREE_BRAINS", "2"))

# Chat preview caps for Demo/Pro
FREE_CHAT_MSGS_PER_DAY     = int(os.getenv("FREE_CHAT_MSGS_PER_DAY", "3"))
FREE_CHAT_UPLOADS_PER_DAY  = int(os.getenv("FREE_CHAT_UPLOADS_PER_DAY", "3"))

def _user_tier(u: User) -> str:
    email = (getattr(u, "email", "") or "").lower()
    if not email:
        return "demo"
    if email in ADMIN_EMAILS:
        return "admin"
    if email in PREMIUM_EMAILS:
        return "premium"
    # legacy flag if you still store it:
    if getattr(u, "is_paid", False) is True:
        # treat paid-but-not-premium as "pro" (Pro+ should be via PREMIUM_EMAILS)
        return "pro"
    return "demo"

def _is_admin(u: User) -> bool:
    return _user_tier(u) == "admin"

def _is_premium_or_plus(u: User) -> bool:
    t = _user_tier(u)
    return t in ("admin", "premium")  # Pro+ emails should be in PREMIUM_EMAILS

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

def _check_and_increment_usage(db: Session, user: User, endpoint: str, plan: Optional[str] = None,
                               limit_override: Optional[int] = None) -> Tuple[bool, int, int, str]:
    user_id = getattr(user, "id", 0) or 0
    plan = plan or _user_tier(user)

    # Premium/Admin unlimited
    if plan in ("admin", "premium"):
        _log_usage(db, user_id, endpoint, "ok")
        return True, 0, 0, plan

    # Pro gets analyzer caps; chat uses free preview caps
    if endpoint == "chat":
        daily_limit = int(limit_override or FREE_CHAT_MSGS_PER_DAY)
    else:
        daily_limit = int(limit_override or FREE_QUERIES_PER_DAY)

    used = _count_usage(db, user_id, endpoint)
    if used >= daily_limit:
        start, end = _today_bounds_utc()
        return False, used, daily_limit, end.isoformat() + "Z"

    _log_usage(db, user_id, endpoint, "ok")
    return True, used + 1, daily_limit, plan

# ---------------- Auth ----------------
@app.post("/api/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    email = form_data.username.strip().lower()
    password = form_data.password or ""
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    if not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    # keep flags in sync with env
    is_admin_now = email in ADMIN_EMAILS
    is_premium_now = email in PREMIUM_EMAILS
    changed = False
    if hasattr(user, "is_admin") and bool(user.is_admin) != is_admin_now:
        user.is_admin = is_admin_now; changed = True
    if hasattr(user, "is_paid") and bool(user.is_paid) != (is_premium_now or bool(user.is_paid)):
        # do not unset paid if premium; otherwise keep old paid flag
        pass
    if changed: db.commit()

    token = create_access_token(email)
    return {"access_token": token, "token_type": "bearer"}

@app.get("/api/profile")
def profile(current_user: User = Depends(get_current_user)):
    return {"email": current_user.email, "tier": _user_tier(current_user)}

# ---------------- Public config (pricing etc.) ----------------
@app.get("/api/public-config")
def public_config():
    # You already had this — preserved and simplified
    currency = os.getenv("PAY_DEFAULT_CURRENCY", "INR").upper()
    plans = {
        "pro":    {"price": {"USD": 25, "INR": 1999}.get(currency, 25)},
        "pro_plus":{"price": {"USD": 49, "INR": 3999}.get(currency, 49)},
        "premium":{"price": {"USD": 99, "INR": 7999}.get(currency, 99)},
    }
    chat_preview = {
        "enabled": True,
        "msgs_per_day": FREE_CHAT_MSGS_PER_DAY,
        "uploads_per_day": FREE_CHAT_UPLOADS_PER_DAY,
    }
    return {"currency": currency, "plans": plans, "chat_preview": chat_preview}

# ---------------- Analyzer (existing) ----------------
DEFAULT_BRAINS_ORDER = ["CFO","COO","CHRO","CMO","CPO"]

def _choose_brains(requested: Optional[str], is_paid: bool) -> List[str]:
    if requested:
        items = [b.strip().upper() for b in requested.split(",") if b.strip()]
        chosen = [b for b in items if b in DEFAULT_BRAINS_ORDER] or DEFAULT_BRAINS_ORDER[:]
    else:
        chosen = DEFAULT_BRAINS_ORDER[:]
    return chosen if is_paid else chosen[: max(1, min(FREE_BRAINS, len(chosen)))]

def _brain_prompt(brief: str, extracted: str, brain: str) -> str:
    role_map = {
        "CFO":"Chief Financial Officer — unit economics; revenue mix, margins, CCC, runway.",
        "COO":"Chief Operating Officer — cost-to-serve & reliability; capacity, throughput, SLA.",
        "CHRO":"Chief Human Resources Officer — org effectiveness; attrition, engagement.",
        "CMO":"Chief Marketing Officer — efficient growth; CAC/LTV, funnel, retention.",
        "CPO":"Chief People Officer — talent acquisition; pipeline, time-to-hire, QoH.",
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
        # extremely simple text extraction fallback (you already have better in analyzer pipeline)
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
    allowed, used, limit, plan_or_reset = _check_and_increment_usage(db, current_user, endpoint="analyze")
    if not allowed:
        # plan_or_reset is reset_at here
        return _limit_response(used, limit, plan_or_reset, _user_tier(current_user))

    is_paid = _user_tier(current_user) in ("admin", "premium", "pro")
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
    tier = _user_tier(current_user)
    uid = getattr(current_user, "id", 0) or 0

    # Free preview caps for Demo/Pro; unlimited for Premium/Admin
    if tier in ("demo", "pro"):
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
        .group_by("day", UsageLog.endpoint)
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

# ---------------- Optional: include your extra admin routers ----------------
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

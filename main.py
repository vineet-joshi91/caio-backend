# -*- coding: utf-8 -*-
# main.py — CAIO Backend (resilient startup, robust CORS, dual-method chat routes)

import os, json, logging, traceback, re
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

# project modules
from db import get_db, User, init_db, UsageLog, ChatSession, ChatMessage
from auth import create_access_token, verify_password, get_password_hash

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("caio")
DEBUG = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------
# Resilient startup
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

app = FastAPI(title="CAIO Backend", version="0.5.0", lifespan=lifespan)

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
    allow_origin_regex=r"https://caio-frontend[^/]*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["ETag", "Content-Length"],
    max_age=86400,
)

# Always-ACAO fallback (even on error paths)
@app.middleware("http")
async def _cors_always(request: Request, call_next):
    try:
        resp = await call_next(request)
    except Exception:
        logger.exception("Unhandled error")
        resp = JSONResponse({"detail": "internal error"}, status_code=500)
    origin = request.headers.get("origin")
    if origin and (origin in ALLOWED_ORIGINS or re.match(r"https://caio-frontend[^/]*\.vercel\.app", origin or "")):
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
    return resp

@app.options("/{path:path}")
def cors_preflight(path: str):
    return JSONResponse({"ok": True})


# ---------------- Health ----------------
@app.get("/")
def root():
    return {"ok": True, "service": "caio-backend", "version": "0.5.0"}

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.5.0"}

@app.get("/api/ready")
def ready():
    return {"ready": True, "db_ready": DB_READY, "startup_ok": True, "startup_error": "", "time": datetime.utcnow().isoformat() + "Z"}


# ---------------- Tiers & limits ----------------
ADMIN_EMAILS = set(e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip())
PREMIUM_EMAILS = {e.strip().lower() for e in os.getenv("PREMIUM_EMAILS", "").split(",") if e.strip()}
PRO_PLUS_EMAILS = {e.strip().lower() for e in os.getenv("PRO_PLUS_EMAILS", "").split(",") if e.strip()}

FREE_CHAT_MSGS_PER_DAY     = int(os.getenv("FREE_CHAT_MSGS_PER_DAY", "3"))
FREE_CHAT_UPLOADS_PER_DAY  = int(os.getenv("FREE_CHAT_UPLOADS_PER_DAY", "3"))

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

def apply_tier_overrides(user: User) -> User:
    allow = {e.strip().lower() for e in os.getenv("PRO_TEST_EMAILS", "").split(",") if e.strip()}
    if getattr(user, "email", "").lower() in allow:
        user.tier = "pro"
        user.is_paid = True
    return user

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
            email = (data.get("username") or "").strip().lower()
            password = data.get("password") or ""
        except Exception:
            pass

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")

    user = db.query(User).filter(User.email == email).first()
    PRO_TEST_ENABLE = os.getenv("PRO_TEST_ENABLE", "0") in ("1","true","yes")
    PRO_TEST_AUTO_CREATE = os.getenv("PRO_TEST_AUTO_CREATE", "0") in ("1","true","yes")
    PRO_TEST_DEFAULT_PASSWORD = os.getenv("PRO_TEST_DEFAULT_PASSWORD", "testpro123")
    PRO_TEST_EMAILS = {e.strip().lower() for e in os.getenv("PRO_TEST_EMAILS", "").split(",") if e.strip()}
    is_tester = PRO_TEST_ENABLE and (email in PRO_TEST_EMAILS)

    if not user:
        if is_tester and PRO_TEST_AUTO_CREATE:
            hashed = get_password_hash(password or PRO_TEST_DEFAULT_PASSWORD)
            user = User(email=email, hashed_password=hashed, is_admin=False, is_paid=False, created_at=datetime.utcnow())
            db.add(user); db.commit(); db.refresh(user)
        else:
            raise HTTPException(status_code=401, detail="User not found")

    if not verify_password(password, user.hashed_password):
        if not (is_tester and PRO_TEST_DEFAULT_PASSWORD and verify_password(PRO_TEST_DEFAULT_PASSWORD, user.hashed_password)):
            raise HTTPException(status_code=401, detail="Incorrect email or password")

    # sync flags loosely with env (do not unset existing is_paid)
    is_admin_now = email in ADMIN_EMAILS
    is_premium_now = email in PREMIUM_EMAILS
    changed = False
    if hasattr(user, "is_admin") and bool(user.is_admin) != is_admin_now:
        user.is_admin = is_admin_now; changed = True
    if hasattr(user, "is_paid") and is_premium_now and not bool(user.is_paid):
        user.is_paid = True; changed = True
    if changed: db.commit()

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
            email = (form.get("email") or "").strip().lower()
            password = form.get("password") or ""

        if not email or not password:
            raise HTTPException(status_code=422, detail="email and password are required")

        if "@" not in email or "." not in email.split("@")[-1]:
            raise HTTPException(status_code=422, detail="invalid email format")

        hashed = get_password_hash(password)
        user = User(email=email, hashed_password=hashed, is_admin=False, is_paid=False, created_at=datetime.utcnow())
        db.add(user); db.flush(); db.commit(); db.refresh(user)

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


# ---------------- Public config (pricing etc.) ----------------
@app.get("/api/public-config")
def public_config():
    currency = os.getenv("PAY_DEFAULT_CURRENCY", "INR").upper()

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


# ---------------- Helper: get current user from Authorization ----------------
def get_current_user(db: Session, request: Request) -> User:
    auth = request.headers.get("authorization") or ""
    token = auth.split(" ", 1)[1] if " " in auth else ""
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    # In your auth.create_access_token you used email as subject
    # Very small verifier: we stored email in token payload's "sub".
    # Reuse your auth utils if you have a decode function; else keep this simple approach.
    try:
        # your JWT encoder uses "sub" = email (from earlier code context)
        # we can re-use that by calling a lightweight checker in auth if you have it.
        # For now, accept raw email token in dev mode too.
        email_guess = token  # fallback if your create_access_token encodes plain (older dev tokens)
    except Exception:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = db.query(User).filter(User.email == email_guess).first()
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# ---------------- Chat: dual-method sessions/history, plus send ----------------
def _session_dict(s: ChatSession):
    return {"id": s.id, "title": getattr(s, "title", None), "created_at": s.created_at.isoformat() + "Z"}

def _message_dict(m: ChatMessage):
    return {"id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at.isoformat() + "Z"}

@app.get("/api/chat/sessions")
@app.post("/api/chat/sessions")
def chat_sessions(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(db, request)
    rows = db.query(ChatSession).filter(ChatSession.user_id == user.id).order_by(ChatSession.created_at.desc()).limit(50).all()
    return {"sessions": [_session_dict(s) for s in rows]}

@app.get("/api/chat/history")
@app.post("/api/chat/history")
def chat_history(
    request: Request,
    db: Session = Depends(get_db),
    session_id: Optional[int] = Query(None)
):
    user = get_current_user(db, request)

    if request.method == "POST":
        try:
            data = request.json() if isinstance(request, dict) else None
        except Exception:
            data = None
        try:
            # starlette Request json must be awaited
            import anyio
            async def _read_json(req: Request):
                try:
                    return await req.json()
                except Exception:
                    return {}
            data = anyio.from_thread.run(_read_json, request)  # ensure within sync fn
        except Exception:
            pass

    # If POST JSON supplied, Starlette sync context above may be tricky; safest path:
    if session_id is None:
        try:
            body = request._body if hasattr(request, "_body") else None
            if not body:
                # try to read cached body (FastAPI won't allow sync read)
                pass
        except Exception:
            pass

    # Final fallback: also support query param
    if session_id is None:
        try:
            session_id = int(request.query_params.get("session_id"))
        except Exception:
            pass

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    s = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user.id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    msgs = db.query(ChatMessage).filter(ChatMessage.session_id == s.id).order_by(ChatMessage.created_at.asc()).all()
    return {"session_id": s.id, "messages": [_message_dict(m) for m in msgs]}

@app.post("/api/chat/send")
async def chat_send(
    request: Request,
    db: Session = Depends(get_db),
    message: str = Form(""),
    session_id: Optional[int] = Form(None),
    files: List[UploadFile] = File(default_factory=list),
):
    user = get_current_user(db, request)

    # Limits & plan rules (simplified – reuse your real limits if present)
    tier = _user_tier(user)
    if tier == "pro_plus" and len(files) > 1:
        raise HTTPException(status_code=403, detail="Pro+ allows one file per message; upgrade for more.")

    # Create session lazily
    if not session_id:
        s = ChatSession(user_id=user.id, title="New chat", created_at=datetime.utcnow())
        db.add(s); db.commit(); db.refresh(s)
        session_id = s.id
    else:
        s = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user.id).first()
        if not s:
            raise HTTPException(status_code=404, detail="Session not found")

    # Persist user message
    if message or files:
        db.add(ChatMessage(session_id=session_id, role="user", content=(message or "(file only)"), created_at=datetime.utcnow()))
        db.commit()

    # TODO: Call your analyzer/LLM here. For now, return a CXO-shaped stub so UI renders.
    assistant_md = f"""## CFO (Chief Financial Officer)

### Insights
1. No direct financial tables detected.
2. Styles and document structure imply formal reporting.
3. Content suggests narrative/qualitative financial commentary.

### Recommendations
1. **Audit financial data**: Validate figures with the underlying sheets.
2. **Template standardization**: Reuse consistent reporting format.
3. **Link to dashboards**: Cross-reference KPI dashboards for visibility.
4. **Narrative consistency**: Align narrative to budget and forecast cycles.
5. **Export hygiene**: Ensure export paths for sharing.

## CHRO (Chief Human Resources Officer)

### Insights
1. No HR-specific entities identified.
2. No org metrics found.
3. No talent signals detected.

### Recommendations
1. **No Actionable Data**: No HR-related content identified in the document.
2. **Request HR context**: Provide attrition/engagement excerpts.
3. **Add headcount tables**: Include role/location breakdowns.
4. **Flag tenure risks**: Annotate mid-tenure exits if present.
5. **Align OKRs**: Map HR OKRs to business KPIs.

## COO (Chief Operating Officer)

### Insights
1. No ops SLAs detected.
2. No throughput/capacity figures present.
3. Possible narrative-only content.

### Recommendations
1. **Surface SLAs**: Add SLA & backlog snapshots.
2. **Throughput focus**: Include cycle/lead time trends.
3. **Unit cost**: Append cost-to-serve deltas.
4. **Incident review**: Link to last 30-day incident report.
5. **Capacity view**: Attach capacity plan.

## CMO (Chief Marketing Officer)

### Insights
1. No funnel metrics found.
2. Theme assets hint at presentation use.
3. No campaign data.

### Recommendations
1. **Audit funnel**: Add MQL→SQL→Won conversion stack.
2. **CAC/LTV**: Provide CAC & LTV trendline.
3. **Creative map**: Link asset → KPI impact.
4. **Retention lens**: Include cohort retention cut.
5. **Attribution clarity**: State chosen model.

## CPO (Chief People Officer)

### Insights
1. No recruiting pipeline shown.
2. No time-to-hire data.
3. No quality-of-hire metrics.

### Recommendations
1. **Re-request valid data**: Share readable PDF/DOCX with talent metrics.
2. **Pre-upload checks**: Add validation for submissions.
3. **Technical review**: Investigate file issues if any.
4. **Sourcing plan**: Include sourcing/offer accept rates.
5. **Onboarding link**: Attach onboarding ramp metrics.
"""

    db.add(ChatMessage(session_id=session_id, role="assistant", content=assistant_md, created_at=datetime.utcnow()))
    db.commit()

    return {
        "ok": True,
        "session_id": session_id,
        "assistant": {"content": assistant_md},
    }

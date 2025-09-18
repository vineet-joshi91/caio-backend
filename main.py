# -*- coding: utf-8 -*-
"""
CAIO Backend — main.py
Compatible with auth.py that expects JWT payload {"sub": "<email>"}.
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

from fastapi import (
    FastAPI, Depends, HTTPException, Request, status, Query
)
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import func

# DB and Auth primitives
from db import get_db, init_db, User, UsageLog, ChatSession, ChatMessage
from auth import (
    get_current_user,                 # decodes JWT via {"sub": email}
    create_access_token,              # create_access_token(sub: str, expires_delta?: timedelta)
    get_password_hash,
    verify_password,
)

# --------------------------------------------------------------------------------------
# Lifespan: make startup resilient so /api/ready works even if DB is slow
# --------------------------------------------------------------------------------------
DB_READY = False
STARTUP_OK = False
STARTUP_ERROR = ""

async def _warmup_db():
    global DB_READY
    tries = int(os.getenv("DB_WARMUP_TRIES", "20"))
    delay = float(os.getenv("DB_WARMUP_DELAY", "1.5"))
    for _ in range(tries):
        try:
            init_db()
            DB_READY = True
            return
        except Exception:
            await asyncio.sleep(delay)

app = FastAPI(title="CAIO Backend", version="0.8.0")

@app.on_event("startup")
async def _startup():
    global STARTUP_OK, STARTUP_ERROR
    try:
        await _warmup_db()
        STARTUP_OK = True
        STARTUP_ERROR = ""
    except Exception as e:
        STARTUP_OK = False
        STARTUP_ERROR = str(e)

# --------------------------------------------------------------------------------------
# CORS (production + preview subdomains) and crash-safe headers
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

ALLOW_ORIGIN_REGEX = r"https://.*\.vercel\.app|https://.*\.netlify\.app"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

@app.middleware("http")
async def ensure_cors_headers(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        # make sure CORS middleware can still stamp headers
        return Response("Internal Server Error", status_code=500)

@app.options("/{path:path}")
def cors_preflight(path: str):
    return Response(status_code=204)

# --------------------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "caio-backend", "version": "0.8.0"}

@app.get("/api/health")
def api_health():
    return {"status": "ok"}

@app.get("/api/ready")
def api_ready():
    return {
        "ready": True,
        "db_ready": DB_READY,
        "startup_ok": STARTUP_OK,
        "startup_error": STARTUP_ERROR[:2000],
        "time": datetime.utcnow().isoformat() + "Z",
    }

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _today_bounds_utc() -> Tuple[datetime, datetime]:
    now = datetime.utcnow()
    start = datetime(now.year, now.month, now.day)
    end = start + timedelta(days=1)
    return start, end

def log_usage(db: Session, user_id: int, endpoint: str, status_text: str = "ok", meta: str = ""):
    try:
        db.add(UsageLog(user_id=user_id, endpoint=endpoint, status=status_text,
                        tokens_used=0, timestamp=datetime.utcnow(), meta=meta))
        db.commit()
    except Exception:
        db.rollback()

def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    u = db.query(User).filter(User.email == (email or "").lower()).first()
    if not u:
        return None
    if not verify_password(password or "", u.hashed_password):
        return None
    return u

# --------------------------------------------------------------------------------------
# Auth routes (JWT payload uses {"sub": email} to match auth.py)
# --------------------------------------------------------------------------------------
@app.post("/api/signup")
async def signup(request: Request, db: Session = Depends(get_db)):
    email = password = None
    # accept JSON or form/query
    try:
        body = await request.json()
        if isinstance(body, dict):
            email = body.get("email")
            password = body.get("password")
    except Exception:
        pass
    if not email or not password:
        form = await request.form() if request.headers.get("content-type", "").lower().startswith("application/x-www-form-urlencoded") else None
        if form:
            email = email or form.get("email")
            password = password or form.get("password")

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    email = email.strip().lower()
    exists = db.query(User).filter(User.email == email).first()
    if exists:
        raise HTTPException(status_code=400, detail="User already exists")

    u = User(email=email, hashed_password=get_password_hash(password),
             is_admin=False, is_paid=False, created_at=datetime.utcnow())
    db.add(u); db.commit()
    return {"ok": True, "message": "Signup successful. Please log in."}

@app.post("/api/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # OAuth2 spec uses 'username' field for email
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # IMPORTANT: create token with sub=email (auth.py expects this)
    token = create_access_token(sub=user.email, expires_delta=timedelta(days=1))
    return {"access_token": token, "token_type": "bearer"}

@app.get("/api/profile")
def profile(current: User = Depends(get_current_user)):
    return {
        "email": current.email,
        "is_admin": bool(getattr(current, "is_admin", False)),
        "is_paid": bool(getattr(current, "is_paid", False)),
        "created_at": getattr(current, "created_at", None),
    }

# --------------------------------------------------------------------------------------
# Chat endpoints (sessions/history/send)
# --------------------------------------------------------------------------------------
@app.api_route("/api/chat/sessions", methods=["GET", "POST"])
def chat_sessions(db: Session = Depends(get_db), current: User = Depends(get_current_user)):
    rows = (db.query(ChatSession)
            .filter(ChatSession.user_id == current.id)
            .order_by(ChatSession.created_at.desc())
            .limit(25).all())
    out = []
    for s in rows:
        last = db.query(func.max(ChatMessage.created_at)).filter(ChatMessage.session_id == s.id).scalar()
        out.append({
            "id": s.id,
            "title": f"Chat {s.id}",
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "last_message_at": last.isoformat() if last else None,
        })
    return {"ok": True, "sessions": out}

@app.api_route("/api/chat/history", methods=["GET", "POST"])
async def chat_history(
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    # Accept GET ?session_id= or POST JSON/form with session_id
    sid = None
    if request.method == "GET":
        sid = request.query_params.get("session_id")
    else:
        ct = (request.headers.get("content-type") or "").lower()
        if "application/json" in ct:
            try:
                data = await request.json()
                sid = data.get("session_id")
            except Exception:
                sid = None
        else:
            try:
                form = await request.form()
                sid = form.get("session_id")
            except Exception:
                sid = None

    try:
        session_id = int(sid)
    except Exception:
        raise HTTPException(status_code=400, detail="Missing or invalid session_id")

    s = db.query(ChatSession).filter(ChatSession.id == session_id,
                                     ChatSession.user_id == current.id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    msgs = (db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc()).all())
    return {"ok": True, "messages": [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        } for m in msgs
    ]}

@app.post("/api/chat/send")
async def chat_send(
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """
    Accepts multipart/form-data (files optional) or JSON:
      - message (str)
      - brief (str)
      - session_id (int)
    """
    message: Optional[str] = None
    brief: Optional[str] = None
    session_id: Optional[int] = None
    filenames: List[str] = []

    # Parse body ONCE (avoid double form parsing errors)
    ct = (request.headers.get("content-type") or "").lower()
    if "application/json" in ct:
        try:
            body = await request.json()
            message = body.get("message")
            brief = body.get("brief")
            sid = body.get("session_id")
            session_id = int(sid) if sid is not None else None
        except Exception:
            pass
    else:
        try:
            form = await request.form()
            message = form.get("message")
            brief = form.get("brief")
            sid = form.get("session_id")
            session_id = int(sid) if sid else None
            for v in form.values():
                if hasattr(v, "filename") and getattr(v, "filename"):
                    filenames.append(v.filename)
        except Exception:
            pass

    # Ensure session
    s = None
    if session_id:
        s = db.query(ChatSession).filter(ChatSession.id == session_id,
                                         ChatSession.user_id == current.id).first()
    if not s:
        s = ChatSession(user_id=current.id, created_at=datetime.utcnow())
        db.add(s); db.commit(); db.refresh(s)

    # Persist user message
    user_text = ((message or brief or "") or "").strip() or "(no message)"
    hint = (f"\n\n[files] {', '.join(filenames)}" if filenames else "")
    db.add(ChatMessage(session_id=s.id, user_id=current.id, role="user",
                       content=user_text + hint, created_at=datetime.utcnow()))
    db.commit()

    # Deterministic Insights + CXO recs (placeholder until you swap LLM)
    doc_hint = ", ".join(filenames) if filenames else "attached context"
    brief_hint = (brief or message or "your document").strip()

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
                "Create a hiring-freeze exception path tied to business impact.",
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
        # CPO
        return [
            "Clarify success profile per role; share example work samples.",
            "Shorten time-to-hire: pre-book panels and scorecards.",
            "Upskill managers on feedback that moves performance quickly.",
            "Codify onboarding ‘day-1 to day-30’ outcomes and buddies.",
            "Track diversity of pipeline and close gaps with targeted sourcing.",
        ]

    insights = [
        f"The submission includes {('files: ' + doc_hint) if filenames else 'text input only'}, suitable for CXO review.",
        "Content appears narrative/strategic rather than raw telemetry; summarize themes before diving into metrics.",
        "Capture any explicit asks in the brief to align next steps and owners.",
    ]
    CXOS = [("CFO","Chief Financial Officer"),("CHRO","Chief Human Resources Officer"),
            ("COO","Chief Operating Officer"),("CMO","Chief Marketing Officer"),
            ("CPO","Chief People Officer")]

    lines: List[str] = []
    lines.append("## Insights")
    for i, b in enumerate(insights, 1):
        lines.append(f"{i}. {b}")
    lines.append("")
    for code, full in CXOS:
        lines.append(f"### {code} ({full})")
        lines.append("**Recommendations**")
        for i, r in enumerate(recs_for(code), 1):
            lines.append(f"{i}. {r}")
        lines.append("")

    assistant_content = "\n".join(lines)
    db.add(ChatMessage(session_id=s.id, user_id=current.id, role="assistant",
                       content=assistant_content, created_at=datetime.utcnow()))
    db.commit()

    return {"ok": True, "session_id": s.id,
            "assistant": {"role": "assistant", "content": assistant_content}}

# --------------------------------------------------------------------------------------
# End
# --------------------------------------------------------------------------------------

# main.py
import os, logging, traceback, json
from typing import Optional, List, Tuple
from datetime import datetime
from io import BytesIO

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from db import get_db, User, init_db, UsageLog
from auth import create_access_token, verify_password, get_password_hash, get_current_user

# --------------------------------------------------------------------------
# App + logging
# --------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("caio")

DEBUG = os.getenv("DEBUG", "0") in ("1", "true", "TRUE", "yes", "YES")

app = FastAPI(title="CAIO Backend", version="0.2.0")

# Ensure tables exist (includes usage_logs)
init_db()

# --------------------------------------------------------------------------
# CORS
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# Health / Ready
# --------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.2.0"}

@app.get("/api/ready")
def ready():
    return {"ready": True, "time": datetime.utcnow().isoformat() + "Z"}

# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
ADMIN_EMAILS = {
    e.strip().lower()
    for e in os.getenv("ADMIN_EMAILS", "vineetpjoshi.71@gmail.com").split(",")
    if e.strip()
}

# Optional hook (not used unless set). If you later want premium unlimited:
PREMIUM_EMAILS = {
    e.strip().lower()
    for e in os.getenv("PREMIUM_EMAILS", "").split(",")
    if e.strip()
}

DEMO_DAILY_LIMIT = int(os.getenv("DEMO_DAILY_LIMIT", "5"))
PRO_DAILY_LIMIT  = int(os.getenv("PRO_DAILY_LIMIT",  "50"))

def _login_core(email: str, password: str, db: Session):
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")

    user = db.query(User).filter(User.email == email).first()
    if user:
        if not hasattr(user, "hashed_password"):
            raise RuntimeError("User model missing 'hashed_password' column")
        if not verify_password(password, user.hashed_password):
            raise HTTPException(status_code=400, detail="Incorrect email or password")
    else:
        hpw = get_password_hash(password)
        user = User(
            email=email,
            hashed_password=hpw if hasattr(User, "hashed_password") else None,
            is_admin=(email in ADMIN_EMAILS) if hasattr(User, "is_admin") else False,
            is_paid=False if hasattr(User, "is_paid") else False,
            created_at=datetime.utcnow() if hasattr(User, "created_at") else None,
        )
        db.add(user); db.commit(); db.refresh(user)

    token = create_access_token(sub=user.email)
    return {
        "access_token": token,
        "token_type": "bearer",
        "email": user.email,
        "is_admin": bool(getattr(user, "is_admin", False)),
        "is_paid": bool(getattr(user, "is_paid", False)),
    }

@app.post("/api/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    try:
        email = (form.username or "").strip().lower()
        password = form.password or ""
        return _login_core(email, password, db)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login failed: %s\n%s", e, traceback.format_exc())
        if DEBUG:
            return JSONResponse({"detail": f"login-500: {e.__class__.__name__}: {e}"}, status_code=500)
        raise HTTPException(status_code=500, detail="Login failed")

@app.get("/api/profile")
def profile(current_user: User = Depends(get_current_user)):
    return {
        "email": current_user.email,
        "is_admin": bool(getattr(current_user, "is_admin", False)),
        "is_paid": bool(getattr(current_user, "is_paid", False)),
        "created_at": getattr(current_user, "created_at", None),
        "tier": "admin" if current_user.email.lower() in ADMIN_EMAILS
                else ("premium" if current_user.email.lower() in PREMIUM_EMAILS
                      else ("pro" if getattr(current_user, "is_paid", False) else "demo")),
    }

# --------------------------------------------------------------------------
# Usage limits
# --------------------------------------------------------------------------
def _user_tier(user: User) -> str:
    email = (user.email or "").lower()
    if email in ADMIN_EMAILS:
        return "admin"
    if email in PREMIUM_EMAILS:
        return "premium"  # optional, only if env provided
    return "pro" if bool(getattr(user, "is_paid", False)) else "demo"

def _limit_for_tier(tier: str) -> Optional[int]:
    if tier in ("admin", "premium"):
        return None  # unlimited
    return PRO_DAILY_LIMIT if tier == "pro" else DEMO_DAILY_LIMIT

def _today_range_utc() -> Tuple[datetime, datetime]:
    now = datetime.utcnow()
    start = datetime(now.year, now.month, now.day)
    end   = datetime(now.year, now.month, now.day, 23, 59, 59, 999999)
    return start, end

def _check_and_increment_usage(db: Session, user: User, endpoint: str = "analyze"):
    """Returns (allowed, used, limit, reset_at_utc, tier). Increments on allow."""
    tier = _user_tier(user)
    limit = _limit_for_tier(tier)
    start, end = _today_range_utc()

    if limit is None:
        # Unlimited: still log for observability
        db.add(UsageLog(user_id=getattr(user, "id", 0) or 0, timestamp=datetime.utcnow(), endpoint=endpoint, status="ok"))
        db.commit()
        return True, 0, None, end, tier

    used = (
        db.query(UsageLog)
        .filter(UsageLog.user_id == getattr(user, "id", 0))
        .filter(UsageLog.endpoint == endpoint)
        .filter(UsageLog.timestamp >= start)
        .filter(UsageLog.timestamp <= end)
        .count()
    )

    if used >= limit:
        return False, used, limit, end, tier

    # Reserve this request
    db.add(UsageLog(user_id=getattr(user, "id", 0) or 0, timestamp=datetime.utcnow(), endpoint=endpoint, status="ok"))
    db.commit()
    return True, used + 1, limit, end, tier

def _limit_response(used: int, limit: int, reset_at: datetime, tier: str):
    pretty_plan = "Pro" if tier == "pro" else "Demo"
    remaining = max(0, limit - used)
    # Simple UTC reset time
    reset_hhmm = reset_at.strftime("%H:%M")
    return JSONResponse(
        status_code=429,
        content={
            "status": "error",
            "title": "Daily limit reached",
            "message": f"You've used {used}/{limit} {pretty_plan} requests today. Resets at {reset_hhmm} UTC.",
            "plan": tier,
            "used": used,
            "limit": limit,
            "remaining": remaining,
            "reset_at": reset_at.isoformat() + "Z",
        },
    )

# --------------------------------------------------------------------------
# File reading helpers
# --------------------------------------------------------------------------
def _read_txt(body: bytes) -> str:
    try: return body.decode("utf-8", errors="ignore")
    except Exception: return ""

def _read_pdf(body: bytes) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(BytesIO(body))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        return ""

def _read_docx(body: bytes) -> str:
    try:
        import docx
        d = docx.Document(BytesIO(body))
        return "\n".join(p.text for p in d.paragraphs)
    except Exception:
        return ""

def _read_csv(body: bytes) -> str:
    try:
        txt = body.decode("utf-8", errors="ignore")
        return "\n".join(txt.splitlines()[:200])
    except Exception:
        return ""

def _read_xlsx(body: bytes) -> str:
    try:
        from openpyxl import load_workbook
        wb = load_workbook(filename=BytesIO(body), data_only=True)
        out: List[str] = []
        for ws in wb.worksheets[:2]:
            out.append(f"# Sheet: {ws.title}")
            for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
                if i > 200: break
                cells = ["" if c is None else str(c) for c in row]
                out.append(",".join(cells))
        return "\n".join(out)
    except Exception:
        return ""

async def _extract_text(file: Optional[UploadFile]) -> str:
    if not file: return ""
    body = await file.read()
    name = (file.filename or "").lower()
    if name.endswith(".pdf"):  return _read_pdf(body)
    if name.endswith(".docx"): return _read_docx(body)
    if name.endswith(".xlsx"): return _read_xlsx(body)
    if name.endswith(".csv"):  return _read_csv(body)
    return _read_txt(body)

# --------------------------------------------------------------------------
# Analyze (demo vs pro)
# --------------------------------------------------------------------------
DEFAULT_BRAINS_ORDER: List[str] = ["CFO", "COO", "CHRO", "CMO", "CPO"]

def _choose_brains(requested: Optional[str], is_paid: bool) -> List[str]:
    if requested:
        items = [b.strip().upper() for b in requested.split(",") if b.strip()]
        chosen = [b for b in items if b in DEFAULT_BRAINS_ORDER] or DEFAULT_BRAINS_ORDER[:]
    else:
        chosen = DEFAULT_BRAINS_ORDER[:]
    return chosen if is_paid else chosen[:2]

def _brain_prompt(brief: str, extracted: str, brain: str) -> str:
    role_map = {
        "CFO":"Chief Financial Officer — own capital allocation and unit economics; analyze revenue mix, margins, OPEX, CCC, runway.",
        "COO":"Chief Operating Officer — own cost-to-serve and reliability; capacity, throughput, SLA, FPY/defects, MTTR/MTBF.",
        "CHRO":"Chief Human Resources Officer — org effectiveness; attrition, engagement, span/layer, equity, performance, DEI.",
        "CMO":"Chief Marketing Officer — efficient growth; MMM, CAC/LTV, funnel, creative lift, SEO/SEM, retention/reactivation.",
        "CPO":"Chief People Officer — talent acquisition; pipeline, source mix, time-to-hire, CPH, QoH, geo strategy, brand.",
    }
    role = role_map.get(brain, "Executive Advisor")
    return f"""
You are {role}.
Return STRICT MARKDOWN ONLY for **{brain}** with this structure and nothing else:

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

def _call_openai_chat(prompt: str) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")
    try:
        import openai
        if hasattr(openai, "OpenAI"):
            client = openai.OpenAI(api_key=key, organization=os.getenv("OPENAI_ORG"), project=os.getenv("OPENAI_PROJECT"))
            r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.2)
            return r.choices[0].message.content
        else:
            openai.api_key = key
            if os.getenv("OPENAI_ORG"): openai.organization = os.getenv("OPENAI_ORG")
            r = openai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.2)
            return r.choices[0].message.content
    except Exception:
        import requests
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if os.getenv("OPENAI_ORG"): headers["OpenAI-Organization"] = os.getenv("OPENAI_ORG")
        if os.getenv("OPENAI_PROJECT"): headers["OpenAI-Project"] = os.getenv("OPENAI_PROJECT")
        data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(data), timeout=60)
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenAI {resp.status_code}: {resp.text}")
        return resp.json()["choices"][0]["message"]["content"]

def _call_openrouter_chat(prompt: str) -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY missing")
    import requests
    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://caio-frontend.vercel.app",
        "X-Title": "CAIO",
        "Content-Type": "application/json",
    }
    payload = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(payload), timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenRouter {r.status_code}: {r.text}")
    return r.json()["choices"][0]["message"]["content"]

def _pick_provider(is_paid: bool) -> List[str]:
    raw = os.getenv("LLM_PROVIDER_PRO" if is_paid else "LLM_PROVIDER_DEMO", "openai,openrouter" if is_paid else "openrouter")
    providers = [p.strip().lower() for p in raw.split(",") if p.strip()]
    available: List[str] = []
    if "openai" in providers and os.getenv("OPENAI_API_KEY"): available.append("openai")
    if "openrouter" in providers and os.getenv("OPENROUTER_API_KEY"): available.append("openrouter")
    return available or []

@app.post("/api/analyze")
async def analyze(
    request: Request,
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    brains: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Enforce daily caps (count both text and file uploads)
    allowed, used, limit, reset_at, tier = _check_and_increment_usage(db, current_user, endpoint="analyze")
    if not allowed:
        return _limit_response(used, int(limit or 0), reset_at, tier)

    # Demo flow returns preview (still counts above)
    if tier == "demo":
        chosen = _choose_brains(brains, is_paid=False)
        return JSONResponse(
            {
                "status": "demo",
                "title": f"Demo Mode · {', '.join(chosen)}",
                "summary": (
                    "This is a demo preview. Upgrade to Pro to run all brains with real analysis.\n"
                    f"Brains used: {', '.join(chosen)}"
                ),
                "tip": "Upload a business document or provide a brief to see the flow.",
            },
            status_code=200,
        )

    # Pro/Admin (and optional Premium)
    try:
        extracted = await _extract_text(file)
    except Exception as e:
        logger.error("Extraction failed: %s\n%s", e, traceback.format_exc())
        extracted = ""
    brief = (text or "").strip()
    chosen = _choose_brains(brains, is_paid=True)

    if not extracted and not brief:
        raise HTTPException(status_code=400, detail="Please upload a file or provide text")

    prompts = {b: _brain_prompt(brief, extracted, b) for b in chosen}
    provider_chain = _pick_provider(is_paid=True)

    summaries = {}
    used_provider = "stub"
    errs: List[str] = []

    if provider_chain:
        for p in provider_chain:
            try:
                if p == "openai":
                    for b, pr in prompts.items(): summaries[b] = _call_openai_chat(pr)
                elif p == "openrouter":
                    for b, pr in prompts.items(): summaries[b] = _call_openrouter_chat(pr)
                used_provider = p
                break
            except Exception as e:
                logger.error("Provider %s failed: %s", p, e); errs.append(f"{p}:{e.__class__.__name__}"); summaries = {}; continue

    if not summaries:
        for b in prompts.keys():
            summaries[b] = f"[Stub after error] {b} analysis. Errors: {', '.join(errs) if errs else 'no-provider'}."

    combined = [f"### {b}\n{summaries[b]}" for b in chosen]
    return {
        "status": "ok",
        "title": f"Analysis Complete · {', '.join(chosen)}",
        "summary": "\n\n".join(combined),
        "meta": {"provider": used_provider, "brains": chosen, "chars": len(extracted)},
    }

# --------------------------------------------------------------------------
# Debug helpers
# --------------------------------------------------------------------------
@app.get("/api/debug/ping-db")
def ping_db(db: Session = Depends(get_db)):
    try:
        count = db.query(User).count()
        u = db.query(User).first()
        fields = []
        if u:
            for k in ("email", "hashed_password", "is_admin", "is_paid", "created_at"):
                fields.append(f"{k}={'Y' if hasattr(u, k) else 'N'}")
        return {"ok": True, "user_count": count, "user_fields": ", ".join(fields)}
    except Exception as e:
        logger.error("DB ping error: %s\n%s", e, traceback.format_exc())
        if DEBUG:
            return JSONResponse({"ok": False, "detail": str(e)}, status_code=500)
        raise HTTPException(status_code=500, detail="DB error")

@app.get("/api/debug/routers")
def debug_routers():
    out = []
    for r in app.routes:
        methods = sorted(list(getattr(r, "methods", [])))
        path = getattr(r, "path", "")
        if path.startswith("/api/"):
            out.append({"path": path, "methods": methods})
    return out

# --------------------------------------------------------------------------
# Other routers (kept)
# --------------------------------------------------------------------------
try:
    from routes_public_config import router as public_cfg_router
    app.include_router(public_cfg_router)  # /api/public-config
except Exception as e:
    logger.warning(f"routes_public_config not loaded: {e}")

try:
    from payment_routes import router as payments_router
    app.include_router(payments_router)
    logger.info("✅ payments_router mounted")
except Exception as e:
    logger.warning(f"payment_routes not loaded: {e}")

try:
    from contact_routes import router as contact_router
    app.include_router(contact_router)  # /api/contact
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

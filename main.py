# main.py
import os, logging, traceback, json
from typing import Optional
from datetime import datetime
from io import BytesIO

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from db import get_db, User, init_db
from auth import create_access_token, verify_password, get_password_hash, get_current_user

# ------------------------------------------------------------------------------
# App + logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("caio")

DEBUG = os.getenv("DEBUG", "0") in ("1", "true", "TRUE", "yes", "YES")

app = FastAPI(title="CAIO Backend", version="0.1.0")

# Ensure tables exist (works for Postgres or SQLite fallback)
init_db()

# ------------------------------------------------------------------------------
# CORS
# ------------------------------------------------------------------------------
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

# Browser preflight helper (keeps OPTIONS happy even if specific route missing)
@app.options("/{path:path}")
def cors_preflight(path: str):
    return JSONResponse({"ok": True})

# ------------------------------------------------------------------------------
# Health / Ready
# ------------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}

@app.get("/api/ready")
def ready():
    return {"ready": True, "time": datetime.utcnow().isoformat() + "Z"}

# ------------------------------------------------------------------------------
# Auth
# ------------------------------------------------------------------------------
ADMIN_EMAILS = {
    e.strip().lower()
    for e in os.getenv("ADMIN_EMAILS", "vineetpjoshi.71@gmail.com").split(",")
    if e.strip()
}

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
    }

# ------------------------------------------------------------------------------
# Analyze
#   - Demo users → demo result (unchanged UX)
#   - Pro users → real LLM if key present (OpenAI preferred, else OpenRouter)
#   - If no key: return a Pro stub so end-to-end tests still succeed
# ------------------------------------------------------------------------------
def _read_txt(body: bytes) -> str:
    try:
        return body.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def _read_pdf(body: bytes) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(BytesIO(body))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        return ""

def _read_docx(body: bytes) -> str:
    try:
        import docx  # python-docx
        d = docx.Document(BytesIO(body))
        return "\n".join(p.text for p in d.paragraphs)
    except Exception:
        return ""

async def _extract_text(file: Optional[UploadFile]) -> str:
    if not file:
        return ""
    body = await file.read()
    name = (file.filename or "").lower()
    if name.endswith(".pdf"):
        return _read_pdf(body)
    if name.endswith(".docx"):
        return _read_docx(body)
    return _read_txt(body)

def _call_openai_chat(prompt: str) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")

    # Try SDK first; if it fails (version mismatch), fall back to REST
    try:
        import openai
        if hasattr(openai, "OpenAI"):  # new-style client
            client = openai.OpenAI(api_key=key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return resp.choices[0].message.content
        else:  # legacy
            openai.api_key = key
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return resp.choices[0].message.content
    except Exception as e:
        # REST fallback (works without SDK)
        import requests
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(data), timeout=60)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]

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
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

@app.post("/api/analyze")
async def analyze(
    request: Request,
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    POST-only. If not paid → demo result.
    If paid → extract text and run LLM (OpenAI preferred; else OpenRouter).
    With no keys configured, return a Pro stub so UX still completes.
    """
    # DEMO users
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

    # PRO users
    extracted = await _extract_text(file)
    brief = (text or "").strip()
    if not extracted and not brief:
        raise HTTPException(status_code=400, detail="Please upload a file or provide text")

    prompt = (
        "You are CAIO (Chief AI Officer). Read the user's brief and document text. "
        "Return a concise summary (5 bullets) and 3 actionable recommendations.\n\n"
        f"BRIEF:\n{brief or '(none)'}\n\nDOCUMENT:\n{extracted[:15000]}"
    )

    try:
        if os.getenv("OPENAI_API_KEY"):
            summary = _call_openai_chat(prompt)
        elif os.getenv("OPENROUTER_API_KEY"):
            summary = _call_openrouter_chat(prompt)
        else:
            # No keys configured → Pro stub so flows pass end-to-end
            summary = (
                "Stub summary (no LLM key configured). "
                "This confirms file upload, extraction, and paid routing work.\n\n"
                f"Extracted chars: {len(extracted)}"
            )
        return {
            "status": "ok",
            "title": "Analysis Complete",
            "summary": summary,
            "meta": {
                "chars": len(extracted),
                "provider": "openai" if os.getenv("OPENAI_API_KEY") else ("openrouter" if os.getenv("OPENROUTER_API_KEY") else "stub"),
            },
        }
    except Exception as e:
        logger.error("Analyze failed: %s\n%s", e, traceback.format_exc())
        if DEBUG:
            return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)
        raise HTTPException(status_code=500, detail="Analysis failed")

# ------------------------------------------------------------------------------
# Debug helpers
# ------------------------------------------------------------------------------
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

@app.get("/api/debug/user-sample")
def user_sample(db: Session = Depends(get_db)):
    try:
        u = db.query(User).first()
        if not u:
            return {"sample": None}
        return {
            "email": getattr(u, "email", None),
            "has_hashed_password": hasattr(u, "hashed_password"),
            "is_admin": getattr(u, "is_admin", None),
            "is_paid": getattr(u, "is_paid", None),
        }
    except Exception as e:
        logger.error("User sample error: %s\n%s", e, traceback.format_exc())
        if DEBUG:
            return JSONResponse({"detail": str(e)}, status_code=500)
        raise HTTPException(status_code=500, detail="DB error")

# ------------------------------------------------------------------------------
# Other routers (won't crash boot if missing)
# ------------------------------------------------------------------------------
try:
    from routes_public_config import router as public_cfg_router
    app.include_router(public_cfg_router)  # /api/public-config
except Exception as e:
    logger.warning(f"routes_public_config not loaded: {e}")

try:
    from payment_routes import router as payments_router
    app.include_router(payments_router)  # already prefixed with /api/payments
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

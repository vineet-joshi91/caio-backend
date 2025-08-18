import os
import logging
from typing import Optional

from fastapi import (
    FastAPI, Depends, HTTPException, UploadFile, File, Form, Body, Request
)
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm

# -----------------------------------------------------------------------------
# App & logging
# -----------------------------------------------------------------------------
app = FastAPI(title="CAIO Backend", version="1.0")
logger = logging.getLogger("uvicorn.error")

# -----------------------------------------------------------------------------
# CORS (env-driven allowlist + explicit global OPTIONS handler)
# -----------------------------------------------------------------------------
_raw = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _raw.split(",") if o.strip()]
if not ALLOWED_ORIGINS:
    # Safe defaults for CAIO; add more origins in Render via ALLOWED_ORIGINS
    ALLOWED_ORIGINS = ["https://caio-frontend.vercel.app", "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,           # ok with Authorization header
    allow_methods=["*"],              # includes OPTIONS
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

@app.options("/{path:path}")
def cors_preflight(path: str):
    # Some platforms are strict about explicit OPTIONS handlers
    return Response(status_code=204)

# -----------------------------------------------------------------------------
# DB wiring
# -----------------------------------------------------------------------------
try:
    from db import Base, engine, get_db, User  # type: ignore

    @app.on_event("startup")
    def _init_db():
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("DB metadata created")
        except Exception as e:
            logger.warning(f"DB metadata creation skipped: {e}")
except Exception as e:
    logger.warning(f"DB imports unavailable: {e}")

    def get_db():  # type: ignore
        raise HTTPException(status_code=500, detail="DB not configured")

    class User:  # type: ignore
        pass

# -----------------------------------------------------------------------------
# Auth wiring
# -----------------------------------------------------------------------------
try:
    # prefer get_current_user, fall back to common alias
    try:
        from auth import get_current_user  # type: ignore
    except Exception:
        from auth import get_current_active_user as get_current_user  # type: ignore

    try:
        from auth import authenticate_user  # type: ignore
    except Exception:
        authenticate_user = None  # type: ignore

    try:
        from auth import create_access_token  # type: ignore
    except Exception:
        create_access_token = None  # type: ignore

    try:
        from auth import get_password_hash  # type: ignore
    except Exception:
        get_password_hash = None  # type: ignore

except Exception as e:
    logger.warning(f"Auth imports unavailable: {e}")

    def get_current_user():  # type: ignore
        raise HTTPException(status_code=401, detail="Auth not configured")

    authenticate_user = None  # type: ignore
    create_access_token = None  # type: ignore
    get_password_hash = None  # type: ignore

# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}

# -----------------------------------------------------------------------------
# Signup
# -----------------------------------------------------------------------------
class SignupPayload(BaseModel):
    email: EmailStr
    password: str

@app.post("/api/signup")
def signup(
    email: Optional[EmailStr] = None,            # allow query (?email=&password=)
    password: Optional[str] = None,
    payload: Optional[SignupPayload] = Body(None),
    db: Session = Depends(get_db),
):
    """
    Accepts JSON body {email,password} OR query params (?email=&password=).
    Returns generic success (no user existence leak).
    """
    try:
        if payload:
            email_v, password_v = payload.email, payload.password
        else:
            email_v, password_v = email, password
        if not email_v or not password_v:
            raise HTTPException(status_code=400, detail="email and password required")

        user = db.query(User).filter(User.email == str(email_v)).first()  # type: ignore
        if not user:
            hashed = (
                get_password_hash(password_v)  # type: ignore
                if callable(get_password_hash) else password_v
            )
            # Support either "hashed_password" or "password_hash" field names.
            kwargs = {
                "email": str(email_v),
                "is_admin": False,
                "is_paid": False,
            }
            if hasattr(User, "hashed_password"):
                kwargs["hashed_password"] = hashed  # type: ignore
            elif hasattr(User, "password_hash"):
                kwargs["password_hash"] = hashed  # type: ignore
            else:
                kwargs["hashed_password"] = hashed  # type: ignore
            user = User(**kwargs)  # type: ignore
            db.add(user)
            db.commit()

        return {"message": "Signup successful. Please log in."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail="Signup failed")

# -----------------------------------------------------------------------------
# Login
# -----------------------------------------------------------------------------
@app.post("/api/login")
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    # guard missing funcs
    if authenticate_user is None or create_access_token is None:
        raise HTTPException(status_code=500, detail="Auth functions not available")

    # pass DB session to authenticate_user  ✅
    user = authenticate_user(form.username, form.password, db)  # type: ignore
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    token = create_access_token(sub=getattr(user, "email", form.username))  # type: ignore
    return {"access_token": token, "token_type": "bearer"}

# -----------------------------------------------------------------------------
# Profile
# -----------------------------------------------------------------------------
@app.get("/api/profile")
def profile(current_user=Depends(get_current_user)):
    try:
        return {
            "email": getattr(current_user, "email", None),
            "is_admin": bool(getattr(current_user, "is_admin", False)),
            "is_paid": bool(getattr(current_user, "is_paid", False)),
            "created_at": str(getattr(current_user, "created_at", "")),
        }
    except Exception as e:
        logger.error(f"profile error: {e}")
        raise HTTPException(status_code=500, detail="profile error")

# -----------------------------------------------------------------------------
# Analyze (text or file) with Demo vs Pro behavior
# -----------------------------------------------------------------------------
def _demo_response(text: Optional[str], filename: Optional[str]):
    preview = (text or "").strip()[:220] if text else (filename or "document")
    return {
        "mode": "demo",
        "summary": f"Sample insights for: {preview}",
        "notes": "This is a demo preview. Upgrade to Pro to run full analysis on your text or uploaded files.",
        "cxo": {
            "CFO":  "Watch unit economics and cash conversion in Q3.",
            "CMO":  "Optimize paid spend; lift organic via content-led SEO.",
            "COO":  "Tighten fulfillment SLAs; reduce defect rate.",
            "CHRO": "Upskill GTM; tie incentives to retention.",
        },
    }

def _no_credits_response():
    return {
        "mode": "pro",
        "error": "NO_CREDITS",
        "message": "Not enough credits (or API unavailable). Please add credits or try again later.",
    }

def _try_project_engines(text: Optional[str], file_bytes: Optional[bytes], filename: Optional[str]):
    """
    Prefer your own engines/brains if present. If they raise, we bubble up and
    decide what to show (demo/no_credits) above.
    """
    # 1) brains
    try:
        from brains import generate_cxo_insights  # type: ignore
        return generate_cxo_insights(text=text, file_bytes=file_bytes, filename=filename)  # type: ignore
    except Exception:
        pass

    # 2) engines
    try:
        from engines import run_analysis  # type: ignore
        return run_analysis(text=text, file_bytes=file_bytes, filename=filename)  # type: ignore
    except Exception:
        pass

    # 3) optional OpenAI fallback (only if key present)
    try:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("no_api_key")
        # If you want a real OpenAI call, wire it here. We deliberately skip to keep
        # deployments deterministic without external spend.
        raise RuntimeError("treat_as_no_credits")
    except Exception:
        # Let caller decide to surface NO_CREDITS
        raise

def _resolve_user_tier(current_user) -> str:
    return "pro" if bool(getattr(current_user, "is_paid", False)) else "demo"

@app.post("/api/analyze")
async def analyze(
    request: Request,
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    current_user=Depends(get_current_user),
):
    """
    - DEMO (is_paid == False): always return a canned, helpful sample.
    - PRO  (is_paid == True): try engines/brains. If API is missing/exhausted,
      return a structured NO_CREDITS response for the UI to display.
    """
    content: Optional[bytes] = None
    fname: Optional[str] = None
    if file is not None:
        content = await file.read()
        fname = file.filename

    if not content and not (text and text.strip()):
        raise HTTPException(status_code=400, detail="Provide text or upload a file")

    tier = _resolve_user_tier(current_user)

    if tier == "demo":
        return JSONResponse(_demo_response(text, fname))

    try:
        result = _try_project_engines(text.strip() if text else None, content, fname)
        return JSONResponse(result)
    except Exception:
        return JSONResponse(_no_credits_response(), status_code=402)

# -----------------------------------------------------------------------------
# Optional routers
# -----------------------------------------------------------------------------
try:
    from admin_routes import router as admin_router  # type: ignore
    app.include_router(admin_router, prefix="/api/admin", tags=["admin"])
    logger.info("Admin router loaded at /api/admin")
except Exception as e:
    logger.warning(f"Admin router NOT loaded: {e}")

try:
    from payments_routes import router as payments_router  # type: ignore
    app.include_router(payments_router, prefix="/api/payments", tags=["payments"])
    logger.info("Payments router loaded at /api/payments")
except Exception as e:
    logger.warning(f"Payments router NOT loaded: {e}")

try:
    from dev_routes import router as dev_router  # type: ignore
    app.include_router(dev_router, prefix="/api/dev", tags=["dev"])
    logger.info("Dev router loaded at /api/dev")
except Exception as e:
    logger.warning(f"Dev router NOT loaded: {e}")

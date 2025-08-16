import os
import logging
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm

# ---- App ----
app = FastAPI(title="CAIO Backend", version="1.0")
logger = logging.getLogger("uvicorn.error")

# ---- CORS (robust, with preflight) ----
raw = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED = [o.strip() for o in raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED if ALLOWED else ["*"],   # set ALLOWED_ORIGINS in Render for prod
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# ---- DB wiring ----
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

    def get_db():
        raise HTTPException(status_code=500, detail="DB not configured")

# ---- Auth wiring ----
try:
    # try common names used across tutorials/boilerplates
    try:
        from auth import get_current_user  # type: ignore
    except Exception:
        from auth import get_current_active_user as get_current_user  # type: ignore

    # optional helpers; ignore if absent
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

# ---- Health ----
@app.get("/api/health")
def health():
    return {"status": "ok"}

# ---- Signup ----
class SignupPayload(BaseModel):
    email: EmailStr
    password: str


@app.post("/api/signup")
def signup(
    email: Optional[EmailStr] = None,
    password: Optional[str] = None,
    payload: Optional[SignupPayload] = Body(None),
    db: Session = Depends(get_db),
):
    """
    Accepts either JSON body {email,password} OR query params (?email=&password=).
    Returns generic success so we don't leak whether the user exists.
    """
    try:
        if payload:
            email_v, password_v = payload.email, payload.password
        else:
            email_v, password_v = email, password
        if not email_v or not password_v:
            raise HTTPException(status_code=400, detail="email and password required")

        # create if not exists
        try:
            user = db.query(User).filter(User.email == str(email_v)).first()
            if not user:
                hashed = (
                    get_password_hash(password_v)  # type: ignore
                    if callable(get_password_hash)
                    else password_v
                )
                # Assume User model has these fields; adjust if your schema differs
                user = User(
                    email=str(email_v),
                    hashed_password=hashed,  # or 'password_hash' per your model
                    is_admin=False,
                    is_paid=False,
                )  # type: ignore
                db.add(user)
                db.commit()
        except Exception as e:
            logger.warning(f"Signup create fallback: {e}")

        return {"message": "Signup successful. Please log in."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail="Signup failed")

# ---- Login ----

from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

@app.post("/api/login")
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    # guard missing funcs
    if authenticate_user is None or create_access_token is None:
        raise HTTPException(status_code=500, detail="Auth functions not available")

    # ✅ pass the DB session into authenticate_user
    user = authenticate_user(form.username, form.password, db)  # <— FIXED
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    token = create_access_token(sub=getattr(user, "email", form.username))
    return {"access_token": token, "token_type": "bearer"}

# ---- Profile ----
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

# ---- Analyze ----
def _safe_generate(text: Optional[str], file_bytes: Optional[bytes], filename: Optional[str]):
    """
    Delegates to your project functions if available; otherwise returns a minimal structured response.
    Replace with your real brains/engines when ready.
    """
    try:
        from brains import generate_cxo_insights  # type: ignore
        return generate_cxo_insights(text=text, file_bytes=file_bytes, filename=filename)  # type: ignore
    except Exception:
        pass
    try:
        from engines import run_analysis  # type: ignore
        return run_analysis(text=text, file_bytes=file_bytes, filename=filename)  # type: ignore
    except Exception:
        pass
    return {
        "summary": (text or "")[:4000],
        "notes": "Fallback analyzer executed. Wire to brains/engines for full output.",
        "cxo": {"CFO": "—", "CMO": "—", "COO": "—", "CHRO": "—"},
    }


@app.post("/api/analyze")
async def analyze(
    request: Request,
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    current_user=Depends(get_current_user),
):
    try:
        content: Optional[bytes] = None
        fname: Optional[str] = None
        if file is not None:
            content = await file.read()
            fname = file.filename
        if not content and not (text and text.strip()):
            raise HTTPException(status_code=400, detail="Provide text or upload a file")
        result = _safe_generate(text.strip() if text else None, content, fname)
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"analyze error: {e}")
        raise HTTPException(status_code=500, detail="Analyze failed")

# ---- Admin router (guarded include) ----
try:
    from admin_routes import router as admin_router  # type: ignore
    app.include_router(admin_router, prefix="/api/admin", tags=["admin"])
    logger.info("Admin router loaded at /api/admin")
except Exception as e:
    logger.warning(f"Admin router NOT loaded: {e}")

# ---- Payments router (guarded include) ----
try:
    from payments_routes import router as payments_router  # type: ignore
    app.include_router(payments_router, prefix="/api/payments", tags=["payments"])
    logger.info("Payments router loaded at /api/payments")
except Exception as e:
    logger.warning(f"Payments router NOT loaded: {e}")

# ---- Dev router (bootstrap) ----
try:
    from dev_routes import router as dev_router  # type: ignore
    app.include_router(dev_router, prefix="/api/dev", tags=["dev"])
    logger.info("Dev router loaded at /api/dev")
except Exception as e:
    logger.warning(f"Dev router NOT loaded: {e}")

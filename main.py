# main.py
import os
from typing import Optional, List
from datetime import datetime
from io import BytesIO
from textwrap import wrap

from fastapi import (
    FastAPI, Depends, HTTPException, UploadFile, File, Form, Request, Body, Query
)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from db import get_db, User, init_db, UsageLog
from auth import create_access_token, verify_password, get_password_hash, get_current_user
from payment_routes import router as payments_router
from routes_public_config import router as public_config_router

# Export libs
from docx import Document
from docx.shared import Pt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
FREE_QUERIES_PER_DAY = int(os.getenv("FREE_QUERIES_PER_DAY", "3"))

# ------------------------------------------------------------------------------
# App
# ------------------------------------------------------------------------------
app = FastAPI(title="CAIO Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(payments_router)
app.include_router(public_config_router)

# Ensure tables
init_db()

# ------------------------------------------------------------------------------
# CORS preflight
# ------------------------------------------------------------------------------
@app.options("/{path:path}")
def cors_preflight(path: str):
    return JSONResponse({"ok": True})

# ------------------------------------------------------------------------------
# Health
# ------------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

@app.get("/ready")
def ready():
    return {"ready": True}

# ------------------------------------------------------------------------------
# Auth: signup/login/profile
# ------------------------------------------------------------------------------
@app.post("/api/signup")
def signup(
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    user = User(
        email=email,
        hashed_password=get_password_hash(password),
        is_admin=False,
        is_paid=False,
        created_at=datetime.utcnow(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(sub=user.email)
    return {
        "access_token": token,
        "token_type": "bearer",
        "email": user.email,
        "is_admin": bool(user.is_admin),
        "is_paid": bool(user.is_paid),
    }

@app.post("/api/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(sub=user.email)
    return {
        "access_token": token,
        "token_type": "bearer",
        "email": user.email,
        "is_admin": bool(user.is_admin),
        "is_paid": bool(user.is_paid),
    }

@app.get("/api/profile")
def profile(current_user: User = Depends(get_current_user)):
    return {
        "email": current_user.email,
        "is_admin": bool(getattr(current_user, "is_admin", False)),
        "is_paid": bool(getattr(current_user, "is_paid", False)),
        "created_at": getattr(current_user, "created_at", None),
    }

# ------------------------------------------------------------------------------
# Usage limiting helpers
# ------------------------------------------------------------------------------
def _today_range_utc():
    now = datetime.utcnow()
    start = datetime(now.year, now.month, now.day)
    end = datetime(now.year, now.month, now.day, 23, 59, 59, 999999)
    return start, end

def _check_and_increment_usage(db: Session, user: User, endpoint: str = "analyze"):
    if bool(getattr(user, "is_paid", False)):
        db.add(UsageLog(user_id=getattr(user, "id", 0) or 0,
                        timestamp=datetime.utcnow(), endpoint=endpoint, status="ok"))
        db.commit()
        return

    start, end = _today_range_utc()
    count = (
        db.query(UsageLog)
        .filter(UsageLog.user_id == getattr(user, "id", 0))
        .filter(UsageLog.endpoint == endpoint)
        .filter(UsageLog.timestamp >= start)
        .filter(UsageLog.timestamp <= end)
        .count()
    )
    if count >= FREE_QUERIES_PER_DAY:
        raise HTTPException(
            status_code=429,
            detail="Daily free limit reached. Upgrade to Pro for unlimited analyses.",
        )

    db.add(UsageLog(user_id=getattr(user, "id", 0) or 0,
                    timestamp=datetime.utcnow(), endpoint=endpoint, status="ok"))
    db.commit()

def _ensure_pro(current_user: User):
    if not bool(getattr(current_user, "is_paid", False)):
        raise HTTPException(status_code=403, detail="Export is available to Pro accounts only.")

# ------------------------------------------------------------------------------
# File readers (lightweight)
# ------------------------------------------------------------------------------
async def _read_upload_as_text(up: UploadFile) -> str:
    if not up:
        return ""
    body = await up.read()
    name = (up.filename or "").lower()

    if name.endswith((".txt", ".md", ".csv", ".tsv")):
        try:
            return body.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    if name.endswith(".docx"):
        try:
            from docx import Document as _Docx
            d = _Docx(BytesIO(body))
            return "\n".join(p.text for p in d.paragraphs)
        except Exception:
            return ""

    if name.endswith(".pdf"):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(BytesIO(body))
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception:
            return ""

    try:
        return body.decode("utf-8", errors="ignore")
    except Exception:
        return ""

# ------------------------------------------------------------------------------
# Analyze (Free limited / Pro unlimited)
# ------------------------------------------------------------------------------
@app.post("/api/analyze")
async def analyze(
    request: Request,
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    brains: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _check_and_increment_usage(db, current_user, endpoint="analyze")

    extracted = ""
    if text and text.strip():
        extracted = text.strip()
    elif file is not None:
        extracted = (await _read_upload_as_text(file)).strip()

    if not extracted:
        raise HTTPException(status_code=400, detail="No input provided.")

    selected = [s.strip().upper() for s in (brains or "CFO,COO,CMO,CHRO").split(",") if s.strip()]
    selected = [b for b in selected if b in {"CFO", "COO", "CMO", "CHRO"}] or ["CFO","COO","CMO","CHRO"]

    sections = []
    for b in selected:
        if b == "CFO":
            sections.append("### CFO\n- Watch cash runway, collection cycles, and discretionary spend.\n- Prioritize high-ROI bets.")
        elif b == "COO":
            sections.append("### COO\n- Remove bottlenecks; standardize SOPs; add checkpoints.\n- Automate repeatable workflows.")
        elif b == "CMO":
            sections.append("### CMO\n- Double down on channels with provable CAC:LTV.\n- Ship weekly experiments; measure real lift.")
        elif b == "CHRO":
            sections.append("### CHRO\n- Shore up engagement; fix feedback loops.\n- Hire for impact roles; mentor for leverage.")

    summary = "\n\n".join(sections) + "\n"
    title = f"Analysis Complete · {', '.join(selected)}"
    return {
        "status": "ok",
        "title": title,
        "summary": summary,
        "provider": "stub",
        "brains": selected,
        "chars": len(extracted),
    }

# ------------------------------------------------------------------------------
# Export helpers: DOCX/PDF
# ------------------------------------------------------------------------------
def _write_docx(title: str, md_text: str) -> BytesIO:
    doc = Document()
    h = doc.add_heading(title, level=1)
    for run in h.runs: run.font.size = Pt(16)

    for line in md_text.splitlines():
        line = line.rstrip()
        if not line.strip():
            doc.add_paragraph(""); continue
        if line.startswith("### "):
            p = doc.add_paragraph()
            r = p.add_run(line[4:].strip()); r.bold = True; r.font.size = Pt(12); continue
        if line.startswith("- "):
            p = doc.add_paragraph(line[2:].strip())
            p.style = doc.styles["List Bullet"]; continue
        doc.add_paragraph(line)

    bio = BytesIO(); doc.save(bio); bio.seek(0); return bio

def _write_pdf(title: str, md_text: str) -> BytesIO:
    bio = BytesIO(); c = canvas.Canvas(bio, pagesize=A4)
    width, height = A4; left, right, top, bottom = 50, width-50, height-50, 50; y = top
    def new_page():
        nonlocal y; c.showPage(); y = top

    c.setFont("Helvetica-Bold", 14)
    for ln in wrap(title, 80):
        if y < bottom: new_page()
        c.drawString(left, y, ln); y -= 18

    c.setFont("Helvetica", 11)
    for raw in md_text.splitlines():
        txt = raw.rstrip()
        if not txt:
            y -= 8; 
            if y < bottom: new_page(); 
            continue
        if txt.startswith("### "):
            c.setFont("Helvetica-Bold", 12)
            for ln in wrap(txt[4:].strip(), 85):
                if y < bottom: new_page()
                c.drawString(left, y, ln); y -= 16
            c.setFont("Helvetica", 11); continue
        if txt.startswith("- "):
            for ln in wrap("• " + txt[2:].strip(), 95):
                if y < bottom: new_page()
                c.drawString(left + 10, y, ln); y -= 14
            continue
        for ln in wrap(txt, 100):
            if y < bottom: new_page()
            c.drawString(left, y, ln); y -= 14

    c.save(); bio.seek(0); return bio

# ------------------------------------------------------------------------------
# Export endpoints (Pro only) — support POST and GET; add /word alias
# ------------------------------------------------------------------------------
def _extract_export_content(
    payload: Optional[dict],
    title_q: Optional[str],
    md_q: Optional[str],
    md_text_q: Optional[str],
):
    title = (payload.get("title") if payload else None) or (title_q or "CAIO Analysis")
    md_text = (
        (payload.get("md_text") if payload else None)
        or (payload.get("markdown") if payload else None)
        or md_text_q or md_q or ""
    )
    return title.strip(), md_text

@app.api_route("/api/export/docx", methods=["POST", "GET"])
def export_docx(
    request: Request,
    payload: Optional[dict] = Body(None),
    title: Optional[str] = Query(None),
    md: Optional[str] = Query(None),
    md_text: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
):
    _ensure_pro(current_user)
    t, content = _extract_export_content(payload, title, md, md_text)
    if not content.strip():
        raise HTTPException(status_code=400, detail="No content to export.")
    bio = _write_docx(t, content)
    headers = {"Content-Disposition": f'attachment; filename="{t.replace(" ", "_").lower()}.docx"'}
    return StreamingResponse(
        bio,
        headers=headers,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

# alias for Word
@app.api_route("/api/export/word", methods=["POST", "GET"])
def export_word(
    request: Request,
    payload: Optional[dict] = Body(None),
    title: Optional[str] = Query(None),
    md: Optional[str] = Query(None),
    md_text: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
):
    return export_docx(request, payload, title, md, md_text, current_user)  # reuse

@app.api_route("/api/export/pdf", methods=["POST", "GET"])
def export_pdf(
    request: Request,
    payload: Optional[dict] = Body(None),
    title: Optional[str] = Query(None),
    md: Optional[str] = Query(None),
    md_text: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
):
    _ensure_pro(current_user)
    t, content = _extract_export_content(payload, title, md, md_text)
    if not content.strip():
        raise HTTPException(status_code=400, detail="No content to export.")
    bio = _write_pdf(t, content)
    headers = {"Content-Disposition": f'attachment; filename="{t.replace(" ", "_").lower()}.pdf"'}
    return StreamingResponse(bio, headers=headers, media_type="application/pdf")

# main.py
import os
import re
from typing import Optional, List, Dict, Tuple
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
DEFAULT_ORIGINS = "https://caio-frontend.vercel.app,https://caioai.netlify.app,http://localhost:3000"
ALLOWED = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",")
    if o.strip()
]
FREE_QUERIES_PER_DAY = int(os.getenv("FREE_QUERIES_PER_DAY", "3"))

PREFERRED_ORDER = ["CFO", "CHRO", "COO", "CMO", "CPO"]

# ------------------------------------------------------------------------------
# App
# ------------------------------------------------------------------------------
app = FastAPI(title="CAIO Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(payments_router)
app.include_router(public_config_router)
init_db()

# ------------------------------------------------------------------------------
# Health
# ------------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/ready")
def ready():
    return {"ready": True}

@app.options("/{path:path}")
def cors_preflight(path: str):
    return JSONResponse({"ok": True})

# ------------------------------------------------------------------------------
# Auth
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
    db.add(user); db.commit(); db.refresh(user)

    token = create_access_token(sub=user.email)
    return {
        "access_token": token,
        "token_type": "bearer",
        "email": user.email,
        "is_admin": bool(user.is_admin),
        "is_paid": bool(user.is_paid),
    }

@app.post("/api/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
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
# Usage limits
# ------------------------------------------------------------------------------
def _today_range_utc():
    now = datetime.utcnow()
    return datetime(now.year, now.month, now.day), datetime(now.year, now.month, now.day, 23, 59, 59, 999999)

def _check_and_increment_usage(db: Session, user: User, endpoint: str = "analyze"):
    if bool(getattr(user, "is_paid", False)):
        db.add(UsageLog(user_id=getattr(user, "id", 0) or 0, timestamp=datetime.utcnow(), endpoint=endpoint, status="ok"))
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
        raise HTTPException(status_code=429, detail="Daily free limit reached. Upgrade to Pro for unlimited analyses.")
    db.add(UsageLog(user_id=getattr(user, "id", 0) or 0, timestamp=datetime.utcnow(), endpoint=endpoint, status="ok"))
    db.commit()

def _ensure_pro(current_user: User):
    if not bool(getattr(current_user, "is_paid", False)):
        raise HTTPException(status_code=403, detail="Export is available to Pro accounts only.")

# ------------------------------------------------------------------------------
# File reading helper
# ------------------------------------------------------------------------------
async def _read_upload_as_text(up: UploadFile) -> str:
    if not up: return ""
    body = await up.read()
    name = (up.filename or "").lower()
    if name.endswith((".txt", ".md", ".csv", ".tsv")):
        try: return body.decode("utf-8", errors="ignore")
        except Exception: return ""
    if name.endswith(".docx"):
        try:
            from docx import Document as _Docx
            return "\n".join(p.text for p in _Docx(BytesIO(body)).paragraphs)
        except Exception:
            return ""
    if name.endswith(".pdf"):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(BytesIO(body))
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception:
            return ""
    try: return body.decode("utf-8", errors="ignore")
    except Exception: return ""

# ------------------------------------------------------------------------------
# Analyze (stub content; your engine plugs in here)
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
    extracted = (text or "").strip() or (await _read_upload_as_text(file)).strip() if file else (text or "").strip()
    if not extracted:
        raise HTTPException(status_code=400, detail="No input provided.")
    chosen = [s.strip().upper() for s in (brains or "CFO,COO,CMO,CHRO").split(",") if s.strip()]
    chosen = [b for b in chosen if b in {"CFO","COO","CMO","CHRO","CPO"}] or ["CFO","COO","CMO","CHRO"]

    # simple, deterministic stub
    sections = []
    for b in chosen:
        if b == "CFO":
            sections.append("### CFO\nInsights\n- Cash runway and collections need tight monitoring.\n- Discretionary spend should be gated by ROI.\nRecommendations\n- **Tighten AR**: Shorten DSO by revising payment terms.\n- **Budget guardrails**: Freeze low-ROI lines.\n- **Forecast weekly**: Update a rolling 13-week CF.")
        elif b == "COO":
            sections.append("### COO\nInsights\n- Bottlenecks in handoffs reduce throughput.\n- SOP variance is high across teams.\nRecommendations\n- **Map bottlenecks**: Add checkpoints.\n- **Automate repeatables**: RPA where volume is high.\n- **QA cadence**: Weekly audits.")
        elif b == "CMO":
            sections.append("### CMO\nInsights\n- CAC:LTV has room to improve.\n- Under-invested in compounding channels.\nRecommendations\n- **Double down**: On channels with proven lift.\n- **Weekly experiments**: 3–5 tests with clear readouts.\n- **Retention**: Shore up lifecycle journeys.")
        elif b == "CHRO":
            sections.append("### CHRO\nInsights\n- Engagement pockets are uneven.\n- Attrition risk in key pods.\nRecommendations\n- **Manager 1:1s**: Reinforce feedback loops.\n- **Retention plans**: For high-impact roles.\n- **Mentor ladder**: Pair seniors to coach.")
        elif b == "CPO":
            sections.append("### CPO\nInsights\n- Roadmap prioritization unclear.\n- Voice of customer feedback underused.\nRecommendations\n- **RICE scoring**: Standardize prioritization.\n- **VOC loops**: Close feedback-to-roadmap gap.\n- **Ship cadence**: Smaller, faster iterations.")
    summary = "\n\n".join(sections)
    return {"status":"ok","title":f"Analysis Complete · {', '.join(chosen)}","summary":summary,"provider":"stub","brains":chosen,"chars":len(extracted)}

# ------------------------------------------------------------------------------
# Markdown → structured sections
# ------------------------------------------------------------------------------
def _parse_sections(md_text: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Parse markdown into {brain: {insights:[], recs:[]}}
    Recognizes:
      - Headings: "### CFO" (any brain token)
      - Subheads: "Insights" / "Recommendations" (case-insensitive)
      - Bullets: "- text" or "1. text"
      - "**Title**: details" pattern for bold title
    """
    data: Dict[str, Dict[str, List[str]]] = {}
    brain: Optional[str] = None
    mode: Optional[str] = None  # 'insights'|'recs'
    for raw in md_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("### "):
            brain = line[4:].strip().upper()
            if brain not in data:
                data[brain] = {"insights": [], "recs": []}
            mode = None
            continue
        low = line.lower()
        if low == "insights":
            mode = "insights"; continue
        if low.startswith("recommendation"):
            mode = "recs"; continue
        if brain and mode in ("insights","recs"):
            # normalize bullets/numbers
            item = re.sub(r"^(\d+\.|\-|\•)\s*", "", line).strip()
            data[brain]["insights" if mode=="insights" else "recs"].append(item)
    return data

def _collective_insights(struct: Dict[str, Dict[str, List[str]]], top_per_brain: int = 1) -> List[str]:
    out: List[str] = []
    for b in PREFERRED_ORDER:
        if b in struct and struct[b]["insights"]:
            out.extend(struct[b]["insights"][:top_per_brain])
    # Include any other brains not in preferred order
    for b, v in struct.items():
        if b not in PREFERRED_ORDER and v["insights"]:
            out.extend(v["insights"][:top_per_brain])
    return out

def _split_bold_title(text: str) -> Tuple[Optional[str], str]:
    """
    Parse '**Title**: rest' or '**Title** – rest' → (Title, rest)
    Returns (None, text) if no bold pattern found.
    """
    m = re.match(r"^\*\*(.+?)\*\*\s*[:\-–]\s*(.*)$", text)
    if m:
        title = m.group(1).strip()
        rest = m.group(2).strip()
        return (title or None, rest)
    # Remove raw ** if present without colon
    if "**" in text:
        text = text.replace("**", "")
    return (None, text)

# ------------------------------------------------------------------------------
# DOCX writer (Collective Insights first; then {BRAIN} Recommends)
# ------------------------------------------------------------------------------
def _write_docx(title: str, md_text: str) -> BytesIO:
    struct = _parse_sections(md_text)
    doc = Document()
    h = doc.add_heading(title, level=1)
    for run in h.runs:
        run.font.size = Pt(16)

    # Collective Insights
    coll = _collective_insights(struct, top_per_brain=1)
    if coll:
        doc.add_heading("Collective Insights", level=2)
        for idx, item in enumerate(coll, 1):
            t, rest = _split_bold_title(item)
            p = doc.add_paragraph(style="List Number")
            if t:
                r1 = p.add_run(f"{t}")
                r1.bold = True
                if rest:
                    p.add_run(f": {rest}")
            else:
                p.add_run(item)

    # Recommendations per brain (preferred order)
    for brain in PREFERRED_ORDER + [b for b in struct.keys() if b not in PREFERRED_ORDER]:
        recs = struct.get(brain, {}).get("recs", [])
        if not recs:
            continue
        doc.add_heading(f"{brain} Recommends", level=2)
        for idx, item in enumerate(recs, 1):
            t, rest = _split_bold_title(item)
            p = doc.add_paragraph(style="List Number")
            if t:
                r1 = p.add_run(f"{t}")
                r1.bold = True
                if rest:
                    p.add_run(f": {rest}")
            else:
                p.add_run(item)

    bio = BytesIO(); doc.save(bio); bio.seek(0); return bio

# ------------------------------------------------------------------------------
# PDF writer (mirrors DOCX layout)
# ------------------------------------------------------------------------------
def _write_pdf(title: str, md_text: str) -> BytesIO:
    struct = _parse_sections(md_text)
    bio = BytesIO()
    c = canvas.Canvas(bio, pagesize=A4)
    width, height = A4
    left, right, top, bottom = 50, width - 50, height - 50, 50
    y = top

    def new_page():
        nonlocal y
        c.showPage()
        y = top

    def draw_para(text: str, bold: bool = False, indent: int = 0, leading: int = 14, wrap_at: int = 100):
        nonlocal y
        if bold: c.setFont("Helvetica-Bold", 12)
        else: c.setFont("Helvetica", 11)
        for ln in wrap(text, wrap_at):
            if y < bottom: new_page()
            c.drawString(left + indent, y, ln)
            y -= leading

    # Title
    c.setFont("Helvetica-Bold", 14)
    for ln in wrap(title, 80):
        if y < bottom: new_page()
        c.drawString(left, y, ln); y -= 18

    # Collective Insights
    coll = _collective_insights(struct, top_per_brain=1)
    if coll:
        draw_para("Collective Insights", bold=True, wrap_at=85)
        for i, item in enumerate(coll, 1):
            t, rest = _split_bold_title(item)
            # "1. " prefix; bold title then normal rest
            prefix = f"{i}. "
            if y < bottom: new_page()
            c.setFont("Helvetica", 11)
            c.drawString(left, y, prefix);  # prefix
            x_offset = left + 20
            if t:
                c.setFont("Helvetica-Bold", 11)
                c.drawString(x_offset, y, t)
                x_offset += 7 * len(t) / 2  # rough offset; subsequent lines wrap as normal
                if rest:
                    c.setFont("Helvetica", 11)
                    c.drawString(x_offset, y, f": {rest}")
            else:
                c.setFont("Helvetica", 11)
                c.drawString(x_offset, y, item)
            y -= 14

    # Recommendations per brain
    for brain in PREFERRED_ORDER + [b for b in struct.keys() if b not in PREFERRED_ORDER]:
        recs = struct.get(brain, {}).get("recs", [])
        if not recs: continue
        draw_para(f"{brain} Recommends", bold=True, wrap_at=85)
        for i, item in enumerate(recs, 1):
            t, rest = _split_bold_title(item)
            prefix = f"{i}. "
            if y < bottom: new_page()
            c.setFont("Helvetica", 11)
            c.drawString(left, y, prefix)
            x_offset = left + 20
            if t:
                c.setFont("Helvetica-Bold", 11)
                c.drawString(x_offset, y, t)
                x_offset += 7 * len(t) / 2
                if rest:
                    c.setFont("Helvetica", 11)
                    c.drawString(x_offset, y, f": {rest}")
            else:
                c.setFont("Helvetica", 11)
                c.drawString(x_offset, y, item)
            y -= 14

    c.save(); bio.seek(0); return bio

# ------------------------------------------------------------------------------
# Export endpoints (POST + GET; /word alias)
# ------------------------------------------------------------------------------
def _extract_export_content(payload: Optional[dict], title_q: Optional[str], md_q: Optional[str], md_text_q: Optional[str]):
    title = (payload.get("title") if payload else None) or (title_q or "CAIO Analysis")
    md_text = ((payload.get("md_text") if payload else None) or (payload.get("markdown") if payload else None) or md_text_q or md_q or "").strip()
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
    if not content: raise HTTPException(status_code=400, detail="No content to export.")
    bio = _write_docx(t, content)
    headers = {"Content-Disposition": f'attachment; filename="{t.replace(" ", "_").lower()}.docx"'}
    return StreamingResponse(bio, headers=headers, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

@app.api_route("/api/export/word", methods=["POST", "GET"])
def export_word(
    request: Request,
    payload: Optional[dict] = Body(None),
    title: Optional[str] = Query(None),
    md: Optional[str] = Query(None),
    md_text: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
):
    return export_docx(request, payload, title, md, md_text, current_user)

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
    if not content: raise HTTPException(status_code=400, detail="No content to export.")
    bio = _write_pdf(t, content)
    headers = {"Content-Disposition": f'attachment; filename="{t.replace(" ", "_").lower()}.pdf"'}
    return StreamingResponse(bio, headers=headers, media_type="application/pdf")

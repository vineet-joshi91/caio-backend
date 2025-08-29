# main.py
import os, logging, traceback, json
from typing import Optional, List, Dict
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

# Ensure tables exist (Postgres or SQLite fallback)
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
# Analyze (Demo: OpenRouter-only, capped to 2 brains; Pro: OpenAI→OpenRouter, full)
# Extracts PDF/DOCX/XLSX/CSV/TXT; never breaks UX (safe stubs on failure)
# ------------------------------------------------------------------------------
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
    "CFO":  "Chief Financial Officer — own capital allocation and unit economics; analyze revenue mix, gross-margin bridge, OPEX discipline, pricing/discount leakage, cash conversion cycle (DSO/DPO/DIH), working-capital health, runway & covenant headroom, cohort LTV/CAC and payback, scenario planning and risk (FX, rates, credit).",
    "COO":  "Chief Operating Officer — own cost-to-serve and reliability; analyze capacity & utilization, throughput, takt/cycle time, queue depth and SLA adherence, constraint/bottleneck analysis (TOC), FPY/defect rate and rework, on-time delivery, inventory turns & fulfillment, MTTR/MTBF, incident/safety rate, and logistics cost.",
    "CHRO": "Chief Human Resources Officer — own organizational effectiveness; analyze attrition/retention and regretted loss, eNPS/engagement heat-maps, performance & succession matrices, span/layer and org design, pay equity & compensation bands, promotion velocity, compliance & ER cases, DEI pipeline health, learning hours and skills taxonomy.",
    "CMO":  "Chief Marketing Officer — own efficient growth; analyze demand-mix and MMM/incrementality, CAC/LTV & payback by segment/channel, funnel conversion & leakage, creative testing (CTR/CVR) and lift, brand lift & share of voice, SEO/SEM and content velocity, partner/affiliate ROI, pricing/packaging signals, retention and reactivation.",
    "CPO":  "Chief People Officer — own workforce planning & external talent acquisition; analyze req pipeline velocity/coverage, source-mix & channel ROI, recruiter productivity, time-to-hire and time-to-start, offer-acceptance & decliner reasons, cost-per-hire & vendor performance, quality-of-hire (90-day survival, time-to-productivity), geo strategy & workforce ramps, employer brand performance.",
    }
    role = role_map.get(brain, "Executive Advisor")

    # One-message (user) prompt that behaves like a system+user combo.
    # Keeps your existing _call_openai/_call_openrouter unchanged.
    return f"""
You are {role}.
Return STRICT MARKDOWN ONLY for **{brain}** with this structure and nothing else:

### Insights
1. <concise, concrete insight using data in the brief/text>
2. <insight>
3. <insight>
4. <optional – only if strong>
5. <optional – only if strong>

### Recommendations
1. **<3–6 word unique headline>**: <ONE sentence action tailored to {brain}; include a lever/metric/₹/$/% or timeline.>
2. **<different headline>**: <ONE sentence action.>
3. **<different headline>**: <ONE sentence action.>

RULES
- Number lists EXACTLY "1. ", "2. ", ... (never "1.:"; no blank items).
- Each recommendation MUST start with **bold headline** then a colon + ONE sentence.
- Headlines must be unique within {brain} and should avoid repeating the same lead verb across roles.
  Prefer varied verbs: Rebalance, Optimize, Streamline, De-risk, Accelerate, Elevate, Consolidate, Diversify.
- Be specific and actionable (add a lever/metric/timeline/owner where possible).
- NO extra sections, code blocks, preambles, or explanations. Markdown only.

[BRIEF]
{brief or "(none)"}

[DOCUMENT TEXT]
{(extracted or "")[:120000]}
""".strip()

# ---------- lightweight readers ----------
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

def _read_csv(body: bytes) -> str:
    try:
        txt = body.decode("utf-8", errors="ignore")
        return "\n".join(txt.splitlines()[:200])  # first ~200 lines
    except Exception:
        return ""

def _read_xlsx(body: bytes) -> str:
    try:
        from openpyxl import load_workbook
        wb = load_workbook(filename=BytesIO(body), data_only=True)
        out: List[str] = []
        for ws in wb.worksheets[:2]:  # first 2 sheets
            out.append(f"# Sheet: {ws.title}")
            for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
                if i > 200:
                    break
                cells = ["" if c is None else str(c) for c in row]
                out.append(",".join(cells))
        return "\n".join(out)
    except Exception:
        return ""

async def _extract_text(file: Optional[UploadFile]) -> str:
    if not file:
        return ""
    body = await file.read()
    name = (file.filename or "").lower()
    if name.endswith(".pdf"):  return _read_pdf(body)
    if name.endswith(".docx"): return _read_docx(body)
    if name.endswith(".xlsx"): return _read_xlsx(body)
    if name.endswith(".csv"):  return _read_csv(body)
    return _read_txt(body)  # txt/md/unknown

# ---------- LLM callers ----------
def _call_openai_chat(prompt: str) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")

    project = os.getenv("OPENAI_PROJECT")
    org = os.getenv("OPENAI_ORG")

    # Try SDK first; then REST with headers supporting `sk-proj-...`
    try:
        import openai
        if hasattr(openai, "OpenAI"):
            client = openai.OpenAI(api_key=key, organization=org, project=project)
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return r.choices[0].message.content
        else:
            openai.api_key = key
            if org:
                openai.organization = org
            r = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return r.choices[0].message.content
    except Exception:
        import requests
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if org:
            headers["OpenAI-Organization"] = org
        if project:
            headers["OpenAI-Project"] = project
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions",
                             headers=headers, data=json.dumps(data), timeout=60)
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenAI {resp.status_code}: {resp.text}")
        j = resp.json()
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
        "model": "openai/gpt-4o-mini",  # pick any supported model name here
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                      headers=headers, data=json.dumps(payload), timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenRouter {r.status_code}: {r.text}")
    return r.json()["choices"][0]["message"]["content"]

# ---------- provider selection (tier-aware) ----------
def _pick_provider(is_paid: bool) -> List[str]:
    """
    Returns an ordered list of providers to try based on tier and available keys.
    Demo:   LLM_PROVIDER_DEMO (default 'openrouter')
    Pro:    LLM_PROVIDER_PRO  (default 'openai,openrouter')
    """
    if not is_paid:
        raw = os.getenv("LLM_PROVIDER_DEMO", "openrouter")
    else:
        raw = os.getenv("LLM_PROVIDER_PRO", "openai,openrouter")

    providers = [p.strip().lower() for p in raw.split(",") if p.strip()]

    available: List[str] = []
    for p in providers:
        if p == "openai" and os.getenv("OPENAI_API_KEY"):
            available.append("openai")
        if p == "openrouter" and os.getenv("OPENROUTER_API_KEY"):
            available.append("openrouter")
    return available or []

# ---------- analyze endpoint ----------
@app.post("/api/analyze")
async def analyze(
    request: Request,
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    brains: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # DEMO flow (OpenRouter-only by env; capped brains)
    if not getattr(current_user, "is_paid", False):
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

    # PRO flow (OpenAI first, then OpenRouter; full brains)
    try:
        extracted = await _extract_text(file)
    except Exception as e:
        logger.error("Extraction failed: %s\n%s", e, traceback.format_exc())
        extracted = ""  # allow brief-only
    brief = (text or "").strip()
    chosen = _choose_brains(brains, is_paid=True)

    if not extracted and not brief:
        raise HTTPException(status_code=400, detail="Please upload a file or provide text")

    prompts = {b: _brain_prompt(brief, extracted, b) for b in chosen}

    provider_chain = _pick_provider(is_paid=True)
    summaries: Dict[str, str] = {}
    used_provider = "stub"
    error_notes: List[str] = []

    if provider_chain:
        for provider in provider_chain:
            try:
                if provider == "openai":
                    for b, p in prompts.items():
                        summaries[b] = _call_openai_chat(p)
                elif provider == "openrouter":
                    for b, p in prompts.items():
                        summaries[b] = _call_openrouter_chat(p)
                used_provider = provider
                break  # success
            except Exception as e:
                logger.error("Provider %s failed: %s", provider, e)
                error_notes.append(f"{provider}:{e.__class__.__name__}")
                summaries = {}
                continue

    # Fallback to stubs so UX never breaks
    if not summaries:
        for b in prompts.keys():
            summaries[b] = (
                f"[Stub after error] {b} analysis. "
                f"Errors: {', '.join(error_notes) if error_notes else 'no-provider'}."
            )

    combined = [f"### {b}\n{summaries[b]}" for b in chosen]
    return {
        "status": "ok",
        "title": f"Analysis Complete · {', '.join(chosen)}",
        "summary": "\n\n".join(combined),
        "meta": {"provider": used_provider, "brains": chosen, "chars": len(extracted)},
    }

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

@app.get("/api/debug/routers")
def debug_routers():
    out = []
    for r in app.routes:
        methods = sorted(list(getattr(r, "methods", [])))
        path = getattr(r, "path", "")
        if path.startswith("/api/"):
            out.append({"path": path, "methods": methods})
    return out

# ------------------------------------------------------------------------------
# Other routers (don’t crash boot if optional modules are missing)
# ------------------------------------------------------------------------------
try:
    from routes_public_config import router as public_cfg_router
    app.include_router(public_cfg_router)  # /api/public-config
except Exception as e:
    logger.warning(f"routes_public_config not loaded: {e}")

# ---- Payments router (Razorpay) ----
import logging
logger = logging.getLogger("uvicorn.error")

from payment_routes import router as payments_router   # <-- will raise on import errors
app.include_router(payments_router)                    # prefix is defined inside payment_routes.py
logger.info("✅ payments_router mounted")

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

# ===================== EXPORT ENDPOINTS (PRO ONLY) =====================
from fastapi import Body
from fastapi.responses import StreamingResponse
from docx import Document
from docx.shared import Pt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from textwrap import wrap

def _ensure_pro(current_user: User):
    if not bool(getattr(current_user, "is_paid", False)):
        raise HTTPException(status_code=403, detail="Export is available to Pro accounts only.")

def _parse_analysis_md(md_text: str) -> list[tuple[str, str]]:
    """
    Quick splitter: returns [(brain, content_markdown), ...]
    Assumes sections start with '### {BRAIN}' like we generate.
    """
    parts = []
    current_brain = None
    buf = []
    for line in md_text.splitlines():
        if line.startswith("### "):
            # flush previous
            if current_brain:
                parts.append((current_brain, "\n".join(buf).strip()))
                buf = []
            current_brain = line.replace("###", "").strip()
        else:
            buf.append(line)
    if current_brain:
        parts.append((current_brain, "\n".join(buf).strip()))
    return parts or [("Analysis", md_text)]

def _write_docx(title: str, md_text: str) -> BytesIO:
    doc = Document()
    doc.styles['Normal'].font.name = 'Calibri'
    doc.styles['Normal'].font.size = Pt(11)

    doc.add_heading(title or "CAIO Analysis", level=1)

    for brain, content in _parse_analysis_md(md_text):
        doc.add_heading(brain, level=2)
        # simple markdown bullets → Word paragraphs
        for para in content.split("\n\n"):
            p = para.strip()
            if not p:
                continue
            if p.startswith(("* ", "- ")):
                # bullet list
                for line in p.splitlines():
                    if line.strip().startswith(("* ", "- ")):
                        doc.add_paragraph(line.strip()[2:], style=None).style = None
                        doc.paragraphs[-1].style = doc.styles['List Bullet']
            elif p[0:2].isdigit() or p.startswith("1. "):
                # numbered-ish: split on lines that start with N.
                for line in p.splitlines():
                    doc.add_paragraph(line.split(". ", 1)[-1], style=None)
                    doc.paragraphs[-1].style = doc.styles['List Number']
            else:
                doc.add_paragraph(p)

    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

def _write_pdf(title: str, md_text: str) -> BytesIO:
    # Simple text PDF; for fancier layout we can switch to reportlab platypus later
    bio = BytesIO()
    c = canvas.Canvas(bio, pagesize=A4)
    width, height = A4
    left, top = 50, height - 50
    y = top

    def draw_line(text, bold=False):
        nonlocal y
        if y < 60:
            c.showPage()
            y = top
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 11 if not bold else 12)
        # wrap lines to page width
        max_chars = 95
        for wline in wrap(text, max_chars):
            c.drawString(left, y, wline)
            y -= 14

    draw_line(title or "CAIO Analysis", bold=True)
    y -= 8

    for brain, content in _parse_analysis_md(md_text):
        y -= 4
        draw_line(brain, bold=True)
        y -= 6
        for para in content.split("\n\n"):
            p = para.strip()
            if not p:
                continue
            for line in p.splitlines():
                draw_line(line)

    c.save()
    bio.seek(0)
    return bio

@app.post("/api/export/docx")
def export_docx(
    payload: dict = Body(..., example={"title": "Analysis Complete · CFO, COO...", "markdown": "### CFO\n• ..."}),
    current_user: User = Depends(get_current_user),
):
    _ensure_pro(current_user)
    title = (payload.get("title") or "CAIO Analysis").strip()
    md = payload.get("markdown") or ""
    stream = _write_docx(title, md)
    headers = {"Content-Disposition": f'attachment; filename="CAIO-Analysis.docx"'}
    return StreamingResponse(stream, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers=headers)

@app.post("/api/export/pdf")
def export_pdf(
    payload: dict = Body(..., example={"title": "Analysis Complete · CFO, COO...", "markdown": "### CFO\n• ..."}),
    current_user: User = Depends(get_current_user),
):
    _ensure_pro(current_user)
    title = (payload.get("title") or "CAIO Analysis").strip()
    md = payload.get("markdown") or ""
    stream = _write_pdf(title, md)
    headers = {"Content-Disposition": f'attachment; filename="CAIO-Analysis.pdf"'}
    return StreamingResponse(stream, media_type="application/pdf", headers=headers)
# =================== END EXPORT ENDPOINTS ===================

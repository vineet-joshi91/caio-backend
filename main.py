# main.py
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("caio")

# --- FastAPI app FIRST ---
app = FastAPI(title="CAIO Backend", version="0.1.0")

# --- CORS ---
DEFAULT_ORIGINS = [
    "https://caio-frontend.vercel.app",
    "https://caioai.netlify.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
extra = os.getenv("ALLOWED_ORIGINS", "")
if extra:
    DEFAULT_ORIGINS += [o.strip() for o in extra.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEFAULT_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

@app.options("/{path:path}")
def cors_preflight(path: str):
    # Helpful for quick CORS diagnostics from the browser
    return JSONResponse({"ok": True})

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}

# --- Routers (import AFTER app exists) ---
# Public config (pricing/flags)
try:
    from routes_public_config import router as public_cfg_router
    app.include_router(public_cfg_router, tags=["public"])
    logger.info("Loaded routes_public_config")
except Exception as e:
    logger.warning(f"routes_public_config not loaded: {e}")

# Signup route (new/updated)
try:
    from signup_routes import router as signup_router
    app.include_router(signup_router)
    logger.info("Loaded /api/signup")
except Exception as e:
    logger.warning(f"signup_routes not loaded: {e}")

# Payments
try:
    from payment_routes import router as payments_router
    # payment_routes already sets its own prefix or not; be explicit if needed:
    app.include_router(payments_router, prefix="/api/payments")
    logger.info("Loaded /api/payments/*")
except Exception as e:
    logger.warning(f"payment_routes not loaded: {e}")

# Contact form (optional)
try:
    from contact_routes import router as contact_router
    app.include_router(contact_router)
    logger.info("Loaded /api/contact")
except Exception as e:
    logger.warning(f"contact_routes not loaded: {e}")

# --- Keep the rest of your existing endpoints below ---
# e.g. /api/login, /api/profile, file upload, analysis, etc.
# (No changes needed; they’ll continue to work.)

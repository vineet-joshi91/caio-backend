# payment_routes.py
from __future__ import annotations
import os, json, hmac, hashlib, logging
from typing import Dict, Any, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

# Optional import guard so local dev doesn't break if the lib isn't installed
try:
    import razorpay
    from razorpay.errors import BadRequestError, ServerError, SignatureVerificationError
except Exception:  # pragma: no cover
    razorpay = None
    BadRequestError = ServerError = SignatureVerificationError = Exception  # type: ignore

log = logging.getLogger("payments")
router = APIRouter(prefix="/api/payments", tags=["payments"])

# ------------------------- Environment / Config ------------------------------

MODE = "razorpay"

RZP_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "").strip()
RZP_SECRET = os.getenv("RAZORPAY_SECRET", "").strip()
RZP_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "").strip()

# Which currency to use if we can't detect anything
DEFAULT_CURRENCY = os.getenv("PAY_DEFAULT_CURRENCY", "INR").upper()

# Interval copy (purely for display)
PAY_PERIOD = os.getenv("PAY_PERIOD", "monthly")
PAY_INTERVAL = int(os.getenv("PAY_INTERVAL", "1"))
PAY_INTERVAL_TEXT = os.getenv("PAY_INTERVAL_TEXT", "every 1 monthly")

# Pricing is data-driven via JSON. Edit the env only.
# Example (default when env missing):
# {"INR":{"amount_major":499,"symbol":"₹"},"USD":{"amount_major":49,"symbol":"$"}}
def _load_pricing() -> Dict[str, Dict[str, Any]]:
    raw = os.getenv("PRICING_JSON", "").strip()
    if raw:
        try:
            data = json.loads(raw)
            # normalize keys
            return {k.upper(): v for k, v in data.items()}
        except Exception as e:
            log.warning("Invalid PRICING_JSON, using defaults. %s", e)
    return {
        "INR": {"amount_major": 499, "symbol": "₹"},
        "USD": {"amount_major": 49,  "symbol": "$"},
    }

PRICING = _load_pricing()
ALLOWED_CURRENCIES = set(PRICING.keys())

HAS_SECRET = bool(RZP_SECRET)

_rzp: Optional["razorpay.Client"] = None
if razorpay and RZP_KEY_ID and RZP_SECRET:
    _rzp = razorpay.Client(auth=(RZP_KEY_ID, RZP_SECRET))
else:
    log.warning("Razorpay client not initialized (missing keys or lib).")

# In-memory cache: (currency, amount, period, interval) -> plan_id
PLAN_CACHE: Dict[Tuple[str, int, str, int], str] = {}

# ------------------------- Helpers ------------------------------------------

def _pick_currency(request: Request, explicit: Optional[str]) -> str:
    """
    1) If explicit currency provided and supported -> use it.
    2) Else detect from headers (IN -> INR, else USD if available).
    3) Else fallback to DEFAULT_CURRENCY (if supported), otherwise first available.
    """
    if explicit:
        c = explicit.upper()
        if c in ALLOWED_CURRENCIES:
            return c
        raise HTTPException(status_code=400, detail=f"Unsupported currency: {explicit}")

    # Geo hints from common CDNs / platforms
    cc = (
        request.headers.get("x-vercel-ip-country")
        or request.headers.get("cf-ipcountry")
        or request.headers.get("x-country-code")
        or ""
    ).upper()

    if cc == "IN" and "INR" in ALLOWED_CURRENCIES:
        return "INR"
    if "USD" in ALLOWED_CURRENCIES:
        return "USD"

    # fallback
    if DEFAULT_CURRENCY in ALLOWED_CURRENCIES:
        return DEFAULT_CURRENCY
    return next(iter(ALLOWED_CURRENCIES))  # last-resort

def _pricing_for(currency: str) -> Dict[str, Any]:
    cfg = PRICING.get(currency.upper())
    if not cfg or "amount_major" not in cfg:
        raise HTTPException(status_code=500, detail=f"Pricing missing for {currency}")
    return cfg

def _ensure_plan(currency: str, amount_major: int) -> str:
    key = (currency, amount_major, PAY_PERIOD, PAY_INTERVAL)
    if key in PLAN_CACHE:
        return PLAN_CACHE[key]
    if not _rzp:
        raise HTTPException(status_code=503, detail="Razorpay not initialized on server.")
    try:
        plan = _rzp.plan.create(
            {
                "period": PAY_PERIOD,
                "interval": PAY_INTERVAL,
                "item": {
                    "name": f"CAIO Pro ({currency})",
                    "amount": amount_major * 100,  # subunits
                    "currency": currency,
                },
            }
        )
        PLAN_CACHE[key] = plan["id"]
        return plan["id"]
    except BadRequestError as e:
        msg = getattr(e, "args", [str(e)])[0]
        raise HTTPException(status_code=400, detail=f"Plan create failed: {msg}") from e
    except ServerError as e:
        msg = getattr(e, "args", [str(e)])[0]
        raise HTTPException(status_code=502, detail=f"Razorpay server error: {msg}") from e

# ------------------------- Routes -------------------------------------------

@router.get("/ping")
def ping():
    return {"ok": True, "mode": MODE, "currencies": sorted(ALLOWED_CURRENCIES)}

@router.get("/subscription-config")
def subscription_config(request: Request, currency: Optional[str] = None):
    display_currency = _pick_currency(request, currency)
    out = {
        "mode": MODE,
        "key_id": RZP_KEY_ID or None,
        "has_secret": HAS_SECRET,
        "interval": PAY_INTERVAL_TEXT,
        "defaultCurrency": display_currency,
        "pricing": PRICING,  # frontend can render toggle if >1
    }
    return out

class CreateBody(BaseModel):
    currency: Optional[str] = None
    notes: Optional[Dict[str, Any]] = None

@router.post("/subscription/create", status_code=201)
def create_subscription(request: Request, body: CreateBody):
    if MODE != "razorpay":
        raise HTTPException(status_code=400, detail="Only Razorpay mode is implemented.")
    currency = _pick_currency(request, body.currency)
    p = _pricing_for(currency)
    amount_major = int(p["amount_major"])

    plan_id = _ensure_plan(currency, amount_major)

    try:
        sub = _rzp.subscription.create(
            {"plan_id": plan_id, "total_count": 0, "customer_notify": 1, "notes": body.notes or {}}
        )
        return {
            "currency": currency,
            "amount_major": amount_major,
            "plan_id": plan_id,
            "subscription_id": sub.get("id"),
            "status": sub.get("status"),
            "short_url": sub.get("short_url"),
            "raw": sub,
        }
    except BadRequestError as e:
        msg = getattr(e, "args", [str(e)])[0]
        raise HTTPException(status_code=400, detail=f"Subscription create failed: {msg}") from e
    except ServerError as e:
        msg = getattr(e, "args", [str(e)])[0]
        raise HTTPException(status_code=502, detail=f"Razorpay server error: {msg}") from e

@router.post("/razorpay/webhook")
async def razorpay_webhook(request: Request):
    if not RZP_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Webhook secret not configured")
    body = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")
    try:
        expected = hmac.new(RZP_WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise SignatureVerificationError("Invalid signature", None)  # type: ignore
        payload = json.loads(body.decode("utf-8"))
        log.info("Webhook OK: %s", payload.get("event"))
        return {"ok": True}
    except SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        log.exception("Webhook error: %s", e)
        raise HTTPException(status_code=500, detail="Webhook processing error")

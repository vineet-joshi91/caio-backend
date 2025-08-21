# payment_routes.py
from fastapi import APIRouter, Depends, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import os, razorpay, httpx, hmac, hashlib, json, logging

from db import get_db, User
from auth import get_current_user

logger = logging.getLogger("payments")
router = APIRouter(prefix="/api/payments", tags=["payments"])

# --- ENV ---
RAZORPAY_KEY_ID   = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_SECRET   = os.getenv("RAZORPAY_SECRET")
WEBHOOK_SECRET    = os.getenv("RAZORPAY_WEBHOOK_SECRET")  # must match Razorpay Dashboard webhook secret
PUBLIC_CONFIG_URL = os.getenv("PUBLIC_CONFIG_URL", "https://caio-backend.onrender.com/api/public-config")

if not (RAZORPAY_KEY_ID and RAZORPAY_SECRET):
    raise RuntimeError("Razorpay env vars missing: set RAZORPAY_KEY_ID and RAZORPAY_SECRET")

rz = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_SECRET))

# --- Helpers ---
async def get_region_pricing(req: Request) -> tuple[str, int]:
    """
    Reads region-aware Pro price from public-config so UI and server match.
    Returns (currency, amount_in_minor_units).
    """
    force = req.query_params.get("force")
    params = {"force": force} if force else {}
    async with httpx.AsyncClient(timeout=8.0) as x:
        r = await x.get(PUBLIC_CONFIG_URL, params=params)
        r.raise_for_status()
        cfg = r.json()
    currency = cfg.get("currency", "INR")
    pro = (cfg.get("plans") or {}).get("pro") or {}
    price_major = int(pro.get("price", 1999))
    amount_minor = price_major * 100  # INR paise or USD cents
    return currency, amount_minor

# --- Routes ---

@router.get("/config")
async def payments_config(request: Request):
    """Frontend helper — returns public key and active currency."""
    try:
        currency, _ = await get_region_pricing(request)
        return {"key_id": RAZORPAY_KEY_ID, "currency": currency}
    except Exception as e:
        logger.exception("payments_config error")
        raise HTTPException(status_code=500, detail=f"Payments config error: {e}")

@router.post("/create-order")
async def create_order(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Creates a Razorpay order with correct amount for region."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        currency, amount_minor = await get_region_pricing(request)

        order = rz.order.create(dict(
            amount=amount_minor,
            currency=currency,
            receipt=f"pro-{current_user.email}",
            payment_capture=1,
            notes={"email": current_user.email, "plan": "pro"}
        ))
        return {
            "order_id": order["id"],
            "amount": order["amount"],
            "currency": order["currency"],
            "key_id": RAZORPAY_KEY_ID,
        }
    except razorpay.errors.BadRequestError as e:
        logger.exception("Razorpay bad request")
        raise HTTPException(status_code=400, detail=f"Razorpay error: {e}")
    except Exception as e:
        logger.exception("Create order error")
        raise HTTPException(status_code=500, detail=f"Create order failed: {e}")

@router.post("/webhook")
async def webhook(
    request: Request,
    x_razorpay_signature: str = Header(None),
    db: Session = Depends(get_db),
):
    """
    Webhook: on payment.captured => mark user.is_paid = True (notes.email).
    """
    if not WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Webhook secret not configured")

    try:
        body_bytes = await request.body()
        payload = body_bytes.decode("utf-8")

        # Verify signature per Razorpay docs
        digest = hmac.new(WEBHOOK_SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(digest, (x_razorpay_signature or "")):
            return JSONResponse({"status": "invalid-signature"}, status_code=400)

        event = json.loads(payload)
        etype = event.get("event", "")
        pay_entity = (event.get("payload", {}).get("payment", {}) or {}).get("entity", {})
        order_entity = (event.get("payload", {}).get("order", {}) or {}).get("entity", {})

        email = (pay_entity.get("notes") or {}).get("email") or (order_entity.get("notes") or {}).get("email")
        if etype == "payment.captured" and email:
            user = db.query(User).filter(User.email == email).first()
            if user:
                user.is_paid = True
                db.add(user)
                db.commit()
                return {"status": "ok", "updated": True}
            return {"status": "ok", "updated": False, "reason": "user-not-found"}

        return {"status": "ignored", "event": etype}
    except Exception as e:
        logging.exception("Webhook processing error")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

@router.get("/status")
async def status(current_user: User = Depends(get_current_user)):
    """Quick check for app to know if the user is paid."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"email": current_user.email, "is_paid": bool(current_user.is_paid)}

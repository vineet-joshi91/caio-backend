# payment_routes.py
from fastapi import APIRouter, Depends, HTTPException, Request, Header, Body
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import os, razorpay, httpx, hmac, hashlib, json, logging

from db import get_db, User
from auth import get_current_user

logger = logging.getLogger("payments")

# NOTE: prefix lives here; DO NOT add another prefix when including in main.py
router = APIRouter(prefix="/api/payments", tags=["payments"])

RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID") or ""
RAZORPAY_SECRET = os.getenv("RAZORPAY_SECRET") or ""
WEBHOOK_SECRET  = os.getenv("RAZORPAY_WEBHOOK_SECRET") or ""
PUBLIC_CONFIG_URL = os.getenv("PUBLIC_CONFIG_URL", "https://caio-backend.onrender.com/api/public-config")

# Subscription knobs (optional)
SUB_AMOUNT_INR      = int(os.getenv("CAIO_SUB_AMOUNT_INR", "499"))
SUB_INTERVAL        = os.getenv("CAIO_SUB_INTERVAL", "monthly")     # daily/weekly/monthly/yearly
SUB_INTERVAL_COUNT  = int(os.getenv("CAIO_SUB_INTERVAL_COUNT", "1"))
SUB_TOTAL_COUNT     = int(os.getenv("CAIO_SUB_TOTAL_COUNT", "12"))
RAZORPAY_PLAN_ID    = (os.getenv("RAZORPAY_PLAN_ID") or "").strip()

_rz_client = None
def get_client() -> razorpay.Client:
    global _rz_client
    if _rz_client:
        return _rz_client
    if not (RAZORPAY_KEY_ID and RAZORPAY_SECRET):
        raise HTTPException(status_code=500, detail="Payments not configured (missing Razorpay keys).")
    _rz_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_SECRET))
    return _rz_client

async def get_region_pricing(req: Request) -> tuple[str, int]:
    """Returns (currency, amount_in_minor_units) for the 'pro' plan."""
    force = req.query_params.get("force")
    params = {"force": force} if force else {}
    async with httpx.AsyncClient(timeout=8.0) as x:
        r = await x.get(PUBLIC_CONFIG_URL, params=params)
        r.raise_for_status()
        cfg = r.json()
    currency = (cfg.get("currency") or "INR").upper()
    pro = (cfg.get("plans") or {}).get("pro") or {}
    price_major = int(pro.get("price", SUB_AMOUNT_INR))
    amount_minor = price_major * 100   # paise / cents
    return currency, amount_minor

# -------------------------
# BASIC CONFIG + ONE-TIME ORDER (existing)
# -------------------------

@router.get("/config")
async def payments_config(request: Request):
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
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        currency, amount_minor = await get_region_pricing(request)
        client = get_client()
        order = client.order.create(dict(
            amount=amount_minor,
            currency=currency,
            receipt=f"pro-{current_user.email}",
            payment_capture=1,
            notes={"email": current_user.email, "plan": "pro"},
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

# -------------------------
# SUBSCRIPTION FLOW (new)
# -------------------------

def _ensure_keys():
    if not (RAZORPAY_KEY_ID and RAZORPAY_SECRET):
        raise HTTPException(status_code=500, detail="Razorpay keys missing on server")

def _create_or_get_plan(client: razorpay.Client, currency: str) -> str:
    """Use RAZORPAY_PLAN_ID if provided, else create a plan matching our interval/amount."""
    if RAZORPAY_PLAN_ID:
        return RAZORPAY_PLAN_ID
    payload = {
        "period": SUB_INTERVAL,
        "interval": SUB_INTERVAL_COUNT,
        "item": {
            "name": f"CAIO Pro ({SUB_AMOUNT_INR} {currency}/{SUB_INTERVAL})",
            "amount": SUB_AMOUNT_INR * 100,
            "currency": currency,
        },
    }
    plan = client.plan.create(payload)
    return plan["id"]

@router.get("/subscription-config")
async def subscription_config(request: Request):
    _ensure_keys()
    currency, _ = await get_region_pricing(request)
    return {
        "mode": "razorpay",
        "currency": currency,
        "amount_major": SUB_AMOUNT_INR,
        "amount": SUB_AMOUNT_INR * 100,
        "interval": f"every {SUB_INTERVAL_COUNT} {SUB_INTERVAL}",
        "key_id": RAZORPAY_KEY_ID,
    }

@router.post("/subscribe")
async def subscribe(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _ensure_keys()
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        currency, _ = await get_region_pricing(request)
        client = get_client()
        plan_id = _create_or_get_plan(client, currency)

        # Create subscription
        sub = client.subscription.create({
            "plan_id": plan_id,
            "total_count": SUB_TOTAL_COUNT,
            "customer_notify": 1,
            "notes": {"email": current_user.email, "app": "CAIO"},
        })
        return {"subscription_id": sub["id"], "key_id": RAZORPAY_KEY_ID, "email": current_user.email}
    except razorpay.errors.BadRequestError as e:
        logger.exception("Razorpay subscription bad request")
        raise HTTPException(status_code=400, detail=f"Razorpay error: {e}")
    except Exception as e:
        logger.exception("Subscription create error")
        raise HTTPException(status_code=500, detail=f"Subscription create failed: {e}")

@router.post("/verify")
async def verify_subscription(
    payload: dict = Body(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Frontend sends:
      { razorpay_payment_id, razorpay_subscription_id, razorpay_signature }
    Signature = HMAC_SHA256( payment_id + '|' + subscription_id, RAZORPAY_SECRET )
    """
    _ensure_keys()
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    required = ("razorpay_payment_id", "razorpay_subscription_id", "razorpay_signature")
    if any(k not in payload for k in required):
        raise HTTPException(status_code=400, detail="Missing fields for verification")

    to_sign = f"{payload['razorpay_payment_id']}|{payload['razorpay_subscription_id']}"
    digest = hmac.new(RAZORPAY_SECRET.encode("utf-8"), to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(digest, payload["razorpay_signature"]):
        raise HTTPException(status_code=400, detail="Signature verification failed")

    try:
        current_user.is_paid = True
        db.add(current_user); db.commit()
    except Exception as e:
        logger.error("DB update failed after verify: %s", e)
        raise HTTPException(status_code=500, detail="Payment verified but failed to update account")

    return {"ok": True, "message": "Payment verified. Pro is active."}

@router.post("/cancel")
async def cancel_subscription(
    body: dict = Body(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _ensure_keys()
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    sub_id = (body or {}).get("subscription_id")
    if not sub_id:
        raise HTTPException(status_code=400, detail="subscription_id required")

    try:
        client = get_client()
        client.subscription.cancel(sub_id, {"cancel_at_cycle_end": 0})
    except razorpay.errors.BadRequestError as e:
        logger.exception("Razorpay cancel bad request")
        raise HTTPException(status_code=400, detail=f"Razorpay error: {e}")
    except Exception as e:
        logger.exception("Cancel error")
        raise HTTPException(status_code=500, detail=f"Cancel failed: {e}")

    try:
        current_user.is_paid = False
        db.add(current_user); db.commit()
    except Exception as e:
        logger.error("DB update failed after cancel: %s", e)

    return {"ok": True, "message": "Subscription cancelled"}

# -------------------------
# WEBHOOKS (order + subscription events)
# -------------------------

@router.post("/webhook")
async def webhook(
    request: Request,
    x_razorpay_signature: str = Header(None),
    db: Session = Depends(get_db),
):
    if not WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Webhook secret not configured")
    try:
        body_bytes = await request.body()
        payload = body_bytes.decode("utf-8")

        # HMAC SHA256 verification
        digest = hmac.new(WEBHOOK_SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(digest, (x_razorpay_signature or "")):
            return JSONResponse({"status": "invalid-signature"}, status_code=400)

        event = json.loads(payload)
        etype = event.get("event", "")

        pay_entity   = (event.get("payload", {}).get("payment", {}) or {}).get("entity", {})
        order_entity = (event.get("payload", {}).get("order", {}) or {}).get("entity", {})
        sub_entity   = (event.get("payload", {}).get("subscription", {}) or {}).get("entity", {})

        email = (
            (pay_entity.get("notes") or {}).get("email")
            or (order_entity.get("notes") or {}).get("email")
            or (sub_entity.get("notes") or {}).get("email")
        )

        if not email:
            return {"status": "ignored", "event": etype, "reason": "no-email"}

        user = db.query(User).filter(User.email == email).first()
        if not user:
            return {"status": "ok", "updated": False, "event": etype, "reason": "user-not-found"}

        # One-time payment success
        if etype == "payment.captured":
            user.is_paid = True

        # Subscription lifecycle
        if etype in ("subscription.activated", "subscription.charged"):
            user.is_paid = True
        if etype in ("subscription.cancelled", "subscription.paused", "subscription.halted"):
            user.is_paid = False

        db.add(user); db.commit()
        return {"status": "ok", "updated": True, "event": etype}

    except Exception as e:
        logging.exception("Webhook processing error")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

# -------------------------
# STATUS
# -------------------------

@router.get("/status")
async def status(current_user: User = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"email": current_user.email, "is_paid": bool(current_user.is_paid)}

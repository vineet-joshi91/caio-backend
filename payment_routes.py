# payment_routes.py
from fastapi import APIRouter, Depends, HTTPException, Request, Header, Body
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Any, Optional
import os, httpx, hmac, hashlib, json, logging

# Your project helpers (already exist in your repo)
from db import get_db, User
from auth import get_current_user

logger = logging.getLogger("payments")

# The router is self-prefixed here; do NOT add another prefix in main.py
router = APIRouter(prefix="/api/payments", tags=["payments"])

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

RAZORPAY_KEY_ID = (
    os.getenv("RAZORPAY_KEY_ID")
    or os.getenv("RAZORPAY_KEYID")  # tolerance
    or ""
)

# Accept common secret names
RAZORPAY_SECRET = (
    os.getenv("RAZORPAY_SECRET")
    or os.getenv("RAZORPAY_KEY_SECRET")
    or os.getenv("RAZORPAY_SECRET_KEY")
    or ""
)

RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET") or ""
PUBLIC_CONFIG_URL       = os.getenv("PUBLIC_CONFIG_URL", "https://caio-backend.onrender.com/api/public-config")

# Optional: force currency if your Razorpay account is not enabled for international
PAYMENTS_FORCE_CURRENCY = (os.getenv("PAYMENTS_FORCE_CURRENCY") or "").upper().strip()

# Subscription knobs (price now comes from region; these define cadence)
SUB_INTERVAL        = os.getenv("CAIO_SUB_INTERVAL", "monthly")     # daily/weekly/monthly/yearly
SUB_INTERVAL_COUNT  = int(os.getenv("CAIO_SUB_INTERVAL_COUNT", "1"))
SUB_TOTAL_COUNT     = int(os.getenv("CAIO_SUB_TOTAL_COUNT", "12"))

# Optional pre-provisioned items
RAZORPAY_PLAN_ID          = (os.getenv("RAZORPAY_PLAN_ID") or "").strip()
RAZORPAY_SUBSCRIPTION_ID  = (os.getenv("RAZORPAY_SUBSCRIPTION_ID") or "").strip()

# ---------------------------------------------------------------------
# Razorpay client (lazy import so API can still boot without the SDK)
# ---------------------------------------------------------------------

_rzp_client: Optional[Any] = None

def get_client():
    global _rzp_client
    if _rzp_client:
        return _rzp_client
    if not (RAZORPAY_KEY_ID and RAZORPAY_SECRET):
        raise HTTPException(status_code=500, detail="Payments not configured (missing Razorpay keys).")
    try:
        import razorpay  # lazy import
    except Exception as e:
        logger.exception("Razorpay import failed")
        raise HTTPException(status_code=500, detail=f"Razorpay SDK not available: {e}")
    _rzp_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_SECRET))
    return _rzp_client

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

async def get_region_pricing(req: Request) -> tuple[str, int]:
    """
    Returns (currency, amount_minor) for Pro plan from PUBLIC_CONFIG.
    Respects PAYMENTS_FORCE_CURRENCY. Falls back to USD 49 / INR 499.
    """
    DEFAULTS = {"USD": 49, "INR": 499}
    force = req.query_params.get("force")  # e.g., ?force=US or ?force=IN
    try:
        async with httpx.AsyncClient(timeout=8.0) as x:
            r = await x.get(PUBLIC_CONFIG_URL, params={"force": force} if force else None)
            r.raise_for_status()
            cfg = r.json()
    except Exception:
        cfg = {}

    currency = (cfg.get("currency") or "INR").upper()
    if PAYMENTS_FORCE_CURRENCY:
        currency = PAYMENTS_FORCE_CURRENCY

    plans = cfg.get("plans") or {}
    pro   = plans.get("pro") or {}
    price_major = pro.get("price")

    if price_major is None:
        price_major = DEFAULTS.get(currency, 49)

    # Guard: if USD 499 accidentally appears, correct to 49
    try:
        pm = float(price_major)
    except Exception:
        pm = 49.0
    if currency == "USD" and pm >= 400:
        pm = 49.0

    amount_minor = int(round(pm * 100))
    return currency, amount_minor

def _ensure_keys():
    if not (RAZORPAY_KEY_ID and RAZORPAY_SECRET):
        raise HTTPException(status_code=500, detail="Razorpay keys missing on server")

def _create_or_get_plan(client: Any, currency: str, amount_minor: int) -> str:
    """
    Use RAZORPAY_PLAN_ID if present; otherwise create a plan with region pricing.
    """
    if RAZORPAY_PLAN_ID:
        return RAZORPAY_PLAN_ID
    plan = client.plan.create({
        "period": SUB_INTERVAL,
        "interval": SUB_INTERVAL_COUNT,
        "item": {
            "name": f"CAIO Pro ({amount_minor // 100} {currency}/{SUB_INTERVAL})",
            "amount": amount_minor,
            "currency": currency,
        },
    })
    return plan["id"]

# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

@router.get("/ping")
def ping():
    return {"ok": True, "mode": "razorpay"}

@router.get("/routes")
def list_routes(request: Request):
    out = []
    for r in request.app.routes:
        methods = sorted(list(getattr(r, "methods", [])))
        path = getattr(r, "path", "")
        if path.startswith("/api/"):
            out.append({"path": path, "methods": methods})
    return out

@router.get("/auth-check")
def auth_check():
    """
    Lightweight auth probe to confirm key/secret are valid in the current mode.
    """
    try:
        client = get_client()
        _ = client.order.all({"count": 1})
        return {"ok": True, "key_id": f"{RAZORPAY_KEY_ID[:10]}..."}
    except Exception as e:
        logger.exception("Auth check failed")
        raise HTTPException(status_code=500, detail=f"Auth failed: {e}")

# ---------------------------------------------------------------------
# Basic config + one-time order (optional)
# ---------------------------------------------------------------------

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
        order = client.order.create({
            "amount": amount_minor,
            "currency": currency,
            "receipt": f"pro-{current_user.email}",
            "payment_capture": 1,
            "notes": {"email": current_user.email, "plan": "pro"},
        })
        return {
            "order_id": order["id"],
            "amount": order["amount"],
            "currency": order["currency"],
            "key_id": RAZORPAY_KEY_ID,
        }
    except Exception as e:
        logger.exception("Create order error")
        raise HTTPException(status_code=500, detail=f"Create order failed: {e}")

# ---------------------------------------------------------------------
# Subscriptions
# ---------------------------------------------------------------------

@router.get("/subscription-config")
async def subscription_config(request: Request):
    _ensure_keys()
    currency, amount_minor = await get_region_pricing(request)
    return {
        "mode": "razorpay",
        "currency": currency,
        "amount_major": amount_minor // 100,
        "amount": amount_minor,
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
        currency, amount_minor = await get_region_pricing(request)
        client = get_client()
        plan_id = _create_or_get_plan(client, currency, amount_minor)
        sub = client.subscription.create({
            "plan_id": plan_id,
            "total_count": SUB_TOTAL_COUNT,
            "customer_notify": 1,
            "notes": {"email": current_user.email, "app": "CAIO"},
        })
        return {"subscription_id": sub["id"], "key_id": RAZORPAY_KEY_ID, "email": current_user.email}
    except Exception as e:
        logger.exception("Subscription create error")
        raise HTTPException(status_code=500, detail=f"Subscription create failed: {e}")

# GET alias so a POST 405 won’t break flows behind proxies
@router.get("/subscribe")
async def subscribe_get(request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return await subscribe(request, current_user, db)

# Use a subscription created in Razorpay Dashboard (smoke test path)
@router.get("/direct-subscription")
def direct_subscription(current_user: User = Depends(get_current_user)):
    _ensure_keys()
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not RAZORPAY_SUBSCRIPTION_ID:
        raise HTTPException(status_code=400, detail="RAZORPAY_SUBSCRIPTION_ID not set on server")
    return {"subscription_id": RAZORPAY_SUBSCRIPTION_ID, "key_id": RAZORPAY_KEY_ID, "email": current_user.email}

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
    except Exception as e:
        logger.exception("Cancel error")
        raise HTTPException(status_code=500, detail=f"Cancel failed: {e}")

    try:
        current_user.is_paid = False
        db.add(current_user); db.commit()
    except Exception as e:
        logger.error("DB update failed after cancel: %s", e)

    return {"ok": True, "message": "Subscription cancelled"}

# ---------------------------------------------------------------------
# Webhooks (order + subscription events)
# ---------------------------------------------------------------------

@router.post("/webhook")
async def webhook(
    request: Request,
    x_razorpay_signature: str = Header(None),
    db: Session = Depends(get_db),
):
    if not RAZORPAY_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Webhook secret not configured")
    try:
        body_bytes = await request.body()
        payload = body_bytes.decode("utf-8")

        # Verify HMAC SHA256
        digest = hmac.new(RAZORPAY_WEBHOOK_SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
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

        # If email missing, try fetching Razorpay customer (via payment/subscription)
        if not email:
            cust_id = pay_entity.get("customer_id") or sub_entity.get("customer_id")
            if cust_id:
                try:
                    client = get_client()
                    cust = client.customer.fetch(cust_id)
                    email = cust.get("email")
                except Exception as e:
                    logger.warning("Customer fetch failed: %s", e)

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
        logger.exception("Webhook processing error")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

# ---------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------

@router.get("/status")
async def status(current_user: User = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"email": current_user.email, "is_paid": bool(current_user.is_paid)}

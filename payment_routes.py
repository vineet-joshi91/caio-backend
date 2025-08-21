# payment_routes.py
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
import os, razorpay, httpx

from db import get_db, User
from auth import get_current_user

router = APIRouter()

RAZORPAY_KEY_ID     = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_SECRET     = os.getenv("RAZORPAY_SECRET")
WEBHOOK_SECRET      = os.getenv("RAZORPAY_WEBHOOK_SECRET")
PUBLIC_CONFIG_URL   = os.getenv("PUBLIC_CONFIG_URL", "https://caio-backend.onrender.com/api/public-config")

if not (RAZORPAY_KEY_ID and RAZORPAY_SECRET):
    raise RuntimeError("Razorpay env vars missing")

rz = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_SECRET))

async def get_pricing_for_request(req: Request) -> tuple[str, int]:
    """
    Pulls region-aware Pro price from your public-config so frontend + backend stay in sync.
    Returns (currency, amount_in_minor_units).
    """
    # forward ?force for testing parity (optional)
    force = req.query_params.get("force")
    params = {"force": force} if force else {}
    async with httpx.AsyncClient(timeout=7.0) as x:
        r = await x.get(PUBLIC_CONFIG_URL, params=params)
        r.raise_for_status()
        cfg = r.json()
    currency = cfg.get("currency", "INR")
    pro = (cfg.get("plans") or {}).get("pro") or {}
    price_major = pro.get("price", 1999)  # fallback
    if currency == "INR":
        amount_minor = int(price_major) * 100        # paise
    elif currency == "USD":
        amount_minor = int(price_major) * 100        # cents (if multi-currency enabled)
    else:
        # default to INR paise
        currency = "INR"
        amount_minor = int(price_major) * 100
    return currency, amount_minor

@router.post("/create-order")
async def create_order(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    currency, amount_minor = await get_pricing_for_request(request)

    # Create order in Razorpay
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

# payment_routes.py
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import httpx, os

from db import get_db, User
from auth import get_current_user

router = APIRouter()  # we'll mount with prefix="/api/payments" from main.py

RAZORPAY_KEY_ID    = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET= os.getenv("RAZORPAY_KEY_SECRET", "")
BACKEND_BASE       = os.getenv("BACKEND_BASE", "https://caio-backend.onrender.com")

def _minor_units(amount_in_major: int) -> int:
    # Razorpay needs paise — INR 1999 => 199900
    return int(amount_in_major) * 100

@router.get("/config")
async def payments_config():
    """
    For the frontend to load Razorpay key + amount without exposing the secret.
    """
    async with httpx.AsyncClient(timeout=8.0) as client:
        r = await client.get(f"{BACKEND_BASE}/api/public-config")
        cfg = r.json() if r.is_success else {}
    currency = (cfg.get("currency") or "INR").upper()
    plans = (cfg.get("plans") or {})
    pro = plans.get("pro") or {"price": 1999, "period": "month"}
    return {
        "key_id": RAZORPAY_KEY_ID,
        "currency": currency,
        "amount_major": pro.get("price", 1999),
        "period": pro.get("period", "month"),
        "mode": "test" if RAZORPAY_KEY_ID.startswith("rzp_test_") else "live"
    }

@router.post("/create-order")
async def create_order(
    req: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Creates a Razorpay order for the logged-in user.
    """
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        raise HTTPException(status_code=500, detail="Razorpay keys not configured")

    # pull amount/currency from your public-config
    async with httpx.AsyncClient(timeout=8.0) as client:
        r = await client.get(f"{BACKEND_BASE}/api/public-config")
        if not r.is_success:
            raise HTTPException(status_code=500, detail="Could not fetch pricing config")
        cfg = r.json()

    currency = (cfg.get("currency") or "INR").upper()
    plans = (cfg.get("plans") or {})
    pro = plans.get("pro") or {"price": 1999, "period": "month"}
    amount_minor = _minor_units(int(pro.get("price", 1999)))

    # Create Razorpay order
    auth = (RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)
    async with httpx.AsyncClient(timeout=15.0, auth=auth) as client:
        rp = await client.post(
            "https://api.razorpay.com/v1/orders",
            json={
                "amount": amount_minor,
                "currency": currency,
                "receipt": f"pro-{user.email}",
                "notes": {"email": user.email},
                "payment_capture": 1,
            },
        )

    if rp.status_code >= 400:
        try:
            detail = rp.json()
        except Exception:
            detail = {"detail": rp.text}
        raise HTTPException(status_code=rp.status_code, detail=detail)

    order = rp.json()
    return {
        "order_id": order.get("id"),
        "amount": order.get("amount"),
        "currency": order.get("currency"),
        "key_id": RAZORPAY_KEY_ID,
        "email": user.email,
        "mode": "test" if RAZORPAY_KEY_ID.startswith("rzp_test_") else "live"
    }

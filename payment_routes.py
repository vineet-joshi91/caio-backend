# payment_routes.py
import os
import hmac
import hashlib
import logging
from typing import Optional, Dict, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Body
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from db import get_db, User  # adjust if your import path differs
from auth import get_current_user  # bearer token dependency

log = logging.getLogger("uvicorn.error")
router = APIRouter()

# --- Environment ---
KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
KEY_SECRET = os.getenv("RAZORPAY_SECRET", "")
WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")
DEFAULT_CURRENCY = os.getenv("RAZORPAY_CURRENCY", "INR")
# ₹499.00 == 49900 paise
DEFAULT_AMOUNT_PAISE = int(os.getenv("RAZORPAY_DEFAULT_AMOUNT_PAISE", "49900"))

RAZORPAY_API = "https://api.razorpay.com/v1"


def _assert_keys():
    if not KEY_ID or not KEY_SECRET:
        raise HTTPException(status_code=500, detail="Razorpay keys not configured")


@router.get("/config")
def payments_config():
    """
    Simple probe so the frontend (or you) can confirm the router is live.
    """
    return {"key_id": KEY_ID, "currency": DEFAULT_CURRENCY, "default_amount_paise": DEFAULT_AMOUNT_PAISE}


@router.post("/create-order")
async def create_order(
    payload: Dict[str, Any] = Body(default={}),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Create a Razorpay order for the logged-in user.
    Body: { plan?: "pro", amount_paise?: int }
    """
    _assert_keys()
    plan = (payload.get("plan") or "pro").lower()
    amount_paise = int(payload.get("amount_paise") or DEFAULT_AMOUNT_PAISE)

    notes = {
        "plan": plan,
        "email": user.email,
    }
    order_req = {
        "amount": amount_paise,
        "currency": DEFAULT_CURRENCY,
        "receipt": f"rcpt_{user.email}",
        "payment_capture": 1,
        "notes": notes,
    }

    auth = (KEY_ID, KEY_SECRET)
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(f"{RAZORPAY_API}/orders", auth=auth, json=order_req)
    if r.status_code >= 300:
        log.error("Razorpay create order failed: %s %s", r.status_code, r.text)
        raise HTTPException(status_code=502, detail="Failed to create payment order")

    j = r.json()  # contains id, amount, currency, etc.
    order_id = j.get("id")
    if not order_id:
        log.error("Razorpay order response missing id: %s", j)
        raise HTTPException(status_code=502, detail="Invalid order response")

    # If you persist orders, insert here (order_id, user_id, plan, amount, status="created")
    # For MVP we rely on webhook 'notes.email' to map user.

    return {
        "order_id": order_id,
        "amount": j.get("amount"),
        "currency": j.get("currency"),
        "key_id": KEY_ID,
    }


def _verify_sig(body: bytes, signature: str) -> bool:
    mac = hmac.new(WEBHOOK_SECRET.encode("utf-8"), body, hashlib.sha256)
    expected = mac.hexdigest()
    return hmac.compare_digest(expected, signature)


def _mark_user_paid_by_email(db: Session, email: str) -> bool:
    u = db.query(User).filter(User.email == email.lower()).first()
    if not u:
        return False
    if not u.is_paid:
        u.is_paid = True
        db.add(u)
        db.commit()
    return True


@router.post("/webhook")
async def webhook(request: Request, db: Session = Depends(get_db)):
    """
    Razorpay webhook:
    - Verify X-Razorpay-Signature
    - On payment.captured / order.paid, set user.is_paid = True
    """
    if not WEBHOOK_SECRET:
        # If secret is missing, acknowledge but don't modify anything
        log.error("WEBHOOK_SECRET not configured")
        return JSONResponse({"ok": True})

    body = await request.body()
    sig = request.headers.get("X-Razorpay-Signature", "")
    if not sig or not _verify_sig(body, sig):
        log.warning("Invalid webhook signature")
        return JSONResponse({"ok": True}, status_code=200)  # ack to avoid retries

    try:
        payload = await request.json()
    except Exception:
        payload = {}

    event = (payload or {}).get("event") or ""
    email: Optional[str] = None

    # try to pull email from notes or payment entity
    try:
        notes = payload["payload"]["payment"]["entity"].get("notes") or {}
        email = (notes.get("email") or "").lower() or None
    except Exception:
        pass

    if not email:
        try:
            email = (payload["payload"]["order"]["entity"].get("notes") or {}).get("email")
        except Exception:
            pass

    if not email:
        try:
            email = payload["payload"]["payment"]["entity"].get("email")
        except Exception:
            pass

    if event in ("payment.captured", "order.paid") and email:
        ok = _mark_user_paid_by_email(db, email)
        log.info("Webhook %s -> %s paid=%s", event, email, ok)
        return JSONResponse({"ok": True})

    log.info("Webhook received: %s (no action)", event)
    return JSONResponse({"ok": True})

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 12:22:49 2025

@author: Vineet
"""

import os
import time
import base64
import hmac
import hashlib
import logging
from typing import Optional, Dict, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Body
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

# Your project modules
from db import get_db, User  # type: ignore
from auth import get_current_user  # type: ignore

router = APIRouter()

log = logging.getLogger("uvicorn.error")

# ---- Environment ----
KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
KEY_SECRET = os.getenv("RAZORPAY_SECRET", "")
WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")

# Default plan settings (you can change these)
DEFAULT_CURRENCY = os.getenv("RAZORPAY_CURRENCY", "INR")
# Amount in paise: ₹499.00 -> 49900
DEFAULT_AMOUNT_PAISE = int(os.getenv("RAZORPAY_DEFAULT_AMOUNT_PAISE", "49900"))

RAZORPAY_API = "https://api.razorpay.com/v1"


def _assert_configured() -> None:
    if not KEY_ID or not KEY_SECRET:
        raise HTTPException(status_code=500, detail="Payments not configured on server")


def _basic_auth_header() -> Dict[str, str]:
    token = base64.b64encode(f"{KEY_ID}:{KEY_SECRET}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


# ---- Public config for frontend (no secrets) ----
@router.get("/config")
def payments_config():
    """
    Frontend can call this to know whether payments are enabled and which key id to use.
    """
    if not KEY_ID:
        raise HTTPException(500, "Payments not configured")
    return {"key_id": KEY_ID, "currency": DEFAULT_CURRENCY}


# ---- Create an order for the logged-in user ----
@router.post("/create-order")
def create_order(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
    # Allow overriding amount from frontend if you want (defaults to Pro plan).
    amount_paise: Optional[int] = Body(None, embed=True),
    plan: Optional[str] = Body(None, embed=True),
):
    """
    Creates a Razorpay Order (one-time Pro purchase).
    - Associates the order with the user via notes.email
    - Returns order_id and meta for Checkout
    """
    _assert_configured()

    amt = amount_paise or DEFAULT_AMOUNT_PAISE
    if amt <= 0:
        raise HTTPException(400, "Invalid amount")

    receipt = f"caio-{current_user.email}-{int(time.time())}"
    payload = {
        "amount": amt,
        "currency": DEFAULT_CURRENCY,
        "receipt": receipt,
        "payment_capture": 1,  # auto-capture after auth
        "notes": {
            "email": current_user.email,
            "plan": plan or "pro",
        },
    }

    try:
        with httpx.Client(timeout=20.0) as client:
            res = client.post(f"{RAZORPAY_API}/orders", json=payload, headers=_basic_auth_header())
        if res.status_code >= 300:
            log.error("Razorpay order error %s: %s", res.status_code, res.text)
            raise HTTPException(res.status_code, "Failed to create order")
        data = res.json()
    except HTTPException:
        raise
    except Exception as e:
        log.exception("create_order error: %s", e)
        raise HTTPException(502, "Payment provider unavailable")

    # Minimal payload for your frontend Checkout
    return {
        "order_id": data.get("id"),
        "amount": data.get("amount"),
        "currency": data.get("currency"),
        "key_id": KEY_ID,
    }


def _verify_webhook_signature(raw_body: bytes, signature: str) -> bool:
    """
    Razorpay sends X-Razorpay-Signature (hex). We compute HMAC SHA256 over raw body.
    """
    if not WEBHOOK_SECRET or not signature:
        return False
    computed = hmac.new(WEBHOOK_SECRET.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, signature)


def _mark_user_paid_by_email(db: Session, email: Optional[str]) -> bool:
    if not email:
        return False
    user = db.query(User).filter(User.email == email).first()  # type: ignore
    if not user:
        return False
    if getattr(user, "is_paid", False):
        return True  # already paid (idempotent)
    try:
        user.is_paid = True  # type: ignore
        db.add(user)         # type: ignore
        db.commit()
        return True
    except Exception as e:
        log.exception("DB update failed while marking paid: %s", e)
        db.rollback()
        return False


# ---- Webhook endpoint (Razorpay -> CAIO) ----
@router.post("/webhook")
async def webhook(request: Request, db: Session = Depends(get_db)):
    """
    Configure in Razorpay Dashboard:
      URL: https://<your-backend>/api/payments/webhook
      Secret: <RAZORPAY_WEBHOOK_SECRET>
      Events: order.paid, payment.captured
    """
    raw = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")

    if not _verify_webhook_signature(raw, signature):
        log.warning("Invalid Razorpay webhook signature")
        raise HTTPException(400, "Invalid signature")

    try:
        event = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    etype = event.get("event", "")
    payload = event.get("payload", {}) or {}

    # Try to find user's email (we put it into notes at order creation)
    email: Optional[str] = None

    # order path
    try:
        order_entity = payload.get("order", {}).get("entity", {}) or {}
        email = (order_entity.get("notes") or {}).get("email") or email
    except Exception:
        pass

    # payment path (some webhooks only include payment)
    try:
        payment_entity = payload.get("payment", {}).get("entity", {}) or {}
        email = (payment_entity.get("notes") or {}).get("email") or email
        # Fallback to payment.email if notes were stripped
        email = payment_entity.get("email") or email
    except Exception:
        pass

    if etype in ("order.paid", "payment.captured") and email:
        ok = _mark_user_paid_by_email(db, email)
        log.info("Webhook %s for %s -> paid=%s", etype, email, ok)
        # Always 200 so Razorpay doesn't retry forever
        return JSONResponse({"ok": True})

    # Other events — acknowledge with 200
    log.info("Webhook received: %s (no action)", etype)
    return JSONResponse({"ok": True})

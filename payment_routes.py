# -*- coding: utf-8 -*-
"""
Payments routes for CAIO — Razorpay create-order + webhook.
Namespace: /api/payments/*
"""

import os
import time
import base64
import hmac
import hashlib
import logging
from typing import Optional, Dict

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Body
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text

# Project modules
from db import get_db, User  # type: ignore
from auth import get_current_user  # type: ignore

log = logging.getLogger("uvicorn.error")

# -------------------- Env --------------------
KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
KEY_SECRET = os.getenv("RAZORPAY_SECRET", "")
WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")
DEFAULT_CURRENCY = os.getenv("RAZORPAY_CURRENCY", "INR")

# Plans in paise (₹ -> x100)
PLAN_INR = {
    "pro": int(os.getenv("RAZORPAY_PLAN_PRO_PAISE", "49900")),         # ₹499.00
    "premium": int(os.getenv("RAZORPAY_PLAN_PREMIUM_PAISE", "149900")), # ₹1499.00
}

RAZORPAY_API = "https://api.razorpay.com/v1"


def _assert_configured() -> None:
    if not KEY_ID or not KEY_SECRET:
        raise HTTPException(status_code=500, detail="Payments not configured on server")


def _basic_auth_header() -> Dict[str, str]:
    tok = base64.b64encode(f"{KEY_ID}:{KEY_SECRET}".encode()).decode()
    return {"Authorization": f"Basic {tok}"}


router = APIRouter(prefix="/api/payments", tags=["payments"])

# -------------------- Public (no secrets) --------------------
@router.get("/config")
def payments_config():
    """
    Frontend fetches this to initialize Razorpay Checkout.
    """
    _assert_configured()
    return {"key_id": KEY_ID, "currency": DEFAULT_CURRENCY, "plans": list(PLAN_INR.keys())}


# -------------------- Create Order --------------------
@router.post("/create-order")
def create_order(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
    plan: Optional[str] = Body(None, embed=True),
    amount_paise: Optional[int] = Body(None, embed=True),
):
    """
    Creates a Razorpay Order for the logged-in user.
    - Associates the order with user via 'notes.email' (and 'receipt')
    - Defaults to the chosen plan (pro/premium) if amount not provided
    """
    _assert_configured()

    plan = (plan or "pro").lower().strip()
    amt = amount_paise if (amount_paise and amount_paise > 0) else PLAN_INR.get(plan)
    if not amt or amt <= 0:
        raise HTTPException(400, "Invalid amount/plan")

    # Unique receipt helps us tie payment->user on the webhook, even if notes go missing
    receipt = f"caio:{current_user.email}:{plan}:{int(time.time())}"

    payload = {
        "amount": amt,
        "currency": DEFAULT_CURRENCY,
        "receipt": receipt,
        "payment_capture": 1,  # auto-capture after authorization
        "notes": {
            "email": current_user.email,
            "plan": plan,
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

    # You can also persist order->user mapping server-side here if you want
    # db.execute(text("INSERT ..."))  # optional

    return {
        "order_id": data.get("id"),
        "amount": data.get("amount"),
        "currency": data.get("currency"),
        "key_id": KEY_ID,
        "plan": plan,
        "email": current_user.email,
    }


# -------------------- Helpers --------------------
def _verify_webhook_signature(raw_body: bytes, signature: str) -> bool:
    """
    Razorpay sends X-Razorpay-Signature (hex). Compute HMAC SHA256 over raw body.
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
        return True  # idempotent
    try:
        user.is_paid = True  # type: ignore
        db.add(user)         # type: ignore
        db.commit()
        return True
    except Exception as e:
        log.exception("DB update failed while marking paid: %s", e)
        db.rollback()
        return False


def _extract_email_from_payload(payload: dict) -> Optional[str]:
    """
    Try multiple paths to find the purchaser's email.
    1) payment.entity.notes.email
    2) order.entity.notes.email
    3) payment.entity.email (fallback if notes missing)
    4) order.entity.receipt -> 'caio:<email>:<plan>:<ts>'
    """
    try:
        pay = (payload.get("payment") or {}).get("entity") or {}
        email = (pay.get("notes") or {}).get("email") or pay.get("email")
        if email:
            return email.strip().lower()
    except Exception:
        pass

    try:
        order = (payload.get("order") or {}).get("entity") or {}
        email = (order.get("notes") or {}).get("email")
        if email:
            return email.strip().lower()
        receipt = (order.get("receipt") or "").strip()
        if receipt.startswith("caio:"):
            parts = receipt.split(":")
            if len(parts) >= 3:
                return parts[1].strip().lower()
    except Exception:
        pass

    return None


# -------------------- Webhook (Razorpay -> CAIO) --------------------
async def _webhook_handler(request: Request, db: Session) -> JSONResponse:
    """
    Configure in Razorpay Dashboard:
      URL   : https://<backend>/api/payments/webhook   (preferred)
              or https://<backend>/razorpay-webhook    (legacy alias)
      Secret: RAZORPAY_WEBHOOK_SECRET
      Events: payment.captured, payment.failed          (+ order.paid if desired)
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
    email = _extract_email_from_payload(payload)

    if etype in ("payment.captured", "order.paid") and email:
        ok = _mark_user_paid_by_email(db, email)
        log.info("Webhook %s for %s -> paid=%s", etype, email, ok)
        return JSONResponse({"ok": True})

    if etype == "payment.failed" and email:
        # Optional: record failure
        try:
            db.execute(text(
                "INSERT INTO payments_log (email, status) VALUES (:email, 'failed')"
            ), {"email": email})
            db.commit()
        except Exception:
            db.rollback()
        log.info("Webhook payment.failed for %s", email)
        return JSONResponse({"ok": True})

    # Acknowledge all other events so Razorpay doesn't retry forever
    log.info("Webhook received: %s (no action)", etype)
    return JSONResponse({"ok": True})


@router.post("/webhook")
async def webhook(request: Request, db: Session = Depends(get_db)):
    return await _webhook_handler(request, db)


# Legacy alias to match dashboards already pointing here:
from fastapi import APIRouter as _AR  # avoid linter warning
_alias = _AR()
@_alias.post("/razorpay-webhook")
async def webhook_alias(request: Request, db: Session = Depends(get_db)):
    return await _webhook_handler(request, db)

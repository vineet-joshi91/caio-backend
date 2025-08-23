# payment_routes.py
from __future__ import annotations

import os, hmac, hashlib, json
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
import httpx

from db import get_db, User  # User must have: subscription_id, plan_status, is_paid
from auth import get_current_user

router = APIRouter(prefix="/api/payments", tags=["payments"])

RAZORPAY_KEY_ID        = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET    = os.getenv("RAZORPAY_KEY_SECRET", "") or os.getenv("RAZORPAY_SECRET", "")
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")
BACKEND_BASE           = os.getenv("BACKEND_BASE", "https://caio-backend.onrender.com")

def _require_keys():
    if not (RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET):
        raise HTTPException(500, "Razorpay keys not configured")

@router.get("/subscription-config")
async def subscription_config():
    """Public info for the Pro subscription (currency, amount, key_id, mode)."""
    async with httpx.AsyncClient(timeout=8.0) as c:
        r = await c.get(f"{BACKEND_BASE}/api/public-config")
    cfg = r.json() if r.is_success else {}
    currency = (cfg.get("currency") or "INR").upper()
    pro = (cfg.get("plans") or {}).get("pro") or {"price": 1999, "period": "month"}
    return {
        "key_id": RAZORPAY_KEY_ID,
        "currency": currency,
        "amount_major": int(pro["price"]),
        "interval": pro.get("period", "month"),
        "mode": "test" if RAZORPAY_KEY_ID.startswith("rzp_test_") else "live",
        "engine": "subscription",
    }

@router.post("/subscribe")
async def subscribe(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Create a Razorpay Subscription for the current user.
    Stores subscription_id and initial status on the user.
    Auto-debits are handled by Razorpay after mandate authentication.
    """
    _require_keys()

    # 1) Price & interval from public-config
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.get(f"{BACKEND_BASE}/api/public-config")
    if not r.is_success:
        raise HTTPException(500, "Could not fetch pricing config")
    cfg = r.json()
    currency = (cfg.get("currency") or "INR").upper()
    amount_major = int((cfg.get("plans") or {}).get("pro", {}).get("price", 1999))
    interval = (cfg.get("plans") or {}).get("pro", {}).get("period", "month")

    auth = (RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)

    # 2) Ensure a Plan exists that matches price/currency/interval
    async with httpx.AsyncClient(timeout=15.0, auth=auth) as c:
        plans = await c.get(
            "https://api.razorpay.com/v1/plans",
            params={"item[name]":"CAIO Pro","period":interval,"item[amount]":amount_major*100,"item[currency]":currency}
        )
        plan_id = None
        if plans.is_success and plans.json().get("items"):
            plan_id = plans.json()["items"][0]["id"]
        if not plan_id:
            pr = await c.post("https://api.razorpay.com/v1/plans", json={
                "period": interval, "interval": 1,
                "item": {"name":"CAIO Pro","amount":amount_major*100,"currency":currency,"description":"Monthly subscription for CAIO Pro"}
            })
            if pr.status_code >= 400:
                raise HTTPException(pr.status_code, pr.text)
            plan_id = pr.json()["id"]

        # 3) Create the Subscription (open‑ended: total_count=0)
        sr = await c.post("https://api.razorpay.com/v1/subscriptions", json={
            "plan_id": plan_id,
            "customer_notify": 1,  # Razorpay can send auth link if needed
            "total_count": 0,
            "notes": {"email": user.email},
        })
        if sr.status_code >= 400:
            raise HTTPException(sr.status_code, sr.text)
        sub = sr.json()

    # 4) Persist on user
    user.subscription_id = sub["id"]
    user.plan_status = sub.get("status", "created")
    db.add(user); db.commit()

    return {"subscription_id": sub["id"], "status": user.plan_status, "key_id": RAZORPAY_KEY_ID}

@router.post("/cancel")
async def cancel_subscription(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Cancel the active subscription immediately."""
    _require_keys()
    if not user.subscription_id:
        raise HTTPException(400, "No active subscription")

    auth = (RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)
    async with httpx.AsyncClient(timeout=15.0, auth=auth) as c:
        r = await c.post(f"https://api.razorpay.com/v1/subscriptions/{user.subscription_id}/cancel")
        if r.status_code >= 400:
            raise HTTPException(r.status_code, r.text)

    user.plan_status = "cancelled"
    user.is_paid = False
    db.add(user); db.commit()
    return {"ok": True, "status": "cancelled"}

@router.post("/webhook")
async def webhook(request: Request, db: Session = Depends(get_db)):
    """Razorpay webhook: keeps user.is_paid and plan_status in sync."""
    if not RAZORPAY_WEBHOOK_SECRET:
        # Allow running without a secret in early dev
        return {"ok": True, "skipped": "no webhook secret set"}

    raw = await request.body()
    given = request.headers.get("x-razorpay-signature") or ""
    expect = hmac.new(RAZORPAY_WEBHOOK_SECRET.encode(), raw, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expect, given):
        raise HTTPException(400, "Invalid signature")

    evt = json.loads(raw.decode("utf-8"))
    etype = evt.get("event", "")
    payload = evt.get("payload", {})

    def _user_from_sub() -> User | None:
        sub_id = (payload.get("subscription") or {}).get("entity", {}).get("id")
        if not sub_id:
            return None
        return db.query(User).filter(User.subscription_id == sub_id).first()

    if etype in ("subscription.activated", "subscription.charged"):
        u = _user_from_sub()
        if u:
            u.plan_status = "active"
            u.is_paid = True
            db.add(u); db.commit()
    elif etype in ("subscription.paused", "subscription.halted", "subscription.cancelled"):
        u = _user_from_sub()
        if u:
            u.plan_status = "cancelled"
            u.is_paid = False
            db.add(u); db.commit()

    return {"ok": True}

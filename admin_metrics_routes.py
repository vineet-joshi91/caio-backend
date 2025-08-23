# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 16:27:04 2025

@author: Vineet
"""

# admin_metrics_routes.py
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import or_
import httpx
from db import get_db, User
from auth import get_current_user

router = APIRouter(prefix="/api/admin", tags=["admin-metrics"])

BACKEND_BASE = "https://caio-backend.onrender.com"

@router.get("/metrics")
async def metrics(db: Session = Depends(get_db), user=Depends(get_current_user)):
    if not user or not getattr(user, "is_admin", False):
        raise HTTPException(403, "Admin only")

    # Active paid users
    active_total = db.query(User).filter(User.is_paid.is_(True)).count()

    # Optional currency-aware counts if you added billing_currency
    active_inr = db.query(User).filter(User.is_paid.is_(True), getattr(User, "billing_currency", None) == "INR").count() if hasattr(User, "billing_currency") else None
    active_usd = db.query(User).filter(User.is_paid.is_(True), getattr(User, "billing_currency", None) == "USD").count() if hasattr(User, "billing_currency") else None

    # Cancellations in last 7 days (uses updated_at if present, else created_at)
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=7)
    updated = getattr(User, "updated_at", None)
    ts_col = updated if updated is not None else User.created_at
    cancelled_7d = db.query(User).filter(
        getattr(User, "plan_status", None) == "cancelled",
        ts_col >= cutoff
    ).count() if hasattr(User, "plan_status") else 0

    # Price (so we can estimate MRR by currency when possible)
    async with httpx.AsyncClient(timeout=8.0) as c:
        r = await c.get(f"{BACKEND_BASE}/api/public-config")
    cfg = r.json() if r.is_success else {}
    price = (cfg.get("plans") or {}).get("pro", {}).get("price", 0)
    currency = (cfg.get("currency") or "INR").upper()

    # MRR estimate
    # If we have currency per user, return split; else a simple single-currency estimate
    mrr = {}
    if active_inr is not None and active_usd is not None:
        mrr["INR"] = (active_inr or 0) * (1999 if currency == "INR" else 1999)  # hardcode your INR price
        mrr["USD"] = (active_usd or 0) * (49)                                   # hardcode your USD price
    else:
        mrr[currency] = active_total * int(price or 0)

    return {
        "active_total": active_total,
        "active_inr": active_inr,
        "active_usd": active_usd,
        "cancelled_7d": cancelled_7d,
        "mrr": mrr,
    }

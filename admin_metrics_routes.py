# -*- coding: utf-8 -*-
"""
Admin metrics (timeseries + totals) — Neon/SQLite safe, no external self-calls.
"""
from datetime import datetime, timedelta, timezone
from typing import Dict, Any
import os

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, case, cast, Date

from db import get_db, User, UsageLog
from auth import get_current_user

router = APIRouter(prefix="/api/admin", tags=["admin-metrics"])

# ---- Pricing knobs (env-driven; no HTTP self-call) ----------------------------
PRO_PRICE_INR       = int(os.getenv("PRO_PRICE_INR", "1999"))
PRO_PRICE_USD       = int(os.getenv("PRO_PRICE_USD", "49"))
PRO_PLUS_PRICE_INR  = int(os.getenv("PRO_PLUS_PRICE_INR", "2999"))
PRO_PLUS_PRICE_USD  = int(os.getenv("PRO_PLUS_PRICE_USD", "59"))
PREMIUM_PRICE_INR   = int(os.getenv("PREMIUM_PRICE_INR", "7999"))
PREMIUM_PRICE_USD   = int(os.getenv("PREMIUM_PRICE_USD", "99"))

DEFAULT_CURRENCY    = os.getenv("DEFAULT_ADMIN_CURRENCY", "INR").upper()  # Fallback

# ---- Helpers ------------------------------------------------------------------
def _ensure_admin(u: User) -> None:
    if not u or not bool(getattr(u, "is_admin", False)):
        raise HTTPException(status_code=403, detail="Admin only")

def _infer_tier_expr():
    """
    SQL expression for tier name.
    If you later add User.plan_tier (e.g., "pro_plus", "premium"), this will surface it.
    Fallback order: admin > (plan_tier if present) > (is_paid -> "pro") > "demo"
    """
    # SQLAlchemy CASE over columns; guard missing attributes
    has_plan_tier = hasattr(User, "plan_tier")
    return case(
        (User.is_admin == True, "admin"),
        (getattr(User, "plan_tier", None).isnot(None) if has_plan_tier else False, getattr(User, "plan_tier", "pro")),
        (User.is_paid == True, "pro"),
        else_="demo",
    )

def _today_utc():
    now = datetime.now(timezone.utc)
    return now.date()

# ---- API ----------------------------------------------------------------------
@router.get("/metrics")
def metrics(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    _ensure_admin(current_user)

    # Totals: active paid users (any paid tier)
    active_total = db.query(User).filter(User.is_paid.is_(True)).count()

    # Optional currency split (if you’ve added billing_currency on User)
    active_inr = active_usd = None
    if hasattr(User, "billing_currency"):
        active_inr = db.query(User).filter(User.is_paid.is_(True), User.billing_currency == "INR").count()
        active_usd = db.query(User).filter(User.is_paid.is_(True), User.billing_currency == "USD").count()

    # Recent cancellations (7d) if plan_status exists; else 0
    cancelled_7d = 0
    if hasattr(User, "plan_status"):
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=7)
        ts_col = getattr(User, "updated_at", None) or getattr(User, "created_at")
        cancelled_7d = db.query(User).filter(User.plan_status == "cancelled", ts_col >= cutoff).count()

    # --- Timeseries from UsageLog (authoritative) --------------------------------
    # Group by UTC date, endpoint, and tier (derived)
    day = cast(UsageLog.timestamp, Date)
    tier_expr = _infer_tier_expr()

    series_rows = (
        db.query(
            day.label("date"),
            UsageLog.endpoint.label("endpoint"),
            tier_expr.label("tier"),
            func.count(UsageLog.id).label("count"),
        )
        .join(User, User.id == UsageLog.user_id)
        .group_by("date", "endpoint", "tier")
        .order_by("date")
        .all()
    )

    series = [
        {
            "date": str(r.date),                # "YYYY-MM-DD" (UTC)
            "endpoint": r.endpoint or "",
            "tier": r.tier,
            "count": int(r.count or 0),
        }
        for r in series_rows
    ]

    # High-level totals
    today_str = str(_today_utc())
    totals_today = sum(x["count"] for x in series if x["date"] == today_str)
    totals_all = int(db.query(func.count(UsageLog.id)).scalar() or 0)

    # --- MRR estimate (split if we know currencies; else default currency) -------
    # If you store plan_tier, you can refine this section easily.
    mrr: Dict[str, int] = {}

    def _add(d: Dict[str, int], k: str, v: int) -> None:
        d[k] = d.get(k, 0) + v

    if active_inr is not None and active_usd is not None:
        # Basic split (Pro baseline); adjust if you later store per-user tier
        _add(mrr, "INR", (active_inr or 0) * PRO_PRICE_INR)
        _add(mrr, "USD", (active_usd or 0) * PRO_PRICE_USD)
    else:
        # Single-currency estimate if we don’t know user currencies
        cur = DEFAULT_CURRENCY
        _add(mrr, cur, active_total * (PRO_PRICE_INR if cur == "INR" else PRO_PRICE_USD))

    return {
        "totals": {
            "today": totals_today,
            "all_time": totals_all,
            "active_paid": active_total,
            "active_inr": active_inr,
            "active_usd": active_usd,
            "cancelled_7d": cancelled_7d,
        },
        "series": series,  # [{date, endpoint, tier, count}]
        "notes": "UTC grouping by day/endpoint/tier; prices from env; no external self-call.",
        "mrr": mrr,
    }

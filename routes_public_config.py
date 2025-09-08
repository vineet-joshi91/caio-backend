# -*- coding: utf-8 -*-
"""
Admin metrics API for CAIO
- GET /api/admin/metrics
- No HTTP self-calls (prices come from ENV)
- Stable timeseries: UTC day x endpoint x tier
- Back/forward compatible with optional columns on User & UsageLog
"""
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
import os

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, case, cast, Date

from db import get_db, User, UsageLog
from auth import get_current_user

router = APIRouter(prefix="/api/admin", tags=["admin-metrics"])

# ---- Env-driven pricing -------------------------------------------------------
PRO_PRICE_INR      = int(os.getenv("PRO_PRICE_INR", "1999"))
PRO_PRICE_USD      = int(os.getenv("PRO_PRICE_USD", "25"))
PRO_PLUS_PRICE_INR = int(os.getenv("PRO_PLUS_PRICE_INR", "3999"))
PRO_PLUS_PRICE_USD = int(os.getenv("PRO_PLUS_PRICE_USD", "49"))
PREMIUM_PRICE_INR  = int(os.getenv("PREMIUM_PRICE_INR", "7999"))
PREMIUM_PRICE_USD  = int(os.getenv("PREMIUM_PRICE_USD", "99"))

DEFAULT_ADMIN_CURRENCY = os.getenv("DEFAULT_ADMIN_CURRENCY", "INR").upper()

# ---- Helpers ------------------------------------------------------------------
def _ensure_admin(u: User) -> None:
    if not u or not bool(getattr(u, "is_admin", False)):
        raise HTTPException(status_code=403, detail="Admin only")

def _tier_expr():
    """
    Returns a SQLAlchemy CASE expression that yields a tier label.
    Priority: admin > plan_tier (if present) > is_paid->"pro" > "demo"
    """
    has_plan_tier = hasattr(User, "plan_tier")
    return case(
        (User.is_admin == True, "admin"),
        (
            (getattr(User, "plan_tier", None).isnot(None) if has_plan_tier else False),
            getattr(User, "plan_tier", "pro"),
        ),
        (User.is_paid == True, "pro"),
        else_="demo",
    )

def _today_utc_str() -> str:
    return str(datetime.now(timezone.utc).date())

# ---- API ----------------------------------------------------------------------
@router.get("/metrics")
def metrics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Returns:
      {
        "totals": {...},
        "series": [{ "date": "YYYY-MM-DD", "endpoint": "/api/...", "tier": "pro", "count": 123 }, ...],
        "mrr": {"INR": 123456, "USD": 7890},
        "notes": "..."
      }
    """
    _ensure_admin(current_user)

    # -------- Account totals --------
    active_total = db.query(User).filter(User.is_paid.is_(True)).count()

    # Optional currency split if you store billing_currency on User
    active_inr: Optional[int] = None
    active_usd: Optional[int] = None
    if hasattr(User, "billing_currency"):
        active_inr = db.query(User).filter(
            User.is_paid.is_(True), User.billing_currency == "INR"
        ).count()
        active_usd = db.query(User).filter(
            User.is_paid.is_(True), User.billing_currency == "USD"
        ).count()

    # Optional recent cancellations if you track plan_status + timestamps
    cancelled_7d = 0
    if hasattr(User, "plan_status"):
        # prefer updated_at if present, else created_at
        ts_col = getattr(User, "updated_at", None) or getattr(User, "created_at")
        if ts_col is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            cancelled_7d = db.query(User).filter(
                User.plan_status == "cancelled", ts_col >= cutoff
            ).count()

    # -------- Usage timeseries: UTC day x endpoint x tier --------
    # Requires UsageLog(timestamp, endpoint, user_id[, status])
    day = cast(UsageLog.timestamp, Date)
    series_rows = (
        db.query(
            day.label("date"),
            UsageLog.endpoint.label("endpoint"),
            _tier_expr().label("tier"),
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

    # Totals from logs
    today = _today_utc_str()
    totals_today = sum(x["count"] for x in series if x["date"] == today)
    totals_all = int(db.query(func.count(UsageLog.id)).scalar() or 0)

    # Optional: free cap hits today (if you log status on UsageLog)
    cap_hits_today = 0
    if hasattr(UsageLog, "status"):
        cap_hits_today = (
            db.query(func.count(UsageLog.id))
            .filter(
                cast(UsageLog.timestamp, Date) == today,
                UsageLog.status.in_(["429", "rate_limited"]),
            )
            .scalar()
            or 0
        )

    # -------- MRR (basic estimate) --------
    # If you later store plan_tier per user, compute finer-grained MRR here.
    mrr: Dict[str, int] = {}

    def _add(d: Dict[str, int], k: str, v: int) -> None:
        d[k] = d.get(k, 0) + v

    if active_inr is not None and active_usd is not None:
        # Baseline uses Pro price; refine once plan_tier is recorded per-user
        _add(mrr, "INR", (active_inr or 0) * PRO_PRICE_INR)
        _add(mrr, "USD", (active_usd or 0) * PRO_PRICE_USD)
    else:
        # Single bucket fallback if user currency unknown
        cur = DEFAULT_ADMIN_CURRENCY if DEFAULT_ADMIN_CURRENCY in ("INR", "USD") else "INR"
        base = PRO_PRICE_INR if cur == "INR" else PRO_PRICE_USD
        _add(mrr, cur, active_total * base)

    return {
        "totals": {
            "today": totals_today,
            "all_time": totals_all,
            "active_paid": active_total,
            "active_inr": active_inr,
            "active_usd": active_usd,
            "cancelled_7d": cancelled_7d,
            "free_cap_hits_today": cap_hits_today,
        },
        "series": series,  # [{date, endpoint, tier, count}]
        "mrr": mrr,
        "notes": "UTC grouping by day/endpoint/tier; prices from ENV; no HTTP self-call.",
    }

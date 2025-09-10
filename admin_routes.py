# admin_routes.py
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from sqlalchemy import func
import os

from db import get_db, User  # adjust if your models live elsewhere
# ChatSession is used for per-user session counts
try:
    from db import ChatSession  # noqa
except Exception:
    ChatSession = None  # backend will still work without session metrics

from auth import get_current_user  # returns DB User from JWT

router = APIRouter()

ADMIN_EMAIL = (os.getenv("ADMIN_EMAIL") or "").strip().lower()

# ------------------------- helpers ------------------------- #
def _assert_admin(current_user: User):
    # treat configured ADMIN_EMAIL as admin in addition to DB flag
    is_admin_email = ((current_user.email or "").lower() == ADMIN_EMAIL) if current_user and current_user.email else False
    if not (getattr(current_user, "is_admin", False) or is_admin_email):
        raise HTTPException(status_code=403, detail="Admin privileges required")

def _tier_of(u: User) -> str:
    # Simple tiering until you split pro/pro_plus/premium flags in DB
    # "admin" is folded under Premium on the UI
    if getattr(u, "is_admin", False):  # Admin → Premium card on UI
        return "admin"
    if getattr(u, "is_paid", False):
        # If you later add a dedicated column for pro_plus, map it here
        return "pro"
    return "demo"

# ------------------------- payloads ------------------------- #
class SetPaidRequest(BaseModel):
    email: EmailStr
    paid: bool

# ------------------------- routes ------------------------- #
@router.post("/set-paid")
def admin_set_paid(
    payload: SetPaidRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Admin-only: mark a user as paid/unpaid.
    Body: { "email": "...", "paid": true|false }
    """
    _assert_admin(current_user)
    target = db.query(User).filter(User.email == str(payload.email)).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    target.is_paid = bool(payload.paid)
    db.commit()
    db.refresh(target)
    return {"email": target.email, "is_paid": target.is_paid}

@router.get("/users/summary")
def users_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    KPI tiles for Admin page
    {
      "total_users": N,
      "demo": N,
      "pro": N,
      "pro_plus": N,   # currently always 0 unless you add DB flag
      "premium": N     # Admins are counted under Premium
    }
    """
    _assert_admin(current_user)

    total = db.query(User).count()
    demo = pro = premium = pro_plus = 0

    for u in db.query(User).all():
        t = _tier_of(u)
        if t == "admin":
            premium += 1
        elif t == "pro":
            pro += 1
        elif t == "demo":
            demo += 1
        else:
            demo += 1

    return {
        "total_users": total,
        "demo": demo,
        "pro": pro,
        "pro_plus": pro_plus,
        "premium": premium,
    }

@router.get("/users/roster")
def users_roster(
    q: str = Query("", alias="q"),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Paged table for Admin page
    Returns:
    {
      "page": 1, "page_size": 25, "total": N,
      "items": [
        { "email": "...", "tier": "demo|pro|pro_plus|premium|admin",
          "created_at": "...", "last_seen": null,
          "total_sessions": 0, "spend_usd": 0.0 }
      ]
    }
    """
    _assert_admin(current_user)

    query = db.query(User)
    term = q.strip()
    if term:
        query = query.filter(User.email.ilike(f"%{term}%"))

    total = query.count()
    rows = (
        query.order_by(User.id.asc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    # Per-user chat session counts (if ChatSession exists)
    sess_counts = {}
    if ChatSession and rows:
        ids = [u.id for u in rows]
        counts = (
            db.query(ChatSession.user_id, func.count(ChatSession.id))
              .filter(ChatSession.user_id.in_(ids))
              .group_by(ChatSession.user_id)
              .all()
        )
        sess_counts = {uid: int(cnt) for uid, cnt in counts}

    items = []
    for u in rows:
        created = getattr(u, "created_at", None)
        created_iso = created.isoformat() + "Z" if created else None
        items.append({
            "email": u.email,
            "tier": _tier_of(u),
            "created_at": created_iso,
            "last_seen": None,            # wire this if you track heartbeats
            "total_sessions": sess_counts.get(u.id, 0),
            "spend_usd": 0.0,             # wire billing data when available
        })

    return {"page": page, "page_size": page_size, "total": total, "items": items}

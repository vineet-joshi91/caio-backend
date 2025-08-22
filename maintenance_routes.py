# maintenance_routes.py

import os
from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

from db import get_db, User
from auth import get_current_user  # must return the user, with .is_admin bool

router = APIRouter(prefix="/api/admin/maintenance", tags=["maintenance"])

ADMIN_TOKEN = os.getenv("ADMIN_MAINTENANCE_TOKEN", "")

SQL_ALTERS = [
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS subscription_id TEXT;",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS plan_status TEXT;",
    "ALTER TABLE users ALTER COLUMN is_admin SET DEFAULT false;",
    "ALTER TABLE users ALTER COLUMN is_active SET DEFAULT true;",
    "ALTER TABLE users ALTER COLUMN is_paid  SET DEFAULT false;",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP;",
]

@router.post("/upgrade-db")
def upgrade_db(
    db: Session = Depends(get_db),
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    user: User = Depends(get_current_user),
):
    if not user or not getattr(user, "is_admin", False):
        raise HTTPException(status_code=403, detail="Admin only")
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Maintenance token invalid")

    for stmt in SQL_ALTERS:
        db.execute(text(stmt))
    db.commit()
    return {"ok": True, "applied": SQL_ALTERS}

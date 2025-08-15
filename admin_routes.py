# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 17:02:45 2025

@author: Vineet
"""

# admin_routes.py
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
import os

# Your project imports
from db import get_db, User  # if User is defined in db.py
# If your User model lives elsewhere, change the import accordingly:
# from models import User

from auth import get_current_user  # must return the current DB user from JWT

router = APIRouter()

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "").strip().lower()


class SetPaidRequest(BaseModel):
    email: EmailStr
    paid: bool


def _assert_admin(current_user: User):
    # treat configured ADMIN_EMAIL as admin too, even if flag not set
    is_admin_email = (current_user.email or "").lower() == ADMIN_EMAIL
    if not (getattr(current_user, "is_admin", False) or is_admin_email):
        raise HTTPException(status_code=403, detail="Admin privileges required")


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

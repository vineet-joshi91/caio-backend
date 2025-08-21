# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 13:56:05 2025

@author: Vineet
"""

# signup_routes.py
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from db import get_db, User
from auth import create_access_token, get_password_hash
from fastapi import Depends
from datetime import datetime

router = APIRouter(prefix="/api", tags=["auth"])

class SignupIn(BaseModel):
  name: str
  organisation: str | None = None
  email: EmailStr
  password: str

@router.post("/signup")
def signup(payload: SignupIn, db: Session = Depends(get_db)):
  email = payload.email.strip().lower()
  user = db.query(User).filter(User.email == email).first()
  if user:
    # update password + extras
    user.hashed_password = get_password_hash(payload.password)
    # try to set optional columns if they exist
    for field in ["name", "organisation", "company", "org", "meta"]:
      if hasattr(user, field):
        try:
          if field == "meta":
            meta = getattr(user, "meta") or {}
            meta.update({"name": payload.name, "organisation": payload.organisation})
            setattr(user, "meta", meta)
          else:
            setattr(user, field, getattr(payload, field, None))
        except Exception:
          pass
  else:
    user = User(
      email=email,
      hashed_password=get_password_hash(payload.password),
      is_admin=False,
      is_paid=False,
      created_at=datetime.utcnow(),
    )
    # try to set optional columns
    for field, val in [("name", payload.name), ("organisation", payload.organisation), ("company", payload.organisation)]:
      if hasattr(user, field):
        try: setattr(user, field, val)
        except Exception: pass
    db.add(user)

  db.commit()
  db.refresh(user)

  token = create_access_token(sub=user.email)
  return {
      "access_token": token,
      "token_type": "bearer",
      "email": user.email,
      "is_admin": bool(user.is_admin),
      "is_paid": bool(user.is_paid),
  }

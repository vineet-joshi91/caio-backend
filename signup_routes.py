# signup_routes.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy.orm import Session
from datetime import datetime

from db import get_db, User
from auth import create_access_token, get_password_hash

router = APIRouter(prefix="/api", tags=["auth"])

class SignupIn(BaseModel):
    name: str
    organisation: str | None = None
    email: EmailStr
    password: str

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("name cannot be empty")
        return v

    @field_validator("password")
    @classmethod
    def password_min_len(cls, v: str) -> str:
        if v is None or len(v) < 6:
            raise ValueError("password must be at least 6 characters")
        return v

@router.post("/signup")
def signup(payload: SignupIn, db: Session = Depends(get_db)):
    """
    Create or update a user, then return a bearer token.
    Safe even if your User model doesn't have 'name'/'organisation' columns.
    """
    email = payload.email.strip().lower()

    user = db.query(User).filter(User.email == email).first()
    if user:
        # Update password + optional columns
        user.hashed_password = get_password_hash(payload.password)
        for field, val in [("name", payload.name), ("organisation", payload.organisation), ("company", payload.organisation)]:
            if hasattr(user, field):
                try:
                    setattr(user, field, val)
                except Exception:
                    # Column exists but cannot be set (type, trigger) — ignore safely
                    pass
    else:
        # Create new user
        user = User(
            email=email,
            hashed_password=get_password_hash(payload.password),
        )
        # Optional flags/columns if they exist on your model
        for field, val in [
            ("is_admin", False),
            ("is_paid", False),
            ("created_at", datetime.utcnow()),
            ("name", payload.name),
            ("organisation", payload.organisation),
            ("company", payload.organisation),
        ]:
            if hasattr(user, field):
                try:
                    setattr(user, field, val)
                except Exception:
                    pass
        db.add(user)

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        # Most common: unique index on email or constraint error
        raise HTTPException(status_code=500, detail=f"Could not save user: {e!s}")

    db.refresh(user)

    token = create_access_token(sub=user.email)
    return {
        "access_token": token,
        "token_type": "bearer",
        "email": user.email,
        "is_admin": bool(getattr(user, "is_admin", False)),
        "is_paid": bool(getattr(user, "is_paid", False)),
    }

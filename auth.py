# auth.py  (put this next to main.py)
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from db import get_db, User  # User must have: email, hashed_password, is_admin, is_paid

# --- Config ---
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-prod")
JWT_ALG = "HS256"
ACCESS_TOKEN_EXPIRE_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRE_MIN", "1440"))  # 24h
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


# --- Password helpers ---
def get_password_hash(password: str) -> str:
    return pwd_ctx.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return pwd_ctx.verify(plain_password, hashed_password)
    except Exception:
        return False


# --- Auth primitives ---
def authenticate_user(email: str, password: str, db: Session = Depends(get_db)) -> Optional[User]:
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return None
    hp = getattr(user, "hashed_password", None)
    if not hp:
        return None
    if not verify_password(password, hp):
        return None
    return user


def create_access_token(sub: str, expires_minutes: int = ACCESS_TOKEN_EXPIRE_MIN) -> str:
    to_encode = {"sub": sub, "exp": datetime.utcnow() + timedelta(minutes=expires_minutes)}
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)


def _decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        sub: str = payload.get("sub")
        if not sub:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
        return sub
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# --- FastAPI dependency used by main.py ---
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    email = _decode_token(token)
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user

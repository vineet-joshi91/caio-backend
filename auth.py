# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 17:57:23 2025

@author: Vineet
"""

# auth.py

from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from db import User, get_db
from sqlalchemy.orm import Session
import os

# --- CONFIG ---

SECRET_KEY = os.environ.get("CAIO_SECRET_KEY", "super-secret-dev-key")  # Set this in .env or Render for security!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- PASSWORD HASHING ---

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# --- JWT TOKEN HELPERS ---

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

# --- USER HELPERS ---

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email.lower()).first()

def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

def create_user(db: Session, email: str, password: str, is_admin=False):
    user = User(
        email=email.lower(),
        hashed_password=hash_password(password),
        is_admin=is_admin,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def is_admin(user: User):
    return user.is_admin if user else False

def is_paid(user: User):
    return user.is_paid if user else False

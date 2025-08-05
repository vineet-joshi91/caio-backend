# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 11:30:23 2025

@author: Vineet
"""

# main.py

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from db import get_db, init_db, User, UsageLog
from auth import (
    create_user, authenticate_user, create_access_token,
    decode_access_token, get_user_by_email, is_admin, is_paid
)
from datetime import timedelta
from typing import Optional
import uvicorn
import os

# If you're using Streamlit/React as frontend, set allowed origins accordingly
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8501",
    # Add your deployed frontend domain(s) here
]

app = FastAPI(
    title="CAIO SaaS API",
    description="Backend API for CAIO SaaS App",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")

@app.on_event("startup")
def startup():
    init_db()  # Ensure DB tables exist on start

# --------- UTILS ---------

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    payload = decode_access_token(token)
    if not payload or "email" not in payload:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    user = get_user_by_email(db, payload["email"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def log_usage(db: Session, user: User, tokens: int, endpoint: str, status: str = "success"):
    log = UsageLog(
        user_id=user.id,
        tokens_used=tokens,
        endpoint=endpoint,
        status=status
    )
    db.add(log)
    db.commit()

# --------- ROUTES ---------

@app.post("/api/signup")
def signup(email: str, password: str, db: Session = Depends(get_db)):
    if get_user_by_email(db, email):
        raise HTTPException(status_code=400, detail="User already exists")
    is_admin_flag = (email.lower() == os.environ.get("ADMIN_EMAIL", "vineetpjoshi.71@gmail.com"))
    user = create_user(db, email, password, is_admin=is_admin_flag)
    return {"message": "Signup successful. Please log in."}

@app.post("/api/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    access_token = create_access_token(
        data={"user_id": user.id, "email": user.email, "is_admin": user.is_admin, "is_paid": user.is_paid},
        expires_delta=timedelta(days=1)
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/profile")
def get_profile(current_user: User = Depends(get_current_user)):
    return {
        "email": current_user.email,
        "is_admin": current_user.is_admin,
        "is_paid": current_user.is_paid,
        "created_at": str(current_user.created_at)
    }

@app.post("/api/analyze")
async def analyze(request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Restrict analyze access to paid/admin users
    if not (current_user.is_paid or current_user.is_admin):
        raise HTTPException(status_code=402, detail="Payment required for full analysis access.")
    data = await request.json()
    text = data.get("text")  # Could be file text or query
    which_brains = data.get("brains", ["CFO", "COO", "CMO", "CHRO"])  # Optional
    # Import your LLM/brains logic here
    from brains import analyze_cfo, analyze_cmo, analyze_coo, analyze_chro
    responses = {}
    tokens_used = 0
    if "CFO" in which_brains:
        result = analyze_cfo(text)
        responses["CFO"] = result
        tokens_used += len(result) // 4  # (Approx token count)
    if "COO" in which_brains:
        result = analyze_coo(text)
        responses["COO"] = result
        tokens_used += len(result) // 4
    if "CMO" in which_brains:
        result = analyze_cmo(text)
        responses["CMO"] = result
        tokens_used += len(result) // 4
    if "CHRO" in which_brains:
        result = analyze_chro(text)
        responses["CHRO"] = result
        tokens_used += len(result) // 4
    log_usage(db, current_user, tokens=tokens_used, endpoint="analyze")
    return {"insights": responses, "tokens_used": tokens_used}

@app.get("/api/admin")
def admin_dashboard(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin access required")
    # Example admin stats
    user_count = db.query(User).count()
    paid_users = db.query(User).filter(User.is_paid.is_(True)).count()
    usage_logs = db.query(UsageLog).count()
    return {
        "users": user_count,
        "paid_users": paid_users,
        "usage_logs": usage_logs,
        "admin_email": current_user.email
    }

# -- Payment routes and Stripe webhook would go here (payments.py) --

# --------- HEALTH CHECK ---------

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

# --------- MAIN ---------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

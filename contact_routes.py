# contact_routes.py
import os, smtplib, ssl
from email.message import EmailMessage
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr

router = APIRouter(prefix="/api", tags=["contact"])

TO_EMAIL = os.getenv("CONTACT_TO_EMAIL", "vineetpjoshi.71@gmail.com")

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")

class ContactIn(BaseModel):
    name: str
    organisation: str | None = None
    email: EmailStr
    need: str | None = None
    message: str | None = None

@router.post("/contact")
def contact(p: ContactIn):
    subject = f"CAIO Contact — {p.name} ({p.email})"
    body = f"""Name: {p.name}
Organisation: {p.organisation or '-'}
Email: {p.email}
Need: {p.need or '-'}
Message:
{p.message or '-'}
"""
    # If SMTP creds are present, send real email.
    if SMTP_HOST and SMTP_USER and SMTP_PASS:
        try:
            msg = EmailMessage()
            msg["From"] = SMTP_USER
            msg["To"] = TO_EMAIL
            msg["Subject"] = subject
            msg.set_content(body)
            ctx = ssl.create_default_context()
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
                s.starttls(context=ctx)
                s.login(SMTP_USER, SMTP_PASS)
                s.send_message(msg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Email send failed: {e!s}")
    # Always return 200 (even if we didn't email); logs in Render keep the details.
    return {"ok": True}

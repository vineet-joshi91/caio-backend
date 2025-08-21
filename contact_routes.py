# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 11:42:24 2025

@author: Vineet
"""

# contact_routes.py
from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse
import os, smtplib, ssl
from email.message import EmailMessage
import logging

router = APIRouter(prefix="/api", tags=["contact"])
logger = logging.getLogger("contact")

CONTACT_TO = os.getenv("CONTACT_TO", "vineetpjoshi.71@gmail.com")

SMTP_HOST = os.getenv("SMTP_HOST")        # e.g., "smtp.gmail.com"
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER")        # e.g., your Gmail address
SMTP_PASS = os.getenv("SMTP_PASS")        # App Password (not your login password)

def _send_email(subject: str, html: str, reply_to: str | None = None):
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS):
        logger.warning("SMTP not configured; skipping real send. Capturing lead in logs.")
        logger.info(f"[LEAD] {subject}\n{html}")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = CONTACT_TO
    if reply_to:
        msg["Reply-To"] = reply_to
    msg.set_content("HTML email required.")
    msg.add_alternative(html, subtype="html")

    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as server:
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
    return True

@router.post("/contact")
async def contact(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    company: str = Form(""),
    function: str = Form(""),
    message: str = Form(""),
):
    html = f"""
    <h2>CAIO – Enterprise Inquiry</h2>
    <p><b>Name:</b> {name}</p>
    <p><b>Email:</b> {email}</p>
    <p><b>Company:</b> {company}</p>
    <p><b>Function:</b> {function}</p>
    <p><b>Need / Message:</b><br>{message.replace('\n','<br>')}</p>
    """
    sent = _send_email("CAIO – Premium / Enterprise Inquiry", html, reply_to=email)
    return JSONResponse({"ok": True, "emailed": bool(sent)})

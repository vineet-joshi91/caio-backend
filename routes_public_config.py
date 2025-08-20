# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 17:42:31 2025

@author: Vineet
"""

# routes_public_config.py
from fastapi import APIRouter, Request
from typing import Literal, Optional
import os

router = APIRouter(prefix="/api", tags=["public"])

# If your proxy/CDN sets a country header, list them here (most do).
COUNTRY_HEADER_CANDIDATES = [
    "cf-ipcountry",             # Cloudflare
    "x-vercel-ip-country",      # Vercel
    "x-country-code",           # Custom/NGINX
    "x-geo-country"             # Generic
]

def resolve_region(request: Request, force: Optional[str]) -> Literal["IN", "INTL"]:
    # Allow override for testing (e.g., .../public-config?force=IN)
    if force:
        force = force.upper()
        return "IN" if force == "IN" else "INTL"

    # Try common country headers
    headers = request.headers
    cc = None
    for h in COUNTRY_HEADER_CANDIDATES:
        if h in headers:
            cc = headers[h]
            break

    # Minimal logic: India gets INR; everyone else → USD
    if (cc or "").upper() == "IN":
        return "IN"
    return "INTL"

@router.get("/public-config")
async def public_config(request: Request, force: Optional[str] = None):
    """
    Returns ONLY non-sensitive, public data used by the marketing site.
    Region is decided by CDN/proxy header or ?force=IN/INTL.
    """
    region = resolve_region(request, force)

    # You can pull these from env later if you want
    if region == "IN":
        currency = "INR"
        pro_price = 1999
        period = "month"
    else:
        currency = "USD"
        pro_price = 49
        period = "month"

    return {
        "region": region,           # "IN" or "INTL"
        "currency": currency,       # "INR" or "USD"
        "plans": {
            "free": {
                "name": "Free",
                "price": 0,
                "period": "month",
                "limits": {"uploads": 3, "brains": 2}
            },
            "pro": {
                "name": "Pro",
                "price": pro_price,
                "period": period
            },
            "premium": {
                "name": "Premium",
                "price": None,      # custom / contact us
                "period": "custom"
            }
        },
        "flags": {
            "promo": False,
            "notice": ""            # e.g., "Maintenance 10pm IST"
        },
        # Optional: strings your landing can display directly
        "copy": {
            "positioning": "CAIO replaces an entire C‑Suite worth of insights at just ₹1,999 or $49 a month."
        }
    }

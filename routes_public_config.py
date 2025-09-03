# -*- coding: utf-8 -*-
"""
Public config (pricing/limits/copy) for CAIO
- Detects region from CDN headers or ?force=IN/INTL override
- Exposes only non-sensitive values for website/app rendering
"""

from fastapi import APIRouter, Request
from typing import Literal, Optional
import os

router = APIRouter(prefix="/api", tags=["public"])

# Common CDN/proxy country headers
COUNTRY_HEADER_CANDIDATES = [
    "cf-ipcountry",          # Cloudflare
    "x-vercel-ip-country",   # Vercel
    "x-country-code",        # Custom/NGINX
    "x-geo-country",         # Generic
]

# ---- Config knobs (env-overridable) ------------------------------------------
PRO_PRICE_INR = int(os.getenv("PRO_PRICE_INR", "1999"))  # set "2999" if you prefer
PRO_PRICE_USD = int(os.getenv("PRO_PRICE_USD", "49"))
FREE_UPLOADS = int(os.getenv("FREE_UPLOADS", "3"))
FREE_BRAINS = int(os.getenv("FREE_BRAINS", "2"))
FREE_QUERIES_PER_DAY = int(os.getenv("FREE_QUERIES_PER_DAY", "3"))
POSITIONING_COPY = os.getenv(
    "POSITIONING_COPY",
    "CAIO replaces an entire C-Suite worth of insights at just ₹1,999 or $49 a month.",
)

def resolve_region(request: Request, force: Optional[str]) -> Literal["IN", "INTL"]:
    """Return 'IN' for India else 'INTL'. Supports ?force=IN/INTL override."""
    if force:
        force = force.upper()
        return "IN" if force == "IN" else "INTL"

    headers = request.headers
    cc = None
    for h in COUNTRY_HEADER_CANDIDATES:
        v = headers.get(h)
        if v:
            cc = v
            break

    if (cc or "").upper() == "IN":
        return "IN"
    return "INTL"

@router.get("/public-config")
async def public_config(request: Request, force: Optional[str] = None):
    """
    Returns ONLY non-sensitive, public data used by the marketing site/frontend.
    Region is decided by CDN/proxy header or ?force=IN/INTL.
    """
    region = resolve_region(request, force)

    if region == "IN":
        currency = "INR"
        pro_price = PRO_PRICE_INR
        period = "month"
    else:
        currency = "USD"
        pro_price = PRO_PRICE_USD
        period = "month"

    return {
        "region": region,         # "IN" or "INTL"
        "currency": currency,     # "INR" or "USD"
        "plans": {
            "free": {
                "name": "Free",
                "price": 0,
                "period": "month",
                "limits": {
                    "uploads": FREE_UPLOADS,
                    "brains": FREE_BRAINS,
                    "queries_per_day": FREE_QUERIES_PER_DAY,  # surfaced to UI
                },
            },
            "pro": {
                "name": "Pro",
                "price": pro_price,
                "period": period,
            },
            "premium": {
                "name": "Premium",
                "price": None,      # contact us / custom
                "period": "custom",
            },
        },
        "flags": {
            "promo": False,
            "notice": "",          # e.g., "Maintenance 10pm IST"
        },
        "copy": {
            "positioning": POSITIONING_COPY
        },
    }

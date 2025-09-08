# backend/routes_public_config.py
# -*- coding: utf-8 -*-
from fastapi import APIRouter, Request
from typing import Literal, Optional
import os

router = APIRouter(prefix="/api", tags=["public"])

COUNTRY_HEADER_CANDIDATES = [
    "cf-ipcountry", "x-vercel-ip-country", "x-country-code", "x-geo-country"
]

# ---- Env-configurable knobs ---------------------------------------------------
PRO_PRICE_INR      = int(os.getenv("PRO_PRICE_INR", "1999"))
PRO_PRICE_USD      = int(os.getenv("PRO_PRICE_USD", "49"))

PRO_PLUS_PRICE_INR = int(os.getenv("PRO_PLUS_PRICE_INR", "2999"))  # Limited Chat
PRO_PLUS_PRICE_USD = int(os.getenv("PRO_PLUS_PRICE_USD", "59"))

PREMIUM_PRICE_INR  = int(os.getenv("PREMIUM_PRICE_INR", "7999"))
PREMIUM_PRICE_USD  = int(os.getenv("PREMIUM_PRICE_USD", "99"))

FREE_UPLOADS          = int(os.getenv("FREE_UPLOADS", "3"))
FREE_BRAINS           = int(os.getenv("FREE_BRAINS", "2"))
FREE_QUERIES_PER_DAY  = int(os.getenv("FREE_QUERIES_PER_DAY", "5"))  # per doc plan

PRO_QUERIES_PER_DAY   = int(os.getenv("PRO_QUERIES_PER_DAY", "50"))
PRO_PLUS_MSGS_PER_DAY = int(os.getenv("PRO_PLUS_MSGS_PER_DAY", "25"))  # limited chat
PREMIUM_MSGS_PER_DAY  = int(os.getenv("PREMIUM_MSGS_PER_DAY", "50"))
UPLOADS_PER_DAY_PAID  = int(os.getenv("UPLOADS_PER_DAY_PAID", "50"))

POSITIONING_COPY = os.getenv(
    "POSITIONING_COPY",
    "CAIO replaces an entire C-Suite worth of insights at just ₹1,999 or $49 a month."
)

def resolve_region(request: Request, force: Optional[str]) -> Literal["IN", "INTL"]:
    if force:
        return "IN" if force.upper() == "IN" else "INTL"
    headers = request.headers
    cc = None
    for h in COUNTRY_HEADER_CANDIDATES:
        v = headers.get(h)
        if v:
            cc = v
            break
    return "IN" if (cc or "").upper() == "IN" else "INTL"

@router.get("/public-config")
async def public_config(request: Request, force: Optional[str] = None):
    region = resolve_region(request, force)
    is_in = region == "IN"
    currency = "INR" if is_in else "USD"

    pro_price      = PRO_PRICE_INR if is_in else PRO_PRICE_USD
    pro_plus_price = PRO_PLUS_PRICE_INR if is_in else PRO_PLUS_PRICE_USD
    premium_price  = PREMIUM_PRICE_INR if is_in else PREMIUM_PRICE_USD

    return {
        "region": region,
        "currency": currency,
        "plans": {
            "free": {
                "name": "Free",
                "price": 0,
                "period": "month",
                "limits": {
                    "uploads": FREE_UPLOADS,
                    "brains": FREE_BRAINS,
                    "queries_per_day": FREE_QUERIES_PER_DAY
                }
            },
            "pro": {
                "name": "Pro",
                "price": pro_price,
                "period": "month",
                "limits": {
                    "queries_per_day": PRO_QUERIES_PER_DAY,
                    "uploads_per_day": UPLOADS_PER_DAY_PAID
                }
            },
            "pro_plus": {
                "name": "Pro+",
                "tagline": "Limited Chat mode + Analyze",
                "price": pro_plus_price,
                "period": "month",
                "limits": {
                    "chat_msgs_per_day": PRO_PLUS_MSGS_PER_DAY,
                    "uploads_per_day": UPLOADS_PER_DAY_PAID
                }
            },
            "premium": {
                "name": "Premium",
                "price": premium_price,
                "period": "month",
                "limits": {
                    "chat_msgs_per_day": PREMIUM_MSGS_PER_DAY,
                    "uploads_per_day": UPLOADS_PER_DAY_PAID
                }
            }
        },
        "flags": {
            "promo": False,
            "notice": ""
        },
        "copy": {
            "positioning": POSITIONING_COPY
        }
    }

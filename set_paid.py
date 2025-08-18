# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 16:31:45 2025

@author: Vineet
"""

# set_paid.py  (place this file in the backend root, next to main.py)
"""
Set or unset is_paid for a user in the 'users' table.

Usage (Windows PowerShell):
  $env:DATABASE_URL = "<your Render Postgres INTERNAL URL>"
  python set_paid.py --email "vineetpjoshi.71@gmail.com" --paid true
"""

import argparse
import os
from sqlalchemy import create_engine, text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True)
    parser.add_argument("--paid", required=True, choices=["true", "false"])
    args = parser.parse_args()

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise SystemExit("DATABASE_URL env var is required")

    engine = create_engine(db_url, future=True)
    paid_val = args.paid.lower() == "true"

    with engine.begin() as conn:
        result = conn.execute(text("UPDATE users SET is_paid = :paid WHERE email = :email"),
                              {"paid": paid_val, "email": args.email})
        if result.rowcount == 0:
            raise SystemExit(f"User not found: {args.email}")
        print(f"OK: {args.email} -> is_paid={paid_val}")

if __name__ == "__main__":
    main()

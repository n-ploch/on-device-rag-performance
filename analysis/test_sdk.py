"""Inspect raw data returned by the Langfuse v1 API.

Run: python analysis/test_sdk.py

Prints raw metadata from one rag.evaluation observation and one score,
so we can see the exact structure before writing transformation code.
"""

import base64
import json
import os

import httpx
from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

lf = get_client()

# ── Observations v1 ──────────────────────────────────────────────────────────
print("=== observations v1 (one rag.evaluation span) ===")
resp = lf.api.observations.get_many(name="rag.evaluation", limit=50)

if not resp.data:
    print("No observations found.")
else:
    for i in range(len(resp.data)):
        obs = resp.data[i]
        print(f"\nobs.metadata raw:")
        print(json.dumps(obs.metadata, indent=2, default=str))
        print(f"\nobs.latency: {obs.latency}")

# ── Scores v1 (direct HTTP) ───────────────────────────────────────────────────
print("\n=== scores v1 (direct HTTP, one score) ===")
host = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000").rstrip("/")
token = base64.b64encode(
    f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
).decode()

r = httpx.get(
    f"{host}/api/public/scores",
    params={"limit": 2},
    headers={"Authorization": f"Basic {token}"},
    timeout=10,
)
r.raise_for_status()
body = r.json()
if body.get("data"):
    print(json.dumps(body["data"][1], indent=2))
else:
    print("No scores found.")

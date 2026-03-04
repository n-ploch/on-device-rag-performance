"""Export rag.evaluation metrics and Langfuse scores to a single Parquet file.

Usage:
    python analysis/langfuse_export.py --session-id mistral_q4_baseline_001
    python analysis/langfuse_export.py --session-id my_run --from 2026-01-01

Reads credentials from .env: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL.
Output: local/metric-export/YYYY-MM-DD_HH-MM_langfuse_export_<session_id>.parquet
"""

from __future__ import annotations

import argparse
import base64
import datetime
import os
from pathlib import Path
from typing import Any

import httpx
import json

import pandas as pd
from dotenv import load_dotenv
from langfuse import get_client

# Keys inside attributes to drop (large text, not useful for analysis)
_DROP_KEYS = {"ground_truth", "retrieval_context"}


def _parse_str(v: Any) -> Any:
    """Coerce string-encoded scalars to their native Python type.

    Handles the special case where a zero numeric value arrives as a
    JSON-encoded OTLP AnyValue string, e.g. '{"intValue":0}'.
    """
    if not isinstance(v, str):
        return v
    if v.startswith("{"):
        try:
            parsed = json.loads(v)
            for key in ("intValue", "doubleValue", "floatValue"):
                if key in parsed:
                    return parsed[key]
        except (json.JSONDecodeError, TypeError):
            pass
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _clean_col(name: str) -> str:
    """Strip 'custom.' prefix and replace dots with underscores."""
    return name.removeprefix("custom.").replace(".", "_")


def _extract_attrs(metadata: dict | None) -> dict[str, Any]:
    """Pull the flat span attributes out of the Langfuse OTLP metadata wrapper.

    Langfuse stores OTLP span attributes under metadata["attributes"] alongside
    "resourceAttributes" and "scope".  All values arrive as plain strings.
    """
    attrs = (metadata or {}).get("attributes", {})
    return {
        _clean_col(k): _parse_str(v)
        for k, v in attrs.items()
        if k not in _DROP_KEYS
    }


# ── Fetching ─────────────────────────────────────────────────────────────────

def fetch_observations(
    from_timestamp: datetime.datetime | None = None,
    to_timestamp: datetime.datetime | None = None,
) -> list[Any]:
    """Paginate through v1 observations for all rag.evaluation spans."""
    lf = get_client()
    results: list[Any] = []
    page = 1
    while True:
        resp = lf.api.observations.get_many(
            name="rag.evaluation",
            page=page,
            limit=50,
            from_start_time=from_timestamp,
            to_start_time=to_timestamp,
        )
        if not resp.data:
            break
        results.extend(resp.data)
        if page >= (resp.meta.total_pages or 1):
            break
        page += 1
    return results


def fetch_scores(
    from_timestamp: datetime.datetime | None = None,
    to_timestamp: datetime.datetime | None = None,
) -> list[dict]:
    """Paginate through v1 /api/public/scores (SDK only exposes score listing via v2)."""
    host = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000").rstrip("/")
    token = base64.b64encode(
        f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
    ).decode()

    results: list[dict] = []
    page = 1
    params: dict[str, Any] = {"limit": 50}
    if from_timestamp:
        params["fromTimestamp"] = from_timestamp.isoformat()
    if to_timestamp:
        params["toTimestamp"] = to_timestamp.isoformat()

    with httpx.Client(timeout=30) as client:
        while True:
            params["page"] = page
            r = client.get(
                f"{host}/api/public/scores",
                params=params,
                headers={"Authorization": f"Basic {token}"},
            )
            r.raise_for_status()
            body = r.json()
            data = body.get("data", [])
            if not data:
                break
            results.extend(data)
            if page >= body.get("meta", {}).get("totalPages", 1):
                break
            page += 1
    return results


# ── DataFrame assembly ────────────────────────────────────────────────────────

def to_dataframe(observations: list[Any], scores: list[dict]) -> pd.DataFrame:
    """Combine observations and scores into one flat DataFrame.

    One row per rag.evaluation span.  Langfuse scores are pivoted wide and
    left-joined on trace_id; score columns are prefixed 'score_'.
    """
    rows = [
        {
            "observation_id": obs.id,
            "trace_id": obs.trace_id,
            "start_time": obs.start_time,
            "end_time": obs.end_time,
            "latency_ms": obs.latency,
            **_extract_attrs(obs.metadata),
        }
        for obs in observations
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if scores:
        score_rows = [
            {"trace_id": s["traceId"], "name": s["name"], "value": s.get("value")}
            for s in scores
            if s.get("traceId") and s.get("name") is not None
        ]
        if score_rows:
            score_wide = (
                pd.DataFrame(score_rows)
                .pivot_table(index="trace_id", columns="name", values="value", aggfunc="first")
                .reset_index()
            )
            score_wide.columns = pd.Index([
                f"score_{c}" if c != "trace_id" else c
                for c in score_wide.columns
            ])
            df = df.merge(score_wide, on="trace_id", how="left")

    return df


# ── Full pipeline ─────────────────────────────────────────────────────────────

def export(
    session_id: str,
    output_dir: str | Path = "local/metric-export",
    from_timestamp: datetime.datetime | None = None,
    to_timestamp: datetime.datetime | None = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Fetching rag.evaluation observations...")
    observations = fetch_observations(from_timestamp, to_timestamp)
    print(f"  {len(observations)} observations")

    print("Fetching scores...")
    scores = fetch_scores(from_timestamp, to_timestamp)
    print(f"  {len(scores)} scores")

    df = to_dataframe(observations, scores)
    print(f"  {len(df)} rows")

    ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d_%H-%M")
    path = out / f"{ts}_langfuse_export_{session_id}.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved → {path}")
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Langfuse metrics to Parquet")
    p.add_argument("--session-id", required=True,
                   help="Label for this export batch (used in the filename)")
    p.add_argument("--output-dir", default="local/metric-export")
    p.add_argument("--from", dest="from_date", metavar="YYYY-MM-DD")
    p.add_argument("--to", dest="to_date", metavar="YYYY-MM-DD")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = _parse_args()

    from_ts = (
        datetime.datetime.fromisoformat(args.from_date).replace(tzinfo=datetime.timezone.utc)
        if args.from_date else None
    )
    to_ts = (
        datetime.datetime.fromisoformat(args.to_date).replace(tzinfo=datetime.timezone.utc)
        if args.to_date else None
    )

    export(session_id=args.session_id, output_dir=args.output_dir,
           from_timestamp=from_ts, to_timestamp=to_ts)


if __name__ == "__main__":
    main()

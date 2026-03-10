"""Export rag.evaluation metrics and Langfuse scores to Parquet files.

Uses trace.list(session_id=...) + trace.get() to fetch all spans and scores
per trace, properly filtered by session_id (= run_id).

Each trace.get() returns TraceWithFullDetails, which includes all child
observations (rag.retrieval, rag.generation) and scores in one call —
no separate fetches or joins needed.

Usage:
    # single session
    python analysis/langfuse_export.py --session-id mistral_q4_baseline_001

    # multiple sessions (comma-separated) — one Parquet file per session
    python analysis/langfuse_export.py --session-id id1,id2,id3

    # glob pattern — lists all sessions, matches client-side, one file per match
    python analysis/langfuse_export.py --session-id "*_q4_*"

    python analysis/langfuse_export.py --session-id my_run --from 2026-01-01

Reads credentials from .env: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL.
Output: local/metric-export/YYYY-MM-DD_HH-MM_langfuse_export_<session_id>.parquet
"""

from __future__ import annotations

import argparse
import datetime
import fnmatch
import json
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langfuse import get_client

# Keys inside attributes to drop (large text, not useful for analysis)
_DROP_KEYS = {
    "ground_truth",
    "retrieval_context",
    "custom.generation.ground_truth",
    "custom.generation.retrieval_context",
    "custom.retrieval.context",
    "gen_ai.prompt",
    "gen_ai.completion",
}

# Span attribute merge order — later entries overwrite earlier on key conflicts,
# so rag.evaluation (root) wins over child spans.
_SPAN_PRIORITY = ["rag.generation", "rag.retrieval", "rag.evaluation"]


def _clean_col(name: str) -> str:
    """Strip 'custom.' prefix and replace dots with underscores."""
    return name.removeprefix("custom.").replace(".", "_")


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


def _extract_attrs(metadata: Any) -> dict[str, Any]:
    """Pull the flat span attributes out of the Langfuse OTLP metadata wrapper.

    Langfuse stores OTLP span attributes under metadata["attributes"] alongside
    "resourceAttributes" and "scope".  All values arrive as plain strings.
    """
    attrs = (metadata or {}).get("attributes", {}) if isinstance(metadata, dict) else {}
    return {
        _clean_col(k): _parse_str(v)
        for k, v in attrs.items()
        if k not in _DROP_KEYS
    }


# ── Fetching ─────────────────────────────────────────────────────────────────

def fetch_traces(
    session_id: str,
    from_timestamp: datetime.datetime | None = None,
    to_timestamp: datetime.datetime | None = None,
) -> list[Any]:
    """Fetch all TraceWithFullDetails for a session.

    Paginates trace.list(session_id=session_id) to collect trace IDs, then
    calls trace.get(trace_id) for each to get full observations and scores.
    """
    lf = get_client()

    trace_ids: list[str] = []
    page = 1
    while True:
        resp = lf.api.trace.list(
            name="rag.evaluation",
            session_id=session_id,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            page=page,
            limit=50,
        )
        if not resp.data:
            break
        trace_ids.extend(t.id for t in resp.data)
        if page >= (resp.meta.total_pages or 1):
            break
        page += 1

    traces = []
    for i, tid in enumerate(trace_ids, 1):
        print(f"\r  Fetching trace {i}/{len(trace_ids)}...", end="", flush=True)
        traces.append(lf.api.trace.get(tid))
    if trace_ids:
        print()

    return traces


def resolve_session_ids(
    spec: str,
    from_timestamp: datetime.datetime | None = None,
    to_timestamp: datetime.datetime | None = None,
) -> list[str]:
    """Expand a session-id spec into a list of concrete session IDs.

    Accepts:
      - A single ID with no wildcards → ``["my_session"]``
      - Comma-separated IDs          → ``["id1", "id2"]``
      - A glob pattern (contains ``*`` or ``?``) → lists all sessions from
        Langfuse and returns those whose ID matches via :func:`fnmatch.fnmatch`.
    """
    # Glob pattern — must list all sessions and filter client-side
    if "*" in spec or "?" in spec:
        print(f"Resolving glob pattern '{spec}' against available sessions...")
        lf = get_client()
        all_ids: list[str] = []
        page = 1
        while True:
            resp = lf.api.sessions.list(
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp,
                page=page,
                limit=50,
            )
            if not resp.data:
                break
            all_ids.extend(s.id for s in resp.data)
            if page >= (resp.meta.total_pages or 1):
                break
            page += 1
        matched = [sid for sid in all_ids if fnmatch.fnmatch(sid, spec)]
        print(f"  {len(matched)}/{len(all_ids)} sessions matched")
        return matched

    # Comma-separated list (or plain single ID)
    return [s.strip() for s in spec.split(",") if s.strip()]


# ── DataFrame assembly ────────────────────────────────────────────────────────

def trace_to_row(trace: Any) -> dict[str, Any]:
    """Flatten a TraceWithFullDetails into one row.

    Span attributes are merged in priority order (rag.generation first,
    rag.evaluation last) so root-span values win on key conflicts.
    Only the first observation per span name is used (deduplicates reruns).
    Scores are pivoted wide with a 'score_' prefix.
    """
    obs_by_name: dict[str, Any] = {}
    for obs in trace.observations or []:
        if obs.name in _SPAN_PRIORITY and obs.name not in obs_by_name:
            obs_by_name[obs.name] = obs

    attrs: dict[str, Any] = {}
    for span_name in _SPAN_PRIORITY:
        if obs := obs_by_name.get(span_name):
            attrs.update(_extract_attrs(obs.metadata))

    root = obs_by_name.get("rag.evaluation")
    row: dict[str, Any] = {
        "trace_id": trace.id,
        "session_id": trace.session_id,
        "observation_id": root.id if root else None,
        "start_time": root.start_time if root else trace.timestamp,
        "end_time": root.end_time if root else None,
        "latency_ms": (root.latency * 1000) if root and root.latency else None,
        **attrs,
    }

    for score in trace.scores or []:
        row[f"score_{score.name}"] = getattr(score, "value", None)

    return row


def to_dataframe(traces: list[Any]) -> pd.DataFrame:
    """Convert a list of TraceWithFullDetails to a flat DataFrame."""
    return pd.DataFrame([trace_to_row(t) for t in traces])


# ── Full pipeline ─────────────────────────────────────────────────────────────

def export(
    session_id: str,
    output_dir: str | Path = "local/metric-export",
    from_timestamp: datetime.datetime | None = None,
    to_timestamp: datetime.datetime | None = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Fetching traces for session '{session_id}'...")
    traces = fetch_traces(session_id, from_timestamp, to_timestamp)
    print(f"  {len(traces)} traces fetched")

    df = to_dataframe(traces)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d_%H-%M")
    path = out / f"{ts}_langfuse_export_{session_id}.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved → {path}")
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Langfuse metrics to Parquet")
    p.add_argument(
        "--session-id", required=True,
        help=(
            "Session ID(s) to export. Accepts: a single ID, a comma-separated "
            "list of IDs, or a glob pattern (e.g. '*_q4_*') which is matched "
            "against all available sessions."
        ),
    )
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

    session_ids = resolve_session_ids(args.session_id, from_ts, to_ts)
    if not session_ids:
        print("No matching sessions found.")
        return

    for i, sid in enumerate(session_ids, 1):
        if len(session_ids) > 1:
            print(f"\n[{i}/{len(session_ids)}]", end=" ")
        export(session_id=sid, output_dir=args.output_dir,
               from_timestamp=from_ts, to_timestamp=to_ts)


if __name__ == "__main__":
    main()

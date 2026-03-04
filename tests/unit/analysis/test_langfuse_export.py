"""Unit tests for analysis/langfuse_export.py pure functions."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

# Make the analysis/ directory importable without installing the package
sys.path.insert(0, str(Path(__file__).parents[3] / "analysis"))

from langfuse_export import (
    _clean_col,
    _extract_attrs,
    _parse_str,
    to_dataframe,
)


# ── _parse_str ────────────────────────────────────────────────────────────────


class TestParseStr:
    def test_non_string_passthrough(self):
        assert _parse_str(3.14) == 3.14
        assert _parse_str(42) == 42
        assert _parse_str(None) is None
        assert _parse_str(True) is True

    def test_int_string(self):
        assert _parse_str("7") == 7
        assert _parse_str("-3") == -3

    def test_float_string(self):
        result = _parse_str("0.333")
        assert isinstance(result, float)
        assert abs(result - 0.333) < 1e-9

    def test_bool_strings(self):
        assert _parse_str("true") is True
        assert _parse_str("True") is True
        assert _parse_str("TRUE") is True
        assert _parse_str("false") is False
        assert _parse_str("False") is False

    def test_otlp_int_value_zero(self):
        assert _parse_str('{"intValue":0}') == 0

    def test_otlp_double_value_zero(self):
        assert _parse_str('{"doubleValue":0}') == 0

    def test_otlp_float_value_zero(self):
        assert _parse_str('{"floatValue":0}') == 0

    def test_otlp_nonzero_int_value(self):
        assert _parse_str('{"intValue":5}') == 5

    def test_malformed_json_string_returned_as_is(self):
        val = '{"broken'
        assert _parse_str(val) == val

    def test_plain_string_returned_as_is(self):
        assert _parse_str("hello") == "hello"
        assert _parse_str("mistral_q4") == "mistral_q4"

    def test_empty_string(self):
        assert _parse_str("") == ""


# ── _clean_col ────────────────────────────────────────────────────────────────


class TestCleanCol:
    def test_strips_custom_prefix(self):
        assert _clean_col("custom.metrics.recall_at_k") == "metrics_recall_at_k"

    def test_replaces_dots_with_underscores(self):
        assert _clean_col("hardware.cpu_percent") == "hardware_cpu_percent"

    def test_no_prefix_no_dots(self):
        assert _clean_col("run_id") == "run_id"

    def test_custom_prefix_only(self):
        assert _clean_col("custom.run_id") == "run_id"

    def test_multiple_dots(self):
        assert _clean_col("custom.a.b.c") == "a_b_c"


# ── _extract_attrs ─────────────────────────────────────────────────────────────


class TestExtractAttrs:
    def _make_metadata(self, attrs: dict) -> dict:
        return {"attributes": attrs, "resourceAttributes": {}, "scope": {}}

    def test_basic_extraction(self):
        meta = self._make_metadata({"custom.metrics.recall_at_k": "0.5", "run_id": "run1"})
        result = _extract_attrs(meta)
        assert result["metrics_recall_at_k"] == 0.5
        assert result["run_id"] == "run1"

    def test_drops_ground_truth(self):
        meta = self._make_metadata({"ground_truth": "answer text", "run_id": "r"})
        result = _extract_attrs(meta)
        assert "ground_truth" not in result
        assert result["run_id"] == "r"

    def test_drops_retrieval_context(self):
        meta = self._make_metadata({"retrieval_context": "ctx", "run_id": "r"})
        result = _extract_attrs(meta)
        assert "retrieval_context" not in result

    def test_none_metadata_returns_empty(self):
        assert _extract_attrs(None) == {}

    def test_missing_attributes_key_returns_empty(self):
        assert _extract_attrs({"resourceAttributes": {}}) == {}

    def test_zero_value_coerced(self):
        meta = self._make_metadata({"custom.metrics.mrr": '{"intValue":0}'})
        result = _extract_attrs(meta)
        assert result["metrics_mrr"] == 0

    def test_bool_coerced(self):
        meta = self._make_metadata({"custom.hardware.gpu_available": "false"})
        result = _extract_attrs(meta)
        assert result["hardware_gpu_available"] is False


# ── to_dataframe ──────────────────────────────────────────────────────────────


def _make_obs(
    obs_id: str,
    trace_id: str,
    latency: float,
    attrs: dict,
    start_time=None,
    end_time=None,
):
    """Return a mock observation object matching the FernLangfuse v1 shape."""
    meta = {"attributes": attrs, "resourceAttributes": {}, "scope": {}}
    ns = types.SimpleNamespace(
        id=obs_id,
        trace_id=trace_id,
        start_time=start_time,
        end_time=end_time,
        latency=latency,
        metadata=meta,
    )
    return ns


class TestToDataframe:
    def test_empty_observations_returns_empty_df(self):
        df = to_dataframe([], [])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_single_observation_no_scores(self):
        obs = _make_obs("o1", "t1", 123.0, {"run_id": "r1", "custom.metrics.recall_at_k": "0.5"})
        df = to_dataframe([obs], [])
        assert len(df) == 1
        assert df.iloc[0]["trace_id"] == "t1"
        assert df.iloc[0]["latency_ms"] == 123.0
        assert df.iloc[0]["metrics_recall_at_k"] == 0.5

    def test_scores_pivoted_and_joined(self):
        obs = _make_obs("o1", "t1", 50.0, {"run_id": "r1"})
        scores = [
            {"traceId": "t1", "name": "faithfulness", "value": 0.9},
            {"traceId": "t1", "name": "relevance", "value": 0.8},
        ]
        df = to_dataframe([obs], scores)
        assert df.iloc[0]["score_faithfulness"] == pytest.approx(0.9)
        assert df.iloc[0]["score_relevance"] == pytest.approx(0.8)

    def test_unmatched_score_trace_id_produces_nan(self):
        obs = _make_obs("o1", "t1", 50.0, {"run_id": "r1"})
        scores = [{"traceId": "t_other", "name": "faithfulness", "value": 1.0}]
        df = to_dataframe([obs], scores)
        assert "score_faithfulness" in df.columns
        assert pd.isna(df.iloc[0]["score_faithfulness"])

    def test_multiple_observations_with_scores(self):
        obs1 = _make_obs("o1", "t1", 10.0, {"run_id": "run_a"})
        obs2 = _make_obs("o2", "t2", 20.0, {"run_id": "run_b"})
        scores = [
            {"traceId": "t1", "name": "faithfulness", "value": 0.7},
            {"traceId": "t2", "name": "faithfulness", "value": 0.95},
        ]
        df = to_dataframe([obs1, obs2], scores)
        assert len(df) == 2
        t1_row = df[df["trace_id"] == "t1"].iloc[0]
        t2_row = df[df["trace_id"] == "t2"].iloc[0]
        assert t1_row["score_faithfulness"] == pytest.approx(0.7)
        assert t2_row["score_faithfulness"] == pytest.approx(0.95)

    def test_score_missing_trace_id_skipped(self):
        obs = _make_obs("o1", "t1", 50.0, {"run_id": "r1"})
        scores = [
            {"name": "faithfulness", "value": 1.0},  # no traceId
            {"traceId": "t1", "name": "relevance", "value": 0.8},
        ]
        df = to_dataframe([obs], scores)
        assert df.iloc[0]["score_relevance"] == pytest.approx(0.8)
        assert "score_faithfulness" not in df.columns

    def test_drop_keys_not_in_output(self):
        obs = _make_obs(
            "o1", "t1", 50.0,
            {"ground_truth": "secret", "retrieval_context": "ctx", "run_id": "r1"},
        )
        df = to_dataframe([obs], [])
        assert "ground_truth" not in df.columns
        assert "retrieval_context" not in df.columns
        assert df.iloc[0]["run_id"] == "r1"

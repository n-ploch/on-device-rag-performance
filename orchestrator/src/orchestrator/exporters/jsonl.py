"""JSONL exporter for OpenTelemetry-compatible span payloads."""

from __future__ import annotations

import json
from pathlib import Path


def _typed_attribute_value(value):
    if value is None:
        return {"nullValue": None}
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": value}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


class JSONLSpanExporter:
    """Serialize spans to newline-delimited JSON documents."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("a", encoding="utf-8")

    def export(self, spans) -> None:
        for span in spans:
            payload = self._serialize_span(span)
            self._file.write(json.dumps(payload) + "\n")
        self._file.flush()

    def shutdown(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def _serialize_span(self, span) -> dict:
        payload = {
            "trace_id": f"{span.context.trace_id:032x}",
            "span_id": f"{span.context.span_id:016x}",
            "name": span.name,
            "start_time_unix_nano": int(span.start_time),
            "end_time_unix_nano": int(span.end_time),
            "status": {"code": span.status.status_code.name},
        }

        parent_id = getattr(getattr(span, "parent", None), "span_id", None)
        if parent_id is not None:
            payload["parent_span_id"] = f"{parent_id:016x}"

        attributes = []
        for key, value in (span.attributes or {}).items():
            attributes.append({"key": key, "value": _typed_attribute_value(value)})
        payload["attributes"] = attributes

        return payload

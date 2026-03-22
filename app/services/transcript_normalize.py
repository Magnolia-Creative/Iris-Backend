"""Convert Modal / WhisperX transcript payloads to JSON-safe Python types (Postgres JSONB, API)."""

from __future__ import annotations

from typing import Any

import numpy as np


def json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.generic):
        return json_safe_value(value.item())
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, dict):
        return {k: json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(v) for v in value]
    return value


def normalize_transcript_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [json_safe_value(dict(seg)) for seg in segments]


def segments_to_full_text(segments: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for seg in segments:
        t = seg.get("text")
        if isinstance(t, str):
            s = t.strip()
            if s:
                parts.append(s)
    return " ".join(parts)

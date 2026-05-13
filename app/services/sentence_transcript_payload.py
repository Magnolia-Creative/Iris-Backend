"""Normalize /transcriptions/sentences API output into JSONB shape used by intent transcript hydration."""

from __future__ import annotations

from typing import Any


def intent_jsonb_from_sentence_api_result(result: dict[str, Any]) -> dict[str, Any]:
    full_text = str(result.get("full_text") or "").strip()
    sentences = result.get("sentences")
    if not isinstance(sentences, list):
        sentences = []
    return {"full_text": full_text, "segments": _segments_for_intent_hydration(sentences)}


def _segments_for_intent_hydration(sentences: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in sentences:
        if not isinstance(s, dict):
            continue
        raw_words = s.get("words")
        if isinstance(raw_words, list) and raw_words:
            words_out: list[dict[str, Any]] = []
            for w in raw_words:
                if not isinstance(w, dict):
                    continue
                word = str(w.get("word") or w.get("text") or "").strip()
                if not word:
                    continue
                start = _as_seconds(w.get("start"))
                end = _as_seconds(w.get("end"))
                if start is None or end is None:
                    continue
                if end <= start:
                    end = start + 1e-6
                words_out.append({"word": word, "start": start, "end": end})
            if words_out:
                seg_text = str(s.get("text") or "").strip() or " ".join(str(x["word"]) for x in words_out)
                out.append({"text": seg_text, "words": words_out})
            continue
        text = str(s.get("text") or "").strip()
        if not text:
            continue
        start = _as_seconds(s.get("start"))
        end = _as_seconds(s.get("end"))
        if start is None:
            start = 0.0
        if end is None:
            end = start + 1e-3
        if end <= start:
            end = start + 1e-6
        out.append({"text": text, "words": [{"word": text, "start": start, "end": end}]})
    return out


def _as_seconds(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None

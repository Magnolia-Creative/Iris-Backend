from copy import deepcopy
from datetime import datetime, timezone
import re
import time
from typing import Any


_PROMPT_ENTITY_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "has",
    "have",
    "include",
    "interview",
    "only",
    "podcast",
    "specifically",
    "that",
    "the",
    "this",
    "today",
    "want",
    "with",
    "you",
}

_session_debug_store: dict[str, dict[str, Any]] = {}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _public_entry(entry: dict[str, Any]) -> dict[str, Any]:
    public = {key: deepcopy(value) for key, value in entry.items() if not key.startswith("_")}
    public["total_elapsed_ms"] = int((time.perf_counter() - entry["_started_perf"]) * 1000)
    return public


def _ensure_entry(session_id: str, state: dict[str, Any] | None = None) -> dict[str, Any]:
    entry = _session_debug_store.get(session_id)
    if entry is None:
        now = _timestamp()
        entry = {
            "session_id": session_id,
            "started_at": now,
            "updated_at": now,
            "current_node": None,
            "current_status_message": None,
            "active_node": None,
            "latest_state": deepcopy(state) if isinstance(state, dict) else None,
            "node_timings": [],
            "events": [],
            "_started_perf": time.perf_counter(),
            "_node_starts": {},
        }
        _session_debug_store[session_id] = entry
        return entry

    if isinstance(state, dict):
        entry["latest_state"] = deepcopy(state)
        status_message = str(state.get("status_message") or "").strip()
        if status_message:
            entry["current_status_message"] = status_message
        status_details = state.get("status_details")
        if isinstance(status_details, dict) and isinstance(status_details.get("node"), str):
            entry["current_node"] = status_details["node"]
    entry["updated_at"] = _timestamp()
    return entry


def initialize_session_debug(session_id: str, state: dict[str, Any] | None = None) -> None:
    _ensure_entry(session_id, state=state)


def update_session_debug_state(session_id: str, state: dict[str, Any]) -> None:
    entry = _ensure_entry(session_id, state=state)
    status_message = str(state.get("status_message") or "").strip()
    status_details = state.get("status_details")
    if status_message:
        entry["current_status_message"] = status_message
    if isinstance(status_details, dict) and isinstance(status_details.get("node"), str):
        entry["current_node"] = status_details["node"]
    entry["updated_at"] = _timestamp()


def record_session_event(session_id: str, event: dict[str, Any]) -> dict[str, Any]:
    entry = _ensure_entry(session_id)
    event_copy = deepcopy(event)
    now_perf = time.perf_counter()
    now_iso = _timestamp()
    event_type = str(event_copy.get("type") or "").strip()
    node_name = event_copy.get("node")
    payload = event_copy.get("payload") if isinstance(event_copy.get("payload"), dict) else {}
    status_message = event_copy.get("status_message")
    if not isinstance(status_message, str):
        status_message = payload.get("status_message")
    status_message = str(status_message or "").strip()

    node_elapsed_ms: int | None = None
    if isinstance(node_name, str) and node_name:
        entry["current_node"] = node_name

    if event_type == "node_start" and isinstance(node_name, str) and node_name:
        entry["_node_starts"][node_name] = now_perf
        entry["active_node"] = {
            "node": node_name,
            "started_at": now_iso,
            "elapsed_ms": 0,
        }

    if event_type == "node_complete" and isinstance(node_name, str) and node_name:
        node_started_perf = entry["_node_starts"].pop(node_name, None)
        if isinstance(node_started_perf, float):
            node_elapsed_ms = int((now_perf - node_started_perf) * 1000)
        timing_entry = {
            "node": node_name,
            "completed_at": now_iso,
            "elapsed_ms": node_elapsed_ms or 0,
            "status_message": status_message or None,
        }
        if payload:
            timing_entry["payload"] = deepcopy(payload)
        entry["node_timings"].append(timing_entry)
        if entry.get("active_node", {}).get("node") == node_name:
            entry["active_node"] = None

    if event_type in {"waiting_for_user", "session_complete", "session_closed", "error"}:
        entry["active_node"] = None

    if isinstance(entry.get("active_node"), dict):
        active_started_at = entry["active_node"].get("started_at")
        if isinstance(active_started_at, str):
            entry["active_node"]["elapsed_ms"] = int(
                (now_perf - entry["_node_starts"].get(entry["active_node"]["node"], now_perf)) * 1000
            )

    if status_message:
        entry["current_status_message"] = status_message

    enriched_event = {
        **event_copy,
        "session_id": event_copy.get("session_id") or session_id,
        "timestamp": now_iso,
        "elapsed_ms": int((now_perf - entry["_started_perf"]) * 1000),
        "event_index": len(entry["events"]) + 1,
    }
    if node_elapsed_ms is not None:
        enriched_event["node_elapsed_ms"] = node_elapsed_ms

    entry["events"].append(enriched_event)
    entry["updated_at"] = now_iso
    return enriched_event


def get_session_debug_data(session_id: str) -> dict[str, Any] | None:
    entry = _session_debug_store.get(session_id)
    if entry is None:
        return None
    return _public_entry(entry)


def _prompt_entity_terms(prompt: str) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z'-]{1,}", prompt):
        normalized = token.strip("'").lower()
        if len(normalized) < 3 or normalized in _PROMPT_ENTITY_STOPWORDS:
            continue
        if normalized not in terms:
            terms.append(normalized)
    return terms


def _extract_prompt_hits(
    segments: list[dict[str, Any]],
    entity_terms: list[str],
) -> list[dict[str, Any]]:
    if not entity_terms:
        return []

    hits: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, float, float, str]] = set()
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        start = segment.get("start")
        end = segment.get("end")
        start_sec = float(start) if isinstance(start, (int, float)) else -1.0
        end_sec = float(end) if isinstance(end, (int, float)) else -1.0
        for term in entity_terms:
            if not re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE):
                continue
            hit_key = (term, start_sec, end_sec, text)
            if hit_key in seen_keys:
                continue
            seen_keys.add(hit_key)
            hits.append(
                {
                    "term": term,
                    "start": None if start_sec < 0 else start_sec,
                    "end": None if end_sec < 0 else end_sec,
                    "text": text,
                }
            )
    return hits


def _analysis_excerpt(
    *,
    full_text: str,
    prompt_hits: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> str:
    if prompt_hits:
        lines = []
        for hit in prompt_hits[:8]:
            start = f"{hit['start']:.3f}" if isinstance(hit.get("start"), float) else "?"
            end = f"{hit['end']:.3f}" if isinstance(hit.get("end"), float) else "?"
            lines.append(f"{start}-{end}: {hit.get('text')}")
        return "\n".join(lines)

    if full_text.strip():
        return full_text[:800]

    preview_lines = []
    for segment in segments[:8]:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        preview_lines.append(text)
    return "\n".join(preview_lines)


def _clip_state_lookup(state: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(state, dict):
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for clip in state.get("clips", []):
        if not isinstance(clip, dict):
            continue
        clip_id = str(clip.get("clip_id") or "").strip()
        if clip_id:
            lookup[clip_id] = clip
    return lookup


def build_session_debug_snapshot(
    *,
    session_id: int,
    session_payload: dict[str, Any] | None,
    state: dict[str, Any] | None,
) -> dict[str, Any]:
    debug_entry = get_session_debug_data(str(session_id)) or {}
    effective_state = deepcopy(state) if isinstance(state, dict) else debug_entry.get("latest_state") or {}
    payload = session_payload if isinstance(session_payload, dict) else {}
    clip_state_lookup = _clip_state_lookup(effective_state)
    prompt_terms = _prompt_entity_terms(str(effective_state.get("user_prompt") or ""))

    transcript_debug_items: list[dict[str, Any]] = []
    for video in payload.get("videos", []):
        if not isinstance(video, dict):
            continue
        clip_id = str(video.get("clip_id") or "").strip()
        clip_state = clip_state_lookup.get(clip_id, {})
        transcript_segments = [segment for segment in video.get("transcript_segments", []) if isinstance(segment, dict)]
        video_report = video.get("video_report") if isinstance(video.get("video_report"), dict) else {}
        clip_meta = video.get("clip_meta") if isinstance(video.get("clip_meta"), dict) else {}
        prompt_hits = _extract_prompt_hits(transcript_segments, prompt_terms)
        full_text = str(video.get("transcript_full_text") or "")

        transcript_debug_items.append(
            {
                "clip_id": clip_id,
                "local_key": video.get("local_key"),
                "file_name": video.get("file_name"),
                "summary": clip_state.get("summary") or "",
                "metadata": deepcopy(clip_state.get("metadata") or {}),
                "transcript_cached": bool(clip_state.get("transcript_cached")),
                "transcript_cache_key": clip_state.get("transcript_cache_key"),
                "segment_count": len(transcript_segments),
                "full_text": full_text,
                "analysis_excerpt": _analysis_excerpt(
                    full_text=full_text,
                    prompt_hits=prompt_hits,
                    segments=transcript_segments,
                ),
                "segments": transcript_segments,
                "prompt_hits": prompt_hits,
                "key_phrases": deepcopy(video_report.get("key_phrases") or []),
                "video_report": deepcopy(video_report),
                "clip_meta": deepcopy(clip_meta),
            }
        )

    timings = {
        "started_at": debug_entry.get("started_at"),
        "updated_at": debug_entry.get("updated_at"),
        "total_elapsed_ms": debug_entry.get("total_elapsed_ms", 0),
        "active_node": debug_entry.get("active_node"),
        "node_timings": debug_entry.get("node_timings", []),
        "event_count": len(debug_entry.get("events", [])),
    }

    return {
        "session_id": session_id,
        "session": {
            "session_name": payload.get("session_name"),
            "session_status": payload.get("session_status"),
            "project_id": payload.get("project_id"),
            "project_name": payload.get("project_name"),
            "uploaded_count": payload.get("uploaded_count"),
            "pending_clip_count": payload.get("pending_clip_count"),
            "settled_clip_count": payload.get("settled_clip_count"),
            "ready_for_websocket": payload.get("ready_for_websocket"),
        },
        "prompt": effective_state.get("user_prompt") if isinstance(effective_state, dict) else None,
        "prompt_terms": prompt_terms,
        "current_node": debug_entry.get("current_node"),
        "current_status_message": debug_entry.get("current_status_message")
        or (
            effective_state.get("status_message")
            if isinstance(effective_state, dict)
            else None
        ),
        "timings": timings,
        "events": debug_entry.get("events", []),
        "state": effective_state,
        "transcripts": transcript_debug_items,
    }

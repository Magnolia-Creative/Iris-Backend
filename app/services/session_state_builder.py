from typing import Any

from app.automake.state import ClipState, SessionGraphState


def _build_summary(full_text: str, fallback_name: str | None) -> str:
    cleaned = " ".join(full_text.split())
    if cleaned:
        return cleaned[:280]
    if fallback_name:
        return f"Clip from {fallback_name}"
    return "Clip transcript not yet summarized."


def _infer_duration(video_payload: dict[str, Any]) -> float | None:
    clip_meta = video_payload.get("clip_meta") if isinstance(video_payload.get("clip_meta"), dict) else {}
    video_report = (
        video_payload.get("video_report") if isinstance(video_payload.get("video_report"), dict) else {}
    )
    for value in (
        clip_meta.get("duration_seconds"),
        clip_meta.get("duration"),
        video_report.get("duration_seconds"),
        video_report.get("duration"),
    ):
        if isinstance(value, (int, float)):
            return float(value)

    transcript_segments = video_payload.get("transcript_segments") or []
    max_end = 0.0
    for segment in transcript_segments:
        if not isinstance(segment, dict):
            continue
        end = segment.get("end")
        if isinstance(end, (int, float)):
            max_end = max(max_end, float(end))
    return max_end or None


def build_initial_state(
    *,
    session_id: str,
    ingest_result: dict[str, Any],
    user_prompt: str,
) -> SessionGraphState:
    return build_initial_state_from_session_payload(
        session_id=session_id,
        session_payload=ingest_result,
        user_prompt=user_prompt,
    )


def build_initial_state_from_session_payload(
    *,
    session_id: str,
    session_payload: dict[str, Any],
    user_prompt: str,
) -> SessionGraphState:
    clips: list[ClipState] = []
    for video_payload in session_payload.get("videos", []):
        if not isinstance(video_payload, dict):
            continue
        transcript_id_raw = video_payload.get("transcript_id")
        if transcript_id_raw is None:
            continue
        clip_id = str(video_payload.get("clip_id"))
        transcript_id = int(transcript_id_raw)
        full_text = video_payload.get("transcript_full_text") or ""
        summary = _build_summary(full_text, video_payload.get("file_name"))
        clips.append(
            {
                "clip_id": clip_id,
                "local_key": video_payload.get("local_key"),
                "transcript_id": transcript_id,
                "summary": summary,
                "metadata": {
                    "file_name": video_payload.get("file_name"),
                    "mime_type": video_payload.get("mime_type"),
                    "extension": video_payload.get("extension"),
                    "file_size_bytes": video_payload.get("file_size_bytes"),
                    "segment_count": len(video_payload.get("transcript_segments") or []),
                    "duration_seconds": _infer_duration(video_payload),
                    "clip_meta": video_payload.get("clip_meta") or {},
                    "video_report": video_payload.get("video_report") or {},
                },
                "transcript_cached": False,
                "transcript_cache_key": None,
            }
        )

    return {
        "session_id": session_id,
        "project_id": int(session_payload["project_id"]),
        "user_prompt": user_prompt,
        "clips": clips,
        "next_action": None,
        "retrieval_plan": None,
        "edit_plan": None,
        "cleanup_plan": None,
        "timeline": [],
        "timeline_notes": [],
        "waiting_for_user": False,
        "iteration_count": 0,
        "notes": [
            f"Initialized session {session_id} for project {session_payload['project_id']}."
        ],
        "errors": [],
    }

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import models
from app.services.clip_task_registry import clip_task_registry
from app.services.sentence_transcript_payload import intent_jsonb_from_sentence_api_result


async def get_transcript_payload(
    db: AsyncSession, transcript_id: int
) -> dict[str, Any] | None:
    result = await db.execute(
        select(models.Transcript).where(models.Transcript.id == transcript_id)
    )
    transcript = result.scalar_one_or_none()
    if transcript is None or not isinstance(transcript.transcript, dict):
        return None
    return transcript.transcript


async def get_sentence_upload_transcript_payload(
    db: AsyncSession, transcript_id: str
) -> dict[str, Any] | None:
    try:
        uid = uuid.UUID(str(transcript_id).strip())
    except (ValueError, TypeError, AttributeError):
        return None
    result = await db.execute(
        select(models.SentenceUploadTranscript).where(models.SentenceUploadTranscript.id == uid)
    )
    row = result.scalar_one_or_none()
    if row is None or not isinstance(row.transcript, dict):
        return None
    return row.transcript


async def insert_sentence_upload_transcript(db: AsyncSession, result: dict[str, Any]) -> str:
    payload = intent_jsonb_from_sentence_api_result(result)
    row = models.SentenceUploadTranscript(transcript=payload)
    db.add(row)
    await db.flush()
    out_id = str(row.id)
    await db.commit()
    return out_id


async def get_transcripts_for_clips(
    db: AsyncSession, clip_ids: list[int]
) -> dict[int, dict[str, Any]]:
    if not clip_ids:
        return {}

    result = await db.execute(
        select(models.Clip)
        .options(selectinload(models.Clip.transcript))
        .where(models.Clip.id.in_(clip_ids))
    )
    clips = result.scalars().all()
    payloads: dict[int, dict[str, Any]] = {}
    for clip in clips:
        transcript = clip.transcript
        if transcript is None or not isinstance(transcript.transcript, dict):
            continue
        payloads[int(clip.id)] = transcript.transcript
    return payloads


async def get_persisted_session_data(
    db: AsyncSession,
    session_id: int,
    *,
    include_ingest_details: bool = False,
) -> dict[str, Any] | None:
    result = await db.execute(
        select(models.Session)
        .options(selectinload(models.Session.clips).selectinload(models.Clip.transcript))
        .where(models.Session.id == session_id)
    )
    session = result.scalar_one_or_none()
    if session is None:
        return None

    videos: list[dict[str, Any]] = []
    for index, clip in enumerate(sorted(session.clips, key=lambda c: int(c.id)), start=1):
        transcript_record = clip.transcript
        transcript_payload = (
            transcript_record.transcript
            if transcript_record and isinstance(transcript_record.transcript, dict)
            else {}
        )
        video_payload = {
            "index": index,
            "session_id": int(session.id),
            "project_id": int(clip.project_id),
            "clip_id": int(clip.id),
            "transcript_id": int(transcript_record.id) if transcript_record else None,
            "local_key": clip.local_key,
            "file_name": clip.file_name,
            "mime_type": clip.mime_type,
            "extension": transcript_payload.get("extension"),
            "processing_status": clip.processing_status,
            "processing_error": clip.processing_error,
        }

        if include_ingest_details:
            transcript_segments = transcript_payload.get("segments")
            if not isinstance(transcript_segments, list):
                transcript_segments = []
            clip_meta = transcript_payload.get("clip_meta")
            if not isinstance(clip_meta, dict):
                clip_meta = {}
            video_report = transcript_payload.get("video_report")
            if video_report is not None and not isinstance(video_report, dict):
                video_report = {}

            video_payload.update(
                {
                    "file_size_bytes": clip.file_size_bytes,
                    "transcript_segments": transcript_segments,
                    "transcript_full_text": transcript_payload.get("full_text") or "",
                    "video_report": video_report or {},
                    "clip_meta": clip_meta,
                }
            )

        videos.append(video_payload)

    pending_clip_count = await clip_task_registry.active_count_for_session(int(session.id))
    settled_clip_count = sum(
        1 for video in videos if video.get("processing_status") in {"ready", "failed", "cancelled"}
    )
    ready_clip_count = sum(1 for video in videos if video.get("processing_status") == "ready")
    has_failures = any(video.get("processing_status") == "failed" for video in videos)
    ready_for_websocket = pending_clip_count == 0 and ready_clip_count > 0 and not has_failures
    session_status = session.status
    if pending_clip_count > 0:
        session_status = "processing"
    elif has_failures:
        session_status = "attention"
    elif ready_clip_count > 0:
        session_status = "ready"
    elif videos:
        session_status = "created"

    return {
        "session_id": int(session.id),
        "session_name": session.name,
        "session_status": session_status,
        "project_id": int(videos[0]["project_id"]) if videos else None,
        "uploaded_count": len(videos),
        "pending_clip_count": pending_clip_count,
        "settled_clip_count": settled_clip_count,
        "ready_for_websocket": ready_for_websocket,
        "videos": videos,
    }

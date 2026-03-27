from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app import models


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
    db: AsyncSession, session_id: int
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
        transcript_segments = transcript_payload.get("segments")
        if not isinstance(transcript_segments, list):
            transcript_segments = []
        clip_meta = transcript_payload.get("clip_meta")
        if not isinstance(clip_meta, dict):
            clip_meta = {}
        video_report = transcript_payload.get("video_report")
        if video_report is not None and not isinstance(video_report, dict):
            video_report = {}

        videos.append(
            {
                "index": index,
                "session_id": int(session.id),
                "project_id": int(clip.project_id),
                "clip_id": int(clip.id),
                "transcript_id": int(transcript_record.id) if transcript_record else None,
                "file_name": clip.file_name,
                "mime_type": clip.mime_type,
                "extension": transcript_payload.get("extension"),
                "file_size_bytes": clip.file_size_bytes,
                "transcript_segments": transcript_segments,
                "transcript_full_text": transcript_payload.get("full_text") or "",
                "video_report": video_report or {},
                "clip_meta": clip_meta,
            }
        )

    return {
        "session_id": int(session.id),
        "session_name": session.name,
        "session_status": session.status,
        "project_id": int(videos[0]["project_id"]) if videos else None,
        "uploaded_count": len(videos),
        "videos": videos,
    }

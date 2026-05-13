import asyncio
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Any

from fastapi import HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import SessionLocal
from app.database import models
from app.services.clip_task_registry import clip_task_registry
from app.services.transcript_cache import delete_cached_transcript
from app.services.transcript_normalize import (
    json_safe_value,
    normalize_transcript_segments,
    segments_to_full_text,
)
from app.config import settings
from app.services.clip_embedding_store import delete_embeddings_for_clip
from app.services import gemini_embedding
from app.services.transcript_store import (
    get_persisted_project_data,
    get_persisted_session_data,
)
from app.services.transcription import print_received_transcript, transcribe_upload_for_ingest
from app.services.vector_index_runner import run_clip_vector_index
from app.services.visual_frame_payload import VisualFrameChunk


logger = logging.getLogger(__name__)


async def create_project(
    db: AsyncSession,
    *,
    name: str | None = None,
) -> dict[str, Any]:
    resolved_name = name or f"project-{datetime.utcnow().isoformat(timespec='seconds')}"
    project = models.Project(name=resolved_name)
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return {
        "project_id": int(project.id),
        "project_name": project.name,
    }


async def create_agent_session(
    db: AsyncSession,
    *,
    session_name: str | None = None,
    project_name: str | None = None,
) -> dict[str, Any]:
    resolved_session_name = (
        session_name
        or project_name
        or f"ingest-{datetime.utcnow().isoformat(timespec='seconds')}"
    )
    resolved_project_name = project_name or f"project-{resolved_session_name}"

    session = models.Session(name=resolved_session_name, status="created")
    project = models.Project(name=resolved_project_name)
    db.add(session)
    db.add(project)
    await db.flush()
    session.project_id = int(project.id)
    await db.commit()
    await db.refresh(session)
    await db.refresh(project)

    return {
        "session_id": int(session.id),
        "session_name": session.name,
        "session_status": session.status,
        "project_id": int(project.id),
        "project_name": project.name,
        "uploaded_count": 0,
        "pending_clip_count": 0,
        "settled_clip_count": 0,
        "ready_for_websocket": False,
        "videos": [],
    }


async def create_agent_session_for_project(
    db: AsyncSession,
    *,
    project_id: int,
    session_name: str | None = None,
) -> dict[str, Any]:
    project = await _require_project(db, project_id=project_id)
    resolved_session_name = session_name or project.name or f"session-{project_id}"
    session = models.Session(
        name=resolved_session_name,
        status="created",
        project_id=project_id,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    payload = await get_persisted_project_data(db, project_id, include_ingest_details=False)
    uploaded = payload["uploaded_count"] if payload else 0
    pending = payload["pending_clip_count"] if payload else 0
    settled = payload["settled_clip_count"] if payload else 0
    ready_ws = payload["ready_for_websocket"] if payload else False

    return {
        "session_id": int(session.id),
        "session_name": session.name,
        "session_status": session.status,
        "project_id": project_id,
        "project_name": project.name,
        "uploaded_count": uploaded,
        "pending_clip_count": pending,
        "settled_clip_count": settled,
        "ready_for_websocket": ready_ws,
        "videos": (payload or {}).get("videos") or [],
    }


async def ingest_session_clips(
    db: AsyncSession,
    videos: list[UploadFile],
    local_keys: list[str],
    session_name: str | None = None,
    project_name: str | None = None,
) -> dict[str, Any]:
    created = await create_agent_session(
        db,
        session_name=session_name,
        project_name=project_name,
    )
    return await process_project_clips(
        db,
        project_id=created["project_id"],
        session_id=created["session_id"],
        videos=videos,
        local_keys=local_keys,
        visual_frames_by_local_key=None,
    )


async def schedule_transcript_processing_tasks(
    *,
    project_id: int,
    pending_rows: list[tuple[dict[str, Any], models.Clip]],
) -> None:
    for prepared_video, clip in pending_rows:
        task = asyncio.create_task(
            _run_clip_transcription(
                project_id=project_id,
                local_key=prepared_video["local_key"],
                clip_id=int(clip.id),
                video_bytes=prepared_video["video_bytes"],
                extension=prepared_video.get("extension") or "",
                index=prepared_video["index"],
                file_name=prepared_video["video"].filename,
                mime_type=prepared_video["video"].content_type,
                clip_correlation_id=prepared_video.get("clip_correlation_id"),
            )
        )
        await clip_task_registry.register(
            project_id=project_id,
            local_key=prepared_video["local_key"],
            task=task,
        )


def schedule_vector_index_tasks(
    *,
    project_id: int,
    session_id: int | None,
    pending_rows: list[tuple[dict[str, Any], models.Clip]],
    visual_frames_by_local_key: dict[str, list[VisualFrameChunk]] | None,
) -> int:
    if not settings.semantic_indexing_enabled or not gemini_embedding.gemini_configured():
        logger.info(
            "[vector_index] skip scheduling project_id=%s session_id=%s reason=%s",
            project_id,
            session_id,
            "SEMANTIC_INDEXING_ENABLED=false"
            if not settings.semantic_indexing_enabled
            else "GEMINI_API_KEY missing",
        )
        return 0
    scheduled = 0
    for prepared_video, clip in pending_rows:
        local_key = prepared_video["local_key"]
        frames = (visual_frames_by_local_key or {}).get(local_key)
        n_visual = len(frames) if frames else 0
        audio_n = len(prepared_video["video_bytes"] or b"")
        logger.info(
            "[vector_index] scheduling task project_id=%s session_id=%s clip_id=%s local_key=%s "
            "audio_bytes=%s visual_frame_payloads=%s",
            project_id,
            session_id,
            int(clip.id),
            local_key,
            audio_n,
            n_visual,
        )
        asyncio.create_task(
            run_clip_vector_index(
                project_id=project_id,
                session_id=session_id,
                clip_id=int(clip.id),
                local_key=local_key,
                audio_bytes=prepared_video["video_bytes"],
                audio_extension=prepared_video.get("extension") or "",
                visual_frames=frames,
            )
        )
        scheduled += 1
    logger.info(
        "[vector_index] scheduled %s clip(s) for embedding project_id=%s session_id=%s",
        scheduled,
        project_id,
        session_id,
    )
    return scheduled


async def process_project_clips(
    db: AsyncSession,
    *,
    project_id: int,
    session_id: int | None,
    videos: list[UploadFile],
    local_keys: list[str],
    visual_frames_by_local_key: dict[str, list[VisualFrameChunk]] | None = None,
) -> dict[str, Any]:
    if not 1 <= len(videos) <= 10:
        raise HTTPException(
            status_code=400, detail="Upload between 1 and 10 videos per request."
        )
    if len(local_keys) != len(videos):
        raise HTTPException(
            status_code=400,
            detail="Each uploaded video must include one local_key.",
        )
    if len(set(local_keys)) != len(local_keys):
        raise HTTPException(status_code=400, detail="Each local_key must be unique within a batch.")

    request_t0 = time.perf_counter()
    logger.info(
        "[INGEST] project_id=%s session_id=%s clip ingest started: %d video(s) local_keys=%s",
        project_id,
        session_id,
        len(videos),
        local_keys,
    )
    project = await _require_project(db, project_id=project_id)
    session: models.Session | None = None
    if session_id is not None:
        session = await _require_session(db, session_id=session_id)
        if session.project_id is not None and int(session.project_id) != project_id:
            raise HTTPException(
                status_code=400,
                detail="project_id does not match the session's linked project.",
            )
        session.status = "processing"

    vector_scheduled = 0
    try:
        prepared_videos: list[dict[str, Any]] = []
        t_prepare = time.perf_counter()
        for index, (video, local_key) in enumerate(zip(videos, local_keys, strict=True), start=1):
            existing_clip = await _get_clip_by_project_and_local_key(
                db,
                project_id=int(project.id),
                local_key=local_key,
            )
            if existing_clip is not None and existing_clip.processing_status == "ready":
                logger.info(
                    "[INGEST] Skipping already-ready clip project_id=%s local_key=%s clip_id=%s",
                    project.id,
                    local_key,
                    existing_clip.id,
                )
                continue
            if existing_clip is not None:
                await _delete_clip_artifacts(db, existing_clip)

            t_read = time.perf_counter()
            video.file.seek(0, 2)
            file_size_bytes = video.file.tell()
            video.file.seek(0)
            video_bytes = await video.read()
            read_s = time.perf_counter() - t_read

            extension = Path(video.filename or "").suffix.lower()
            prepared_videos.append(
                {
                    "index": index,
                    "video": video,
                    "local_key": local_key,
                    "file_size_bytes": file_size_bytes,
                    "video_bytes": video_bytes,
                    "extension": extension,
                    "clip_correlation_id": local_key,
                }
            )
            logger.info(
                "[INGEST] Prepared video %d file=%s mime=%s size=%d read_upload_s=%.3f",
                index,
                video.filename,
                video.content_type,
                file_size_bytes,
                read_s,
            )
        logger.info(
            "[INGEST] All uploads buffered duration_s=%.3f",
            time.perf_counter() - t_prepare,
        )

        logger.info("[INGEST] Scheduling %d background transcription task(s)", len(prepared_videos))
        pending_rows: list[tuple[dict[str, Any], models.Clip]] = []
        for prepared_video in prepared_videos:
            clip = models.Clip(
                project_id=int(project.id),
                session_id=int(session.id) if session is not None else None,
                title=Path(prepared_video["video"].filename or f"video-{prepared_video['index']}").stem,
                file_name=prepared_video["video"].filename,
                local_key=prepared_video["local_key"],
                mime_type=prepared_video["video"].content_type,
                file_size_bytes=prepared_video["file_size_bytes"],
                processing_status="processing",
                processing_started_at=datetime.utcnow(),
            )
            db.add(clip)
            await db.flush()
            pending_rows.append((prepared_video, clip))

        await db.flush()
        await db.commit()

        await schedule_transcript_processing_tasks(
            project_id=int(project.id),
            pending_rows=pending_rows,
        )
        vector_scheduled = schedule_vector_index_tasks(
            project_id=int(project.id),
            session_id=int(session.id) if session is not None else None,
            pending_rows=pending_rows,
            visual_frames_by_local_key=visual_frames_by_local_key,
        )
    except HTTPException:
        await db.rollback()
        raise
    except Exception as exc:
        await db.rollback()
        logger.exception("[INGEST] Unexpected ingest failure")
        raise HTTPException(status_code=500, detail="Ingest failed; check server logs for details.") from exc

    if session is not None:
        result = await _build_session_payload(
            db,
            session_id=int(session.id),
            fallback_project_id=int(project.id),
            fallback_project_name=project.name,
        )
    else:
        result = await _build_project_payload(db, project_id=int(project.id))

    if not settings.semantic_indexing_enabled:
        result["vector_index"] = {"status": "disabled", "reason": "SEMANTIC_INDEXING_ENABLED=false"}
    elif not gemini_embedding.gemini_configured():
        result["vector_index"] = {"status": "disabled", "reason": "GEMINI_API_KEY missing"}
    else:
        result["vector_index"] = {
            "status": "scheduled",
            "scheduled_clip_count": vector_scheduled,
        }
    await db.commit()
    logger.info(
        "[INGEST] project_id=%s session_id=%s ingest completed uploaded_count=%s "
        "vector_index=%s total_duration_s=%.3f",
        project.id,
        session_id,
        result["uploaded_count"],
        result.get("vector_index"),
        time.perf_counter() - request_t0,
    )
    return result


async def cancel_clip_processing(
    db: AsyncSession,
    *,
    project_id: int,
    local_key: str,
    session_id: int | None = None,
) -> dict[str, Any]:
    clip = await _get_clip_by_project_and_local_key(
        db, project_id=project_id, local_key=local_key
    )
    task_cancelled = await clip_task_registry.cancel(project_id=project_id, local_key=local_key)
    deleted_clip_id: int | None = None

    if clip is not None:
        deleted_clip_id = int(clip.id)
        await _delete_clip_artifacts(db, clip)
        await db.commit()
    else:
        await db.rollback()

    if session_id is not None:
        session = await _require_session(db, session_id=session_id)
        if session.project_id is not None and int(session.project_id) != project_id:
            raise HTTPException(
                status_code=400,
                detail="project_id does not match the session's linked project.",
            )
        payload = await _build_session_payload(
            db,
            session_id=session_id,
            fallback_project_id=project_id,
            fallback_project_name=None,
        )
        session.status = payload["session_status"]
        await db.commit()
        return {
            "session_id": session_id,
            "project_id": project_id,
            "local_key": local_key,
            "task_cancelled": task_cancelled,
            "deleted_clip_id": deleted_clip_id,
            "session_status": payload["session_status"],
            "ready_for_websocket": payload["ready_for_websocket"],
        }

    proj_payload = await get_persisted_project_data(db, project_id, include_ingest_details=False)
    return {
        "session_id": None,
        "project_id": project_id,
        "local_key": local_key,
        "task_cancelled": task_cancelled,
        "deleted_clip_id": deleted_clip_id,
        "session_status": (proj_payload or {}).get("session_status") or "created",
        "ready_for_websocket": (proj_payload or {}).get("ready_for_websocket") or False,
    }


async def _run_clip_transcription(
    *,
    project_id: int,
    local_key: str,
    clip_id: int,
    video_bytes: bytes,
    extension: str,
    index: int,
    file_name: str | None,
    mime_type: str | None,
    clip_correlation_id: str | None,
) -> None:
    try:
        result = await transcribe_upload_for_ingest(
            video_bytes,
            extension=extension,
            index=index,
            file_name=file_name,
            clip_correlation_id=clip_correlation_id,
        )
        print_received_transcript(
            source=f"ingest clip_id={clip_id}",
            transcript_id=clip_correlation_id,
            segments=result.get("transcript"),
        )
        async with SessionLocal() as db:
            await _persist_transcription_result(
                db,
                clip_id=clip_id,
                file_name=file_name,
                mime_type=mime_type,
                extension=extension,
                transcription_result=result,
            )
    except asyncio.CancelledError:
        async with SessionLocal() as db:
            await _mark_clip_cancelled(db, clip_id=clip_id)
        raise
    except Exception as exc:  # pragma: no cover - exercised through route behavior
        logger.exception(
            "[INGEST] Background transcription failed clip_id=%s file=%s",
            clip_id,
            file_name,
            exc_info=exc,
        )
        async with SessionLocal() as db:
            await _mark_clip_failed(db, clip_id=clip_id, error_message=str(exc))
    finally:
        await clip_task_registry.pop(project_id=project_id, local_key=local_key)


async def _persist_transcription_result(
    db: AsyncSession,
    *,
    clip_id: int,
    file_name: str | None,
    mime_type: str | None,
    extension: str,
    transcription_result: dict[str, Any],
) -> None:
    clip = await _get_clip_by_id(db, clip_id=clip_id)
    if clip is None:
        await db.rollback()
        return

    raw_segments = transcription_result.get("transcript") or []
    if not isinstance(raw_segments, list):
        raw_segments = []
    video_report = transcription_result.get("video_report")
    clip_meta = transcription_result.get("meta")
    if clip_meta is not None and not isinstance(clip_meta, dict):
        clip_meta = {}

    segment_dicts = [s for s in raw_segments if isinstance(s, dict)]
    transcript_segments = normalize_transcript_segments(segment_dicts)
    full_text = segments_to_full_text(transcript_segments)
    safe_card = json_safe_value(video_report) if video_report is not None else None
    safe_meta = json_safe_value(clip_meta) if clip_meta else {}

    transcript_payload = {
        "source_file": file_name,
        "mime_type": mime_type,
        "extension": extension,
        "segments": transcript_segments,
        "full_text": full_text,
        "video_report": safe_card,
        "clip_meta": safe_meta,
    }

    clip.transcript = models.Transcript(transcript=transcript_payload)
    clip.processing_status = "ready"
    clip.processing_error = None
    clip.provider_job_id = _extract_provider_job_id(safe_meta, safe_card)
    clip.processing_completed_at = datetime.utcnow()
    await db.commit()
    logger.info(
        "[INGEST] Saved background transcription file=%s clip_id=%s transcript_id=%s segments=%d",
        file_name,
        clip.id,
        clip.transcript.id if clip.transcript is not None else None,
        len(transcript_segments),
    )


async def _mark_clip_failed(
    db: AsyncSession,
    *,
    clip_id: int,
    error_message: str,
) -> None:
    clip = await _get_clip_by_id(db, clip_id=clip_id)
    if clip is None:
        await db.rollback()
        return
    clip.processing_status = "failed"
    clip.processing_error = error_message
    clip.processing_completed_at = datetime.utcnow()
    await db.commit()


async def _mark_clip_cancelled(db: AsyncSession, *, clip_id: int) -> None:
    clip = await _get_clip_by_id(db, clip_id=clip_id)
    if clip is None:
        await db.rollback()
        return
    clip.processing_status = "cancelled"
    clip.processing_cancelled_at = datetime.utcnow()
    await db.commit()


async def _build_project_payload(db: AsyncSession, *, project_id: int) -> dict[str, Any]:
    payload = await get_persisted_project_data(db, project_id, include_ingest_details=False)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found.")
    return payload


async def _build_session_payload(
    db: AsyncSession,
    *,
    session_id: int,
    fallback_project_id: int | None,
    fallback_project_name: str | None,
) -> dict[str, Any]:
    payload = await get_persisted_session_data(db, session_id=session_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    payload["project_id"] = payload.get("project_id") or fallback_project_id
    if fallback_project_name is not None:
        payload["project_name"] = fallback_project_name
    else:
        project = (
            await _require_project(db, project_id=int(payload["project_id"]))
            if payload.get("project_id")
            else None
        )
        payload["project_name"] = project.name if project is not None else None

    session = await _require_session(db, session_id=session_id)
    session.status = payload["session_status"]
    await db.flush()
    return payload


async def _require_session(db: AsyncSession, *, session_id: int) -> models.Session:
    result = await db.execute(select(models.Session).where(models.Session.id == session_id))
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return session


async def _require_project(db: AsyncSession, *, project_id: int) -> models.Project:
    result = await db.execute(select(models.Project).where(models.Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found.")
    return project


async def _get_clip_by_project_and_local_key(
    db: AsyncSession,
    *,
    project_id: int,
    local_key: str,
) -> models.Clip | None:
    result = await db.execute(
        select(models.Clip).where(
            models.Clip.project_id == project_id,
            models.Clip.local_key == local_key,
        )
    )
    return result.scalar_one_or_none()


async def _get_clip_by_id(db: AsyncSession, *, clip_id: int) -> models.Clip | None:
    result = await db.execute(select(models.Clip).where(models.Clip.id == clip_id))
    return result.scalar_one_or_none()


async def _delete_clip_artifacts(db: AsyncSession, clip: models.Clip) -> None:
    logger.info(
        "[embeddings] deleting clip artifacts project_id=%s clip_id=%s local_key=%s",
        int(clip.project_id),
        int(clip.id),
        clip.local_key,
    )
    if clip.session_id is not None:
        await delete_cached_transcript(str(clip.session_id), str(clip.id))
    await delete_embeddings_for_clip(db, clip_id=int(clip.id))
    await db.delete(clip)
    await db.flush()


def _extract_provider_job_id(
    clip_meta: dict[str, Any],
    video_report: dict[str, Any] | None,
) -> str | None:
    candidate = clip_meta.get("assemblyai_transcript_id")
    if isinstance(candidate, str) and candidate:
        return candidate
    candidate = clip_meta.get("clip_id")
    if isinstance(candidate, str) and candidate:
        return candidate
    if isinstance(video_report, dict):
        candidate = video_report.get("transcript_id")
        if isinstance(candidate, str) and candidate:
            return candidate
    return None

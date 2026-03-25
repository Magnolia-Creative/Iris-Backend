from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime
import time
import uuid
from decimal import Decimal
import logging
from typing import Any

from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Base, engine, get_db
from app.services.transcription import transcribe_upload_for_ingest
from app.services.transcript_normalize import (
    json_safe_value,
    normalize_transcript_segments,
    segments_to_full_text,
)
from app import models  # noqa: F401


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


class ClipCreate(BaseModel):
    title: str | None = None
    file_name: str | None = None
    source_url: str | None = None
    mime_type: str | None = None
    codec: str | None = None
    frame_rate: Decimal | None = None
    duration_seconds: Decimal | None = None
    width: int | None = None
    height: int | None = None
    file_size_bytes: int | None = None
    language_code: str | None = None
    captured_at: datetime | None = None


class IngestCreate(BaseModel):
    project_name: str
    clip: ClipCreate
    transcript: dict[str, Any]
    summary: str


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Keep simple table creation for local development; use Alembic for production migrations.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/db-health")
async def db_health(db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("SELECT 1"))
    return {"database": "ok", "result": result.scalar_one()}


@app.post("/ingest")
async def ingest_videos(
    videos: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    project_name: str | None = None,
):
    if not 1 <= len(videos) <= 10:
        raise HTTPException(
            status_code=400, detail="Upload between 1 and 10 videos per request."
        )

    request_t0 = time.perf_counter()
    logger.info("[INGEST] Received request with %d video(s)", len(videos))
    video_details: list[dict[str, Any]] = []
    resolved_project_name = project_name or f"ingest-{datetime.utcnow().isoformat(timespec='seconds')}"

    try:
        t_project = time.perf_counter()
        project = models.Project(name=resolved_project_name)
        db.add(project)
        await db.flush()
        logger.info(
            "[INGEST] Created project id=%s name=%s duration_s=%.3f",
            project.id,
            project.name,
            time.perf_counter() - t_project,
        )

        prepared_videos: list[dict[str, Any]] = []
        t_prepare = time.perf_counter()
        for index, video in enumerate(videos, start=1):
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
                    "file_size_bytes": file_size_bytes,
                    "video_bytes": video_bytes,
                    "extension": extension,
                    "clip_correlation_id": uuid.uuid4().hex,
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

        logger.info("[INGEST] Launching %d concurrent transcription task(s)", len(prepared_videos))
        t_transcribe_wall = time.perf_counter()
        transcription_tasks = [
            transcribe_upload_for_ingest(
                pv["video_bytes"],
                extension=pv.get("extension") or "",
                index=pv["index"],
                file_name=pv["video"].filename,
                clip_correlation_id=pv.get("clip_correlation_id"),
            )
            for pv in prepared_videos
        ]
        transcription_results = await asyncio.gather(*transcription_tasks, return_exceptions=True)
        logger.info(
            "[INGEST] Transcription batch wall_duration_s=%.3f (concurrent)",
            time.perf_counter() - t_transcribe_wall,
        )

        t_persist = time.perf_counter()
        pending_rows: list[tuple[dict[str, Any], models.Clip, list[dict[str, Any]]]] = []

        for prepared_video, transcription_result in zip(
            prepared_videos, transcription_results, strict=True
        ):
            index = prepared_video["index"]
            video = prepared_video["video"]
            file_size_bytes = prepared_video["file_size_bytes"]
            extension = prepared_video["extension"]

            if isinstance(transcription_result, Exception):
                logger.exception(
                    "[INGEST] Transcription failed for video %d file=%s",
                    index,
                    video.filename,
                    exc_info=transcription_result,
                )
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"Transcription failed for {video.filename or f'video-{index}'}; "
                        "check server logs for details."
                    ),
                ) from transcription_result

            if not isinstance(transcription_result, dict):
                raise HTTPException(
                    status_code=502,
                    detail="Transcription returned an unexpected shape; expected object from transcribe_clip.",
                )

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
                "source_file": video.filename,
                "mime_type": video.content_type,
                "extension": extension,
                "segments": transcript_segments,
                "full_text": full_text,
                "video_report": video_report,
                "clip_meta": safe_meta,
            }
            clip = models.Clip(
                project_id=project.id,
                title=Path(video.filename or f"video-{index}").stem,
                file_name=video.filename,
                mime_type=video.content_type,
                file_size_bytes=file_size_bytes,
            )
            clip.transcript = models.Transcript(transcript=transcript_payload)
            db.add(clip)
            pending_rows.append((prepared_video, clip, transcript_segments))

        await db.flush()

        for prepared_video, clip, transcript_segments in pending_rows:
            index = prepared_video["index"]
            video = prepared_video["video"]
            file_size_bytes = prepared_video["file_size_bytes"]
            extension = prepared_video["extension"]
            transcript = clip.transcript
            if transcript is None:
                raise RuntimeError("clip.transcript missing after flush")

            tr = transcript.transcript if isinstance(transcript.transcript, dict) else {}
            full_text = tr.get("full_text") or segments_to_full_text(transcript_segments)
            video_report = tr.get("video_report")
            clip_meta = tr.get("clip_meta") if isinstance(tr.get("clip_meta"), dict) else {}

            metadata = {
                "index": index,
                "project_id": project.id,
                "clip_id": clip.id,
                "transcript_id": transcript.id,
                "file_name": video.filename,
                "mime_type": video.content_type,
                "extension": extension,
                "file_size_bytes": file_size_bytes,
                "transcript_segments": transcript_segments,
                "transcript_full_text": full_text,
                "video_report": video_report,
                "clip_meta": clip_meta,
            }
            video_details.append(metadata)

            logger.info(
                "[INGEST] Saved video %d file=%s clip_id=%s transcript_id=%s segments=%d",
                index,
                video.filename,
                clip.id,
                transcript.id,
                len(transcript_segments),
            )

        logger.info(
            "[INGEST] DB persist (single batch flush) duration_s=%.3f",
            time.perf_counter() - t_persist,
        )
        t_commit = time.perf_counter()
        await db.commit()
        logger.info("[INGEST] DB commit duration_s=%.3f", time.perf_counter() - t_commit)
    except HTTPException:
        await db.rollback()
        raise
    except Exception as exc:
        await db.rollback()
        logger.exception("[INGEST] Unexpected ingest failure")
        raise HTTPException(status_code=500, detail="Ingest failed; check server logs for details.") from exc

    logger.info(
        "[INGEST] Completed request project_id=%s with %d processed video(s) total_duration_s=%.3f",
        project.id,
        len(video_details),
        time.perf_counter() - request_t0,
    )
    return {
        "project_id": project.id,
        "project_name": project.name,
        "uploaded_count": len(video_details),
        "videos": video_details,
    }

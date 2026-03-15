from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
import logging
from typing import Any

from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Base, engine, get_db
from app.services.transcription import extract_and_transcribe_async
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

    logger.info("[INGEST] Received request with %d video(s)", len(videos))
    video_details: list[dict[str, Any]] = []
    resolved_project_name = project_name or f"ingest-{datetime.utcnow().isoformat(timespec='seconds')}"

    try:
        project = models.Project(name=resolved_project_name)
        db.add(project)
        await db.flush()
        logger.info(
            "[INGEST] Created project id=%s name=%s for request",
            project.id,
            project.name,
        )

        for index, video in enumerate(videos, start=1):
            video.file.seek(0, 2)
            file_size_bytes = video.file.tell()
            video.file.seek(0)
            video_bytes = await video.read()

            extension = Path(video.filename or "").suffix.lower()
            logger.info(
                "[INGEST] Processing video %d file=%s mime=%s size=%d",
                index,
                video.filename,
                video.content_type,
                file_size_bytes,
            )
            try:
                transcript_segments = await extract_and_transcribe_async(video_bytes)
            except Exception as exc:
                logger.exception(
                    "[INGEST] Transcription failed for video %d file=%s",
                    index,
                    video.filename,
                )
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"Transcription failed for {video.filename or f'video-{index}'}; "
                        "check server logs for details."
                    ),
                ) from exc

            clip = models.Clip(
                project_id=project.id,
                title=Path(video.filename or f"video-{index}").stem,
                file_name=video.filename,
                mime_type=video.content_type,
                file_size_bytes=file_size_bytes,
            )
            db.add(clip)
            await db.flush()

            transcript_payload = {
                "source_file": video.filename,
                "mime_type": video.content_type,
                "extension": extension,
                "segments": transcript_segments,
            }
            transcript = models.Transcript(
                clip_id=clip.id,
                transcript=transcript_payload,
            )
            db.add(transcript)
            await db.flush()

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

        await db.commit()
    except HTTPException:
        await db.rollback()
        raise
    except Exception as exc:
        await db.rollback()
        logger.exception("[INGEST] Unexpected ingest failure")
        raise HTTPException(status_code=500, detail="Ingest failed; check server logs for details.") from exc

    logger.info(
        "[INGEST] Completed request project_id=%s with %d processed video(s)",
        project.id,
        len(video_details),
    )
    return {
        "project_id": project.id,
        "project_name": project.name,
        "uploaded_count": len(video_details),
        "videos": video_details,
    }

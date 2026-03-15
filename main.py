from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
from typing import Any

from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Base, engine, get_db
from app import models  # noqa: F401


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
async def ingest_videos(videos: list[UploadFile] = File(...)):
    if not 1 <= len(videos) <= 10:
        raise HTTPException(
            status_code=400, detail="Upload between 1 and 10 videos per request."
        )

    video_details: list[dict[str, Any]] = []

    for index, video in enumerate(videos, start=1):
        video.file.seek(0, 2)
        file_size_bytes = video.file.tell()
        video.file.seek(0)

        extension = Path(video.filename or "").suffix.lower()
        metadata = {
            "index": index,
            "file_name": video.filename,
            "mime_type": video.content_type,
            "extension": extension,
            "file_size_bytes": file_size_bytes,
        }
        video_details.append(metadata)

        # Logs each uploaded file's metadata so we can inspect requests quickly.
        print(f"[INGEST] Video {index}: {metadata}")

    # DB additions are temporarily disabled while the new ingest workflow is rebuilt.
    # project = models.Project(name=payload.project_name)
    # db.add(project)
    # await db.flush()
    #
    # clip = models.Clip(project_id=project.id, **payload.clip.model_dump())
    # db.add(clip)
    # await db.flush()
    #
    # transcript = models.Transcript(clip_id=clip.id, transcript=payload.transcript)
    # summary = models.Summary(clip_id=clip.id, summary=payload.summary)
    # db.add_all([transcript, summary])
    # await db.commit()

    return {"uploaded_count": len(video_details), "videos": video_details}

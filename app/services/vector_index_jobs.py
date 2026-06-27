import asyncio
import logging
from typing import Any

from app.config import settings
from app.database import models
from app.services import gemini_embedding
from app.services.vector_index_runner import run_clip_vector_index
from app.services.visual_frame_payload import VisualFrameChunk


logger = logging.getLogger(__name__)


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

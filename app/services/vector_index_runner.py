"""Background multimodal embedding indexing for a clip."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.database import SessionLocal
from app.services import clip_embedding_store, gemini_embedding
from app.services.audio_chunking import ffmpeg_available, slice_audio_to_wav_chunks
from app.services.visual_frame_payload import VisualFrameChunk

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-embedding-2"


async def run_clip_vector_index(
    *,
    project_id: int,
    session_id: int,
    clip_id: int,
    local_key: str,
    audio_bytes: bytes,
    audio_extension: str,
    visual_frames: list[VisualFrameChunk] | None,
) -> None:
    if not gemini_embedding.gemini_configured():
        logger.info("[vector_index] GEMINI_API_KEY missing; skip clip_id=%s", clip_id)
        return

    try:
        async with SessionLocal() as db:
            await clip_embedding_store.delete_embeddings_for_clip(db, clip_id=clip_id)
            await db.commit()

        if ffmpeg_available() and audio_bytes:
            wav_chunks = slice_audio_to_wav_chunks(audio_bytes, suffix=audio_extension or ".m4a")
            for idx, start, end, center, wav_bytes in wav_chunks:
                try:
                    emb = await gemini_embedding.embed_audio_bytes(
                        audio_bytes=wav_bytes, mime_type="audio/wav"
                    )
                except Exception as exc:
                    logger.warning(
                        "[vector_index] audio embed failed clip_id=%s idx=%s: %s",
                        clip_id,
                        idx,
                        exc,
                    )
                    continue
                async with SessionLocal() as db:
                    await clip_embedding_store.insert_embedding_row(
                        db,
                        project_id=project_id,
                        session_id=session_id,
                        clip_id=clip_id,
                        local_key=local_key,
                        modality="audio",
                        chunk_index=idx,
                        start_time_seconds=start,
                        end_time_seconds=end,
                        center_time_seconds=center,
                        embedding=emb,
                        model_name=MODEL_NAME,
                    )
                    await db.commit()

        if visual_frames:
            for frame in visual_frames:
                try:
                    emb = await gemini_embedding.embed_image_bytes(
                        image_bytes=frame.image_bytes,
                        mime_type=frame.mime_type,
                        title=f"clip:{clip_id}",
                    )
                except Exception as exc:
                    logger.warning(
                        "[vector_index] image embed failed clip_id=%s chunk=%s: %s",
                        clip_id,
                        frame.chunk_index,
                        exc,
                    )
                    continue
                async with SessionLocal() as db:
                    await clip_embedding_store.insert_embedding_row(
                        db,
                        project_id=project_id,
                        session_id=session_id,
                        clip_id=clip_id,
                        local_key=local_key,
                        modality="visual_frame",
                        chunk_index=frame.chunk_index,
                        start_time_seconds=frame.start_time_seconds,
                        end_time_seconds=frame.end_time_seconds,
                        center_time_seconds=frame.center_time_seconds,
                        embedding=emb,
                        model_name=MODEL_NAME,
                    )
                    await db.commit()

        logger.info(
            "[vector_index] completed clip_id=%s local_key=%s audio_chunks=%s visual_frames=%s",
            clip_id,
            local_key,
            "yes" if audio_bytes else "no",
            len(visual_frames or []),
        )
    except Exception:
        logger.exception("[vector_index] failed clip_id=%s local_key=%s", clip_id, local_key)

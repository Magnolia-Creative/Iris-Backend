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
    session_id: int | None,
    clip_id: int,
    local_key: str,
    audio_bytes: bytes,
    audio_extension: str,
    visual_frames: list[VisualFrameChunk] | None,
) -> None:
    if not gemini_embedding.gemini_configured():
        logger.info(
            "[vector_index] skip run project_id=%s session_id=%s clip_id=%s local_key=%s reason=no_gemini_key",
            project_id,
            session_id,
            clip_id,
            local_key,
        )
        return

    n_audio = len(audio_bytes or b"")
    n_visual = len(visual_frames or [])
    logger.info(
        "[vector_index] start project_id=%s session_id=%s clip_id=%s local_key=%s "
        "audio_bytes=%s visual_frame_chunks=%s model=%s",
        project_id,
        session_id,
        clip_id,
        local_key,
        n_audio,
        n_visual,
        MODEL_NAME,
    )

    audio_stored = 0
    visual_stored = 0

    try:
        async with SessionLocal() as db:
            await clip_embedding_store.delete_embeddings_for_clip(db, clip_id=clip_id)
            await db.commit()
        logger.info(
            "[vector_index] cleared prior embedding rows project_id=%s clip_id=%s",
            project_id,
            clip_id,
        )

        if ffmpeg_available() and audio_bytes:
            wav_chunks = slice_audio_to_wav_chunks(audio_bytes, suffix=audio_extension or ".m4a")
            logger.info(
                "[vector_index] audio slice pass project_id=%s clip_id=%s wav_chunks=%s ffmpeg_ok=True",
                project_id,
                clip_id,
                len(wav_chunks),
            )
            for idx, start, end, center, wav_bytes in wav_chunks:
                try:
                    emb = await gemini_embedding.embed_audio_bytes(
                        audio_bytes=wav_bytes, mime_type="audio/wav"
                    )
                except Exception as exc:
                    logger.warning(
                        "[vector_index] audio embed failed project_id=%s clip_id=%s chunk_index=%s: %s",
                        project_id,
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
                audio_stored += 1
                logger.debug(
                    "[vector_index] stored audio embedding project_id=%s clip_id=%s "
                    "chunk_index=%s time=%.2f-%.2f center=%.2f dim=%s",
                    project_id,
                    clip_id,
                    idx,
                    start,
                    end,
                    center,
                    len(emb),
                )
            logger.info(
                "[vector_index] audio embedding summary project_id=%s clip_id=%s "
                "wav_chunks=%s rows_stored=%s",
                project_id,
                clip_id,
                len(wav_chunks),
                audio_stored,
            )
        elif audio_bytes and not ffmpeg_available():
            logger.warning(
                "[vector_index] skip audio chunks project_id=%s clip_id=%s reason=ffmpeg_or_ffprobe_missing",
                project_id,
                clip_id,
            )

        if visual_frames:
            logger.info(
                "[vector_index] visual embed pass project_id=%s clip_id=%s frames=%s",
                project_id,
                clip_id,
                len(visual_frames),
            )
            for frame in visual_frames:
                try:
                    emb = await gemini_embedding.embed_image_bytes(
                        image_bytes=frame.image_bytes,
                        mime_type=frame.mime_type,
                        title=f"clip:{clip_id}",
                    )
                except Exception as exc:
                    logger.warning(
                        "[vector_index] image embed failed project_id=%s clip_id=%s chunk_index=%s: %s",
                        project_id,
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
                visual_stored += 1
                logger.debug(
                    "[vector_index] stored visual embedding project_id=%s clip_id=%s "
                    "chunk_index=%s time=%.2f-%.2f dim=%s",
                    project_id,
                    clip_id,
                    frame.chunk_index,
                    frame.start_time_seconds,
                    frame.end_time_seconds,
                    len(emb),
                )
            logger.info(
                "[vector_index] visual embedding summary project_id=%s clip_id=%s "
                "frames_attempted=%s rows_stored=%s",
                project_id,
                clip_id,
                len(visual_frames),
                visual_stored,
            )

        logger.info(
            "[vector_index] done project_id=%s session_id=%s clip_id=%s local_key=%s "
            "audio_chunks_stored=%s visual_frames_stored=%s",
            project_id,
            session_id,
            clip_id,
            local_key,
            audio_stored,
            visual_stored,
        )
    except Exception:
        logger.exception(
            "[vector_index] failed project_id=%s session_id=%s clip_id=%s local_key=%s",
            project_id,
            session_id,
            clip_id,
            local_key,
        )

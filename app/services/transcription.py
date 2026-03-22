import asyncio
import logging
import os
import time
from typing import Any

import modal

from app.services.audio_chunking import transcribe_with_chunking


logger = logging.getLogger(__name__)

MODAL_APP_NAME = os.getenv("MODAL_WHISPERX_APP_NAME", "whisperx-stitcher")
MODAL_CLASS_NAME = os.getenv("MODAL_WHISPERX_CLASS_NAME", "WhisperXEngine")


async def transcribe_audio_with_modal(audio_bytes: bytes) -> list[dict[str, Any]]:
    whisperx_cls = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
    engine = whisperx_cls()
    t0 = time.perf_counter()
    segments = await engine.transcribe_batch.remote.aio(audio_bytes)
    logger.info(
        "[TRANSCRIBE] Modal process_audio.remote duration_s=%.3f audio_bytes=%d",
        time.perf_counter() - t0,
        len(audio_bytes),
    )
    return segments or []


async def transcribe_audio_async(audio_bytes: bytes) -> list[dict[str, Any]]:
    return await transcribe_audio_with_modal(audio_bytes)


async def transcribe_upload_async(
    audio_bytes: bytes,
    suffix: str = ".m4a",
) -> list[dict[str, Any]]:
    """Transcribe one upload; splits into overlapping 1-minute chunks when duration > 1:30."""
    return await transcribe_with_chunking(
        audio_bytes,
        suffix,
        transcribe_audio_async,
    )

import asyncio
import logging
import os
import time
from typing import Any

import modal


logger = logging.getLogger(__name__)

MODAL_APP_NAME = os.getenv("MODAL_WHISPERX_APP_NAME", "whisperx-stitcher")
MODAL_CLASS_NAME = os.getenv("MODAL_WHISPERX_CLASS_NAME", "WhisperXEngine")


def transcribe_audio_with_modal(audio_bytes: bytes) -> list[dict[str, Any]]:
    whisperx_cls = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
    engine = whisperx_cls()
    t0 = time.perf_counter()
    segments = engine.process_audio.remote(audio_bytes)
    logger.info(
        "[TRANSCRIBE] Modal process_audio.remote duration_s=%.3f audio_bytes=%d",
        time.perf_counter() - t0,
        len(audio_bytes),
    )
    return segments or []


async def transcribe_audio_async(audio_bytes: bytes) -> list[dict[str, Any]]:
    return await asyncio.to_thread(transcribe_audio_with_modal, audio_bytes)

import logging
import os
import time
from typing import Any

import modal


logger = logging.getLogger(__name__)

MODAL_APP_NAME = os.getenv("MODAL_WHISPERX_APP_NAME", "whisperx-stitcher")
MODAL_TRANSCRIBE_CLIP_NAME = os.getenv("MODAL_TRANSCRIBE_CLIP_FUNCTION", "transcribe_clip")


async def transcribe_clip_modal(
    audio_bytes: bytes,
    suffix: str = ".m4a",
    clip_id: str | None = None,
) -> dict[str, Any]:
    """CPU split/merge, ASR, and clip card generation run inside Modal (`transcribe_clip`)."""
    fn = modal.Function.from_name(MODAL_APP_NAME, MODAL_TRANSCRIBE_CLIP_NAME)
    t0 = time.perf_counter()
    result = await fn.remote.aio(audio_bytes, suffix=suffix, clip_id=clip_id)
    elapsed = time.perf_counter() - t0
    if not isinstance(result, dict):
        raise RuntimeError(f"transcribe_clip returned {type(result).__name__}, expected dict")
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    logger.info(
        "[TRANSCRIBE] transcribe_clip duration_s=%.3f audio_bytes=%d modal_wall_s=%s "
        "window_count=%s split=%s",
        elapsed,
        len(audio_bytes),
        meta.get("wall_s"),
        meta.get("window_count"),
        meta.get("split"),
    )
    return result


async def transcribe_upload_async(
    audio_bytes: bytes,
    suffix: str = ".m4a",
    clip_id: str | None = None,
) -> dict[str, Any]:
    return await transcribe_clip_modal(audio_bytes, suffix=suffix, clip_id=clip_id)


async def transcribe_upload_for_ingest(
    audio_bytes: bytes,
    *,
    extension: str,
    index: int,
    file_name: str | None,
    clip_correlation_id: str | None = None,
) -> dict[str, Any]:
    """Call Modal for one ingest file and log end-to-end duration (used by `/ingest`)."""
    suffix = extension or ".m4a"
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    t0 = time.perf_counter()
    try:
        return await transcribe_upload_async(
            audio_bytes, suffix, clip_id=clip_correlation_id
        )
    finally:
        logger.info(
            "[INGEST] Transcription finished video=%d file=%s duration_s=%.3f",
            index,
            file_name,
            time.perf_counter() - t0,
        )

"""
FastAPI entry: health, frame-based visual analysis, and audio transcription.

The only path that deals with sampled video *frames* / Modal VLM is ``/analyze-video``.
``/transcribe-video`` is audio transcription only (WhisperX / chunking) and is kept aligned
with the historical API shape and logging.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.services.analyze_video import analyze_video_upload_async
from app.services.transcription import transcribe_upload_async
from app.services.transcript_normalize import (
    normalize_transcript_segments,
    segments_to_full_text,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.get("/")
def health():
    return {"status": "ok"}


def _require_video_count(videos: list[UploadFile]) -> None:
    if not 1 <= len(videos) <= 10:
        raise HTTPException(
            status_code=400,
            detail="Upload between 1 and 10 videos per request.",
        )


def _segments_and_full_text(raw: Any) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(raw, list):
        rows = list(raw) if raw is not None else []
    else:
        rows = raw
    segment_dicts = [s for s in rows if isinstance(s, dict)]
    segments = normalize_transcript_segments(segment_dicts)
    return segments, segments_to_full_text(segments)


# ---------------------------------------------------------------------------
# Audio transcription (unchanged contract: transcript_segments / transcript_full_text)
# ---------------------------------------------------------------------------


async def _transcribe_one(prepared: dict[str, Any]) -> list[dict[str, Any]]:
    video = prepared["video"]
    suffix = prepared.get("extension") or ".m4a"
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    t0 = time.perf_counter()
    try:
        return await transcribe_upload_async(prepared["video_bytes"], suffix)
    finally:
        logger.info(
            "[TRANSCRIBE_TEST] Transcription finished video=%d file=%s duration_s=%.3f",
            prepared["index"],
            video.filename,
            time.perf_counter() - t0,
        )


@app.post("/transcribe-video")
async def transcribe_videos(videos: list[UploadFile] = File(...)):
    _require_video_count(videos)

    request_t0 = time.perf_counter()
    logger.info("[TRANSCRIBE_TEST] Received request with %d video(s)", len(videos))
    prepared_videos: list[dict[str, Any]] = []

    for index, video in enumerate(videos, start=1):
        t_read = time.perf_counter()
        video.file.seek(0, 2)
        file_size_bytes = video.file.tell()
        video.file.seek(0)
        video_bytes = await video.read()
        read_s = time.perf_counter() - t_read
        extension = Path(video.filename or "").suffix.lower() or ".m4a"
        prepared_videos.append(
            {
                "index": index,
                "video": video,
                "file_size_bytes": file_size_bytes,
                "video_bytes": video_bytes,
                "extension": extension,
            }
        )
        logger.info(
            "[TRANSCRIBE_TEST] Prepared video %d file=%s mime=%s size=%d read_upload_s=%.3f",
            index,
            video.filename,
            video.content_type,
            file_size_bytes,
            read_s,
        )

    t_transcribe_wall = time.perf_counter()
    transcription_tasks = [_transcribe_one(pv) for pv in prepared_videos]
    transcription_results = await asyncio.gather(*transcription_tasks, return_exceptions=True)
    logger.info(
        "[TRANSCRIBE_TEST] Batch wall_duration_s=%.3f (concurrent)",
        time.perf_counter() - t_transcribe_wall,
    )

    videos_payload: list[dict[str, Any]] = []
    for prepared_video, transcription_result in zip(
        prepared_videos, transcription_results, strict=True
    ):
        index = prepared_video["index"]
        video = prepared_video["video"]
        if isinstance(transcription_result, Exception):
            logger.exception(
                "[TRANSCRIBE_TEST] Transcription failed for video %d file=%s",
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

        segments, full_text = _segments_and_full_text(transcription_result)
        videos_payload.append(
            {
                "index": index,
                "file_name": video.filename,
                "mime_type": video.content_type,
                "extension": prepared_video["extension"],
                "file_size_bytes": prepared_video["file_size_bytes"],
                "transcript_segments": segments,
                "transcript_full_text": full_text,
            }
        )

    logger.info(
        "[TRANSCRIBE_TEST] Completed request with %d processed video(s) total_duration_s=%.3f",
        len(videos_payload),
        time.perf_counter() - request_t0,
    )
    return {
        "uploaded_count": len(videos_payload),
        "videos": videos_payload,
    }


# ---------------------------------------------------------------------------
# Visual analysis only: interval-sampled frames → one-frame MP4 per step → Modal VLM
# ---------------------------------------------------------------------------


async def _analyze_one(
    prepared: dict[str, Any],
    frame_interval_s: float | None,
) -> list[dict[str, Any]]:
    video = prepared["video"]
    suffix = prepared.get("extension") or ".mp4"
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    t0 = time.perf_counter()
    try:
        return await analyze_video_upload_async(
            prepared["video_bytes"],
            suffix,
            interval_s=frame_interval_s,
        )
    finally:
        logger.info(
            "[ANALYZE] Visual (frame) analysis finished video=%d file=%s duration_s=%.3f",
            prepared["index"],
            video.filename,
            time.perf_counter() - t0,
        )


@app.post("/analyze-video")
async def analyze_videos(
    videos: list[UploadFile] = File(...),
    frame_interval_s: Annotated[
        float | None,
        Form(description="Seconds between frame samples; omit to use VIDEO_FRAME_INTERVAL_S / default."),
    ] = None,
):
    """
    Per-upload visual pipeline: sample the video on ``frame_interval_s``, encode each sample as a
    one-frame MP4, run Modal VLM sequentially inside ``analyze_video_upload_async``.
    """
    _require_video_count(videos)
    if frame_interval_s is not None and frame_interval_s <= 0:
        raise HTTPException(status_code=400, detail="frame_interval_s must be positive.")

    request_t0 = time.perf_counter()
    logger.info("[ANALYZE] Received request with %d video(s)", len(videos))
    prepared_videos: list[dict[str, Any]] = []

    for index, video in enumerate(videos, start=1):
        t_read = time.perf_counter()
        video.file.seek(0, 2)
        file_size_bytes = video.file.tell()
        video.file.seek(0)
        video_bytes = await video.read()
        read_s = time.perf_counter() - t_read
        extension = Path(video.filename or "").suffix.lower() or ".mp4"
        prepared_videos.append(
            {
                "index": index,
                "video": video,
                "file_size_bytes": file_size_bytes,
                "video_bytes": video_bytes,
                "extension": extension,
            }
        )
        logger.info(
            "[ANALYZE] Prepared video %d file=%s mime=%s size=%d read_upload_s=%.3f",
            index,
            video.filename,
            video.content_type,
            file_size_bytes,
            read_s,
        )

    t_wall = time.perf_counter()
    analysis_tasks = [
        _analyze_one(pv, frame_interval_s) for pv in prepared_videos
    ]
    analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
    logger.info(
        "[ANALYZE] Batch wall_duration_s=%.3f (concurrent across uploads)",
        time.perf_counter() - t_wall,
    )

    videos_payload: list[dict[str, Any]] = []
    for prepared_video, analysis_result in zip(prepared_videos, analysis_results, strict=True):
        index = prepared_video["index"]
        video = prepared_video["video"]
        if isinstance(analysis_result, Exception):
            logger.exception(
                "[ANALYZE] Visual analysis failed for video %d file=%s",
                index,
                video.filename,
                exc_info=analysis_result,
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Video analysis failed for {video.filename or f'video-{index}'}; "
                    "check server logs for details."
                ),
            ) from analysis_result

        segments, full_text = _segments_and_full_text(analysis_result)
        videos_payload.append(
            {
                "index": index,
                "file_name": video.filename,
                "mime_type": video.content_type,
                "extension": prepared_video["extension"],
                "file_size_bytes": prepared_video["file_size_bytes"],
                "description_segments": segments,
                "description_full_text": full_text,
            }
        )

    logger.info(
        "[ANALYZE] Completed request with %d processed video(s) total_duration_s=%.3f",
        len(videos_payload),
        time.perf_counter() - request_t0,
    )
    return {
        "uploaded_count": len(videos_payload),
        "videos": videos_payload,
    }

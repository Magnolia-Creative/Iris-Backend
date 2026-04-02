import asyncio
import json
import logging
import os
import time
from collections import deque
from typing import Any

import modal

from app.services.video_chunking import (
    FRAME_INTERVAL_S,
    extract_single_frame_as_mp4,
    plan_frame_timestamps,
    probe_duration_seconds,
)

logger = logging.getLogger(__name__)

MODAL_APP_NAME = os.getenv("MODAL_VLM_APP_NAME", "video-vlm")
MODAL_CLASS_NAME = os.getenv("MODAL_VLM_CLASS_NAME", "AnalyzeVideoEngine")
MODAL_METHOD_NAME = os.getenv("MODAL_VLM_METHOD_NAME", "analyze_video")


def _annotation_to_text(annotation: Any) -> str:
    if annotation is None:
        return ""
    if isinstance(annotation, str):
        return annotation.strip()
    if not isinstance(annotation, dict):
        return str(annotation).strip()

    # Modal LLaVA helper returns summary / scene_type / actions / objects
    for key in ("summary", "description", "text", "caption", "analysis", "narrative", "content"):
        val = annotation.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    for key in ("vlm", "output", "result", "response", "annotation"):
        val = annotation.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict):
            inner = _annotation_to_text(val)
            if inner:
                return inner

    for key in ("bullets", "points", "labels", "actions", "objects"):
        val = annotation.get(key)
        if isinstance(val, list):
            parts = [str(x).strip() for x in val if str(x).strip()]
            if parts:
                return "; ".join(parts)

    try:
        compact = json.dumps(annotation, default=str)
        return compact if len(compact) <= 4000 else compact[:3997] + "..."
    except (TypeError, ValueError):
        return str(annotation)


async def analyze_video_upload_async(
    video_bytes: bytes,
    suffix: str = ".mp4",
    interval_s: float | None = None,
) -> list[dict[str, Any]]:
    """
    Every ``interval_s``, take **one** frame at that timestamp, wrap it as a tiny one-frame MP4
    (for Modal’s OpenCV path), then ``await analyze_video.remote`` once — strictly sequential.
    """
    step = float(interval_s) if interval_s is not None else FRAME_INTERVAL_S

    t_probe = time.perf_counter()
    duration = await asyncio.to_thread(probe_duration_seconds, video_bytes, suffix)
    probe_s = time.perf_counter() - t_probe

    times = plan_frame_timestamps(duration, step)
    if not times:
        logger.info(
            "[ANALYZE] video_source_s=%.3f no samples probe_s=%.3f",
            duration,
            probe_s,
        )
        return []

    queue: deque[float] = deque(times)

    engine_cls = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
    engine = engine_cls()
    modal_method = getattr(engine, MODAL_METHOD_NAME, None)
    if modal_method is None:
        raise RuntimeError(
            f"Modal class '{MODAL_CLASS_NAME}' does not expose method '{MODAL_METHOD_NAME}'."
        )

    logger.info(
        "[ANALYZE] video_source_s=%.3f queued_frames=%d interval_s=%.3f probe_s=%.3f",
        duration,
        len(times),
        step,
        probe_s,
    )

    t_work = time.perf_counter()
    segments: list[dict[str, Any]] = []
    while queue:
        ts = queue.popleft()
        frame_mp4 = await asyncio.to_thread(
            extract_single_frame_as_mp4,
            video_bytes,
            ts,
            suffix,
        )
        t0 = time.perf_counter()
        ann = await modal_method.remote.aio(frame_mp4)
        logger.info(
            "[ANALYZE] Modal %s.remote modal_roundtrip_s=%.3f frame_mp4_bytes=%d t_frame=%.3fs queue_remaining=%d",
            MODAL_METHOD_NAME,
            time.perf_counter() - t0,
            len(frame_mp4),
            ts,
            len(queue),
        )
        segments.append(
            {
                "start": float(ts),
                "end": float(min(ts + step, duration)),
                "text": _annotation_to_text(ann),
            }
        )

    logger.info(
        "[ANALYZE] queue drained samples=%d wall_s=%.3f",
        len(segments),
        time.perf_counter() - t_work,
    )
    return segments

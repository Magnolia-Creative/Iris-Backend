"""Parse visual frame manifest and match uploaded image parts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisualFrameChunk:
    chunk_index: int
    start_time_seconds: float
    end_time_seconds: float
    center_time_seconds: float
    image_bytes: bytes
    mime_type: str


def _coerce_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError(f"expected number, got {type(value).__name__}")


def parse_visual_frame_manifest_json(raw: str) -> dict[str, list[dict[str, Any]]]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="visual_frame_manifest is not valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="visual_frame_manifest must be a JSON object.")

    clips = parsed.get("clips")
    if clips is None:
        return {}
    if not isinstance(clips, list):
        raise HTTPException(status_code=400, detail="visual_frame_manifest.clips must be an array.")

    by_local: dict[str, list[dict[str, Any]]] = {}
    for item in clips:
        if not isinstance(item, dict):
            continue
        local_key = item.get("local_key")
        if not isinstance(local_key, str) or not local_key.strip():
            raise HTTPException(status_code=400, detail="Each clip entry must include a non-empty local_key.")
        frames = item.get("frames")
        if not isinstance(frames, list):
            raise HTTPException(
                status_code=400, detail=f"visual_frame_manifest frames for {local_key} must be an array."
            )
        by_local[local_key] = []
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            filename = frame.get("filename")
            if not isinstance(filename, str) or not filename.strip():
                raise HTTPException(
                    status_code=400, detail="Each frame entry must include a non-empty filename."
                )
            by_local[local_key].append(frame)
    return by_local


async def build_visual_frames_by_local_key(
    *,
    manifest_raw: str | None,
    visual_frame_files: list[UploadFile],
) -> dict[str, list[VisualFrameChunk]]:
    if not manifest_raw or not manifest_raw.strip():
        return {}

    spec_by_local = parse_visual_frame_manifest_json(manifest_raw.strip())
    if not spec_by_local:
        return {}

    file_bytes_by_name: dict[str, tuple[bytes, str]] = {}
    for upload in visual_frame_files:
        name = upload.filename or ""
        if not name:
            continue
        data = await upload.read()
        if not data:
            continue
        mime = upload.content_type or "image/jpeg"
        file_bytes_by_name[name] = (data, mime)

    out: dict[str, list[VisualFrameChunk]] = {}
    for local_key, frames in spec_by_local.items():
        chunks: list[VisualFrameChunk] = []
        for frame in frames:
            filename = str(frame.get("filename") or "")
            if filename not in file_bytes_by_name:
                logger.warning(
                    "[visual_frame_payload] missing upload for local_key=%s filename=%s",
                    local_key,
                    filename,
                )
                continue
            image_bytes, mime_type = file_bytes_by_name[filename]
            try:
                chunk_index = int(frame.get("chunk_index", 0))
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400, detail=f"Invalid chunk_index for {local_key} file {filename}."
                ) from exc
            start = _coerce_float(frame["start_time_seconds"])
            end = _coerce_float(frame["end_time_seconds"])
            center = _coerce_float(frame["center_time_seconds"])
            chunks.append(
                VisualFrameChunk(
                    chunk_index=chunk_index,
                    start_time_seconds=start,
                    end_time_seconds=end,
                    center_time_seconds=center,
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                )
            )
        chunks.sort(key=lambda c: c.chunk_index)
        if chunks:
            out[local_key] = chunks
    return out

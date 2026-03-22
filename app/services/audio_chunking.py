"""Split AAC/M4A (and similar) into overlapping windows for long-form transcription."""

from __future__ import annotations

import asyncio
import copy
import logging
import shutil
import subprocess
import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Longer clips are split into windows of this length (seconds), advancing by STEP_SECONDS (overlap).
CHUNK_DURATION_S = 60.0
STEP_SECONDS = 57.0  # 3 s overlap between consecutive windows
# 1:30 — transcribe whole file in one Modal call when duration is at or below this.
SPLIT_THRESHOLD_S = 90.0


def _require_ffprobe() -> None:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe must be installed on the API host to measure audio duration.")


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg must be installed on the API host to split long audio into chunks.")


def probe_duration_seconds(audio_bytes: bytes, suffix: str = ".m4a") -> float:
    _require_ffprobe()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr or proc.stdout or "ffprobe failed")
        return float(proc.stdout.strip())
    finally:
        Path(path).unlink(missing_ok=True)


def plan_chunk_windows(total_duration_s: float) -> list[tuple[float, float]]:
    """Return (start_s, length_s) for each window; advance by STEP_SECONDS for overlap."""
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < total_duration_s - 1e-6:
        length = min(CHUNK_DURATION_S, total_duration_s - start)
        windows.append((start, length))
        if start + length >= total_duration_s - 1e-6:
            break
        start += STEP_SECONDS
    return windows


def extract_chunk_bytes(
    audio_bytes: bytes,
    start_s: float,
    duration_s: float,
    suffix: str = ".m4a",
) -> bytes:
    _require_ffmpeg()
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / f"in{suffix}"
        out = Path(tmp) / f"out{suffix}"
        inp.write_bytes(audio_bytes)
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(start_s),
            "-i",
            str(inp),
            "-t",
            str(duration_s),
            "-c",
            "copy",
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))
        return out.read_bytes()


def offset_segment_times(seg: dict[str, Any], offset_s: float) -> dict[str, Any]:
    g = copy.deepcopy(seg)
    for key in ("start", "end"):
        if key in g and g[key] is not None:
            g[key] = float(g[key]) + offset_s
    words = g.get("words")
    if isinstance(words, list):
        for w in words:
            if isinstance(w, dict):
                for key in ("start", "end"):
                    if key in w and w[key] is not None:
                        w[key] = float(w[key]) + offset_s
    return g


def merge_chunk_segments(
    per_chunk_segments: list[list[dict[str, Any]]],
    chunk_start_times_s: list[float],
    total_duration_s: float,
) -> list[dict[str, Any]]:
    """Assign each segment to its chunk's primary timeline slice, then sort and lightly merge."""
    selected: list[dict[str, Any]] = []
    for chunk_idx, (chunk_start, segs) in enumerate(
        zip(chunk_start_times_s, per_chunk_segments, strict=True)
    ):
        primary_lo = chunk_idx * STEP_SECONDS
        primary_hi = min((chunk_idx + 1) * STEP_SECONDS, total_duration_s)
        for seg in segs:
            g = offset_segment_times(seg, chunk_start)
            gs = float(g.get("start", 0))
            ge = float(g.get("end", gs))
            if ge <= gs:
                continue
            ov_lo = max(gs, primary_lo)
            ov_hi = min(ge, primary_hi)
            overlap = max(0.0, ov_hi - ov_lo)
            dur = ge - gs
            if dur <= 0 or overlap / dur < 0.5:
                continue
            selected.append(g)

    selected.sort(
        key=lambda s: (float(s.get("start", 0)), float(s.get("end", 0))),
    )
    return _consolidate_adjacent_segments(selected)


def _consolidate_adjacent_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not segments:
        return []
    out: list[dict[str, Any]] = []
    cur = copy.deepcopy(segments[0])
    for nxt in segments[1:]:
        cs, ce = float(cur.get("start", 0)), float(cur.get("end", 0))
        ns, ne = float(nxt.get("start", 0)), float(nxt.get("end", 0))
        gap = ns - ce
        if -0.05 < gap < 0.15:
            cur["end"] = max(ce, ne)
            ct = str(cur.get("text", "")).strip()
            nt = str(nxt.get("text", "")).strip()
            if nt and nt not in ct:
                cur["text"] = f"{ct} {nt}".strip()
            continue
        out.append(cur)
        cur = copy.deepcopy(nxt)
    out.append(cur)
    return out


async def transcribe_with_chunking(
    audio_bytes: bytes,
    suffix: str,
    transcribe: Callable[[bytes], Awaitable[list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """Probe duration; either one Modal call or chunk, transcribe in parallel, merge."""
    t_probe = time.perf_counter()
    duration = await asyncio.to_thread(probe_duration_seconds, audio_bytes, suffix)
    probe_s = time.perf_counter() - t_probe

    if duration <= SPLIT_THRESHOLD_S:
        logger.info(
            "[CHUNK] duration_s=%.3f (no split) probe_duration_s=%.3f",
            duration,
            probe_s,
        )
        return await transcribe(audio_bytes)

    windows = plan_chunk_windows(duration)
    logger.info(
        "[CHUNK] Splitting audio duration_s=%.3f into %d overlapping window(s) "
        "probe_duration_s=%.3f",
        duration,
        len(windows),
        probe_s,
    )

    t_extract = time.perf_counter()
    chunk_payloads = await asyncio.gather(
        *[
            asyncio.to_thread(extract_chunk_bytes, audio_bytes, start, length, suffix)
            for start, length in windows
        ]
    )
    extract_s = time.perf_counter() - t_extract

    per_chunk = await asyncio.gather(*[transcribe(blob) for blob in chunk_payloads])

    t_merge = time.perf_counter()
    merged = await asyncio.to_thread(
        merge_chunk_segments,
        list(per_chunk),
        [w[0] for w in windows],
        duration,
    )
    merge_s = time.perf_counter() - t_merge

    overhead_s = probe_s + extract_s + merge_s
    logger.info(
        "[CHUNK] extract_chunks_duration_s=%.3f merge_segments_duration_s=%.3f "
        "chunking_overhead_s=%.3f (probe+extract+merge; excludes Modal transcription)",
        extract_s,
        merge_s,
        overhead_s,
    )
    return merged

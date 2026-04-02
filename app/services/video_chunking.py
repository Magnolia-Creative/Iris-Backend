"""Video duration probing and interval-based samples: one frame each, as a minimal MP4."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# Seconds between frame extractions (anchors at 0s, INTERVAL, 2·INTERVAL, …).
FRAME_INTERVAL_S = float(os.getenv("VIDEO_FRAME_INTERVAL_S", "5"))


def _require_ffprobe() -> None:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe must be installed on the API host to measure video duration.")


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg must be installed on the API host to sample video.")


def probe_duration_seconds(video_bytes: bytes, suffix: str = ".mp4") -> float:
    _require_ffprobe()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(video_bytes)
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


def plan_frame_timestamps(total_duration_s: float, interval_s: float) -> list[float]:
    """Anchor times 0, interval_s, 2 * interval_s, … while within the clip."""
    if total_duration_s <= 0 or interval_s <= 0:
        return []
    times: list[float] = []
    t = 0.0
    while t < total_duration_s - 1e-6:
        times.append(round(t, 4))
        t += interval_s
    return times


def extract_single_frame_as_mp4(
    video_bytes: bytes,
    timestamp_s: float,
    suffix: str = ".mp4",
) -> bytes:
    """
    Encode **exactly one** video frame at ``timestamp_s`` into a minimal H.264 MP4.

    Modal workers expect ``.mp4`` bytes and run ``decode_first_video_frame`` (OpenCV); a one-frame
    MP4 is the smallest valid payload — not a multi-second clip.
    """
    _require_ffmpeg()
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / f"in{suffix}"
        out = Path(tmp) / "frame.mp4"
        inp.write_bytes(video_bytes)
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(timestamp_s),
            "-i",
            str(inp),
            "-frames:v",
            "1",
            "-an",
            "-sn",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            err = proc.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(err or "ffmpeg single-frame mp4 failed")
        return out.read_bytes()

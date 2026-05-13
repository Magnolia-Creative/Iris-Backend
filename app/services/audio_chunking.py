"""Slice audio into overlapping WAV chunks using ffmpeg."""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from app.services.chunk_time_windows import chunk_time_windows

logger = logging.getLogger(__name__)


def _ffprobe_duration_seconds(audio_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out) if out else 0.0


def slice_audio_to_wav_chunks(
    audio_bytes: bytes,
    *,
    suffix: str,
) -> list[tuple[int, float, float, float, bytes]]:
    """Return list of (chunk_index, start, end, center, wav_bytes)."""
    if not audio_bytes:
        return []

    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"source{suffix}"
        src.write_bytes(audio_bytes)
        try:
            duration = _ffprobe_duration_seconds(src)
        except Exception as exc:
            logger.warning("[audio_chunking] ffprobe failed: %s", exc)
            return []

        windows = chunk_time_windows(duration)
        results: list[tuple[int, float, float, float, bytes]] = []
        for idx, (start, end, center) in enumerate(windows):
            chunk_len = max(0.01, end - start)
            out_wav = Path(tmp) / f"chunk_{idx}.wav"
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(start),
                "-i",
                str(src),
                "-t",
                str(chunk_len),
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(out_wav),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as exc:
                logger.warning(
                    "[audio_chunking] ffmpeg chunk idx=%s failed: %s",
                    idx,
                    exc.stderr.decode(errors="replace") if exc.stderr else exc,
                )
                continue
            if out_wav.is_file():
                results.append((idx, start, end, center, out_wav.read_bytes()))
        return results


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

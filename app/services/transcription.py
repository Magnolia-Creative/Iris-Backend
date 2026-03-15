import asyncio
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import modal


MODAL_APP_NAME = os.getenv("MODAL_WHISPERX_APP_NAME", "whisperx-stitcher")
MODAL_CLASS_NAME = os.getenv("MODAL_WHISPERX_CLASS_NAME", "WhisperXEngine")


def extract_wav_from_video_bytes(video_bytes: bytes) -> bytes:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed on the API host.")

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "input_video"
        output_path = Path(temp_dir) / "output_audio.wav"
        input_path.write_bytes(video_bytes)

        # Write mono 16kHz PCM WAV to keep WhisperX I/O predictable.
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(output_path),
        ]
        process = subprocess.run(command, capture_output=True, check=False)
        if process.returncode != 0 or not output_path.exists():
            raise RuntimeError(process.stderr.decode("utf-8", errors="ignore"))

        return output_path.read_bytes()


def transcribe_audio_with_modal(audio_bytes: bytes) -> list[dict[str, Any]]:
    whisperx_cls = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
    engine = whisperx_cls()
    segments = engine.process_audio.remote(audio_bytes)
    return segments or []


def extract_and_transcribe(video_bytes: bytes) -> list[dict[str, Any]]:
    audio_bytes = extract_wav_from_video_bytes(video_bytes)
    return transcribe_audio_with_modal(audio_bytes)


async def extract_and_transcribe_async(video_bytes: bytes) -> list[dict[str, Any]]:
    return await asyncio.to_thread(extract_and_transcribe, video_bytes)

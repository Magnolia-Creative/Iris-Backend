from pathlib import Path
import subprocess

from app.services import audio_chunking


def test_bundled_ffmpeg_chunks_m4a_without_system_tools(monkeypatch, tmp_path: Path):
    from imageio_ffmpeg import get_ffmpeg_exe

    binary = get_ffmpeg_exe()
    source = tmp_path / "tone.m4a"
    subprocess.run(
        [binary, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=3", "-c:a", "aac", str(source)],
        check=True,
    )
    monkeypatch.setattr(audio_chunking.shutil, "which", lambda _: None)

    assert audio_chunking.ffmpeg_available()
    chunks = audio_chunking.slice_audio_to_wav_chunks(source.read_bytes(), suffix=".m4a")

    assert len(chunks) == 1
    assert chunks[0][4].startswith(b"RIFF")

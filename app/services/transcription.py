import logging
import os
import time
from typing import Any
import asyncio
import json
from urllib import error, request

import modal

from app.config import settings


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


def _assemblyai_headers() -> dict[str, str]:
    if not settings.assemblyai_api_key:
        raise RuntimeError(
            "ASSEMBLYAI_API_KEY is not set; set it or switch TRANSCRIPTION_PROVIDER=modal."
        )
    return {"Authorization": settings.assemblyai_api_key}


def _decode_json_response(raw: bytes) -> dict[str, Any]:
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"AssemblyAI returned non-JSON response ({len(raw)} bytes)") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError(f"AssemblyAI returned {type(decoded).__name__}, expected object")
    return decoded


def _http_json_request(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    payload: bytes | None = None,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    req = request.Request(url=url, data=payload, method=method)
    for k, v in headers.items():
        req.add_header(k, v)

    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
            return _decode_json_response(body)
    except error.HTTPError as exc:
        body = exc.read() if hasattr(exc, "read") else b""
        detail = ""
        if body:
            try:
                decoded = json.loads(body.decode("utf-8"))
                if isinstance(decoded, dict):
                    detail = decoded.get("error") or decoded.get("message") or str(decoded)
                else:
                    detail = str(decoded)
            except Exception:
                detail = body.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"AssemblyAI request failed method={method} url={url} status={exc.code} detail={detail}"
        ) from exc


def _assemblyai_ms_to_seconds(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return round(float(value) / 1000.0, 3)
    return None


def _segments_from_assemblyai(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    utterances = transcript.get("utterances")
    segments: list[dict[str, Any]] = []
    if isinstance(utterances, list):
        for utterance in utterances:
            if not isinstance(utterance, dict):
                continue
            text = str(utterance.get("text") or "").strip()
            if not text:
                continue
            segment: dict[str, Any] = {
                "text": text,
                "start": _assemblyai_ms_to_seconds(utterance.get("start")) or 0.0,
                "end": _assemblyai_ms_to_seconds(utterance.get("end")) or 0.0,
            }
            if isinstance(utterance.get("confidence"), (int, float)):
                segment["confidence"] = float(utterance["confidence"])
            if utterance.get("speaker") is not None:
                segment["speaker"] = utterance.get("speaker")
            if utterance.get("channel") is not None:
                segment["channel"] = utterance.get("channel")
            segments.append(segment)

    if segments:
        return segments

    text = str(transcript.get("text") or "").strip()
    if not text:
        return []
    end_seconds = transcript.get("audio_duration")
    if isinstance(end_seconds, (int, float)):
        end = float(end_seconds)
    else:
        end = 0.0
    return [{"text": text, "start": 0.0, "end": end}]


def _key_phrase_highlights_from_assemblyai(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    auto_highlights_result = transcript.get("auto_highlights_result")
    if not isinstance(auto_highlights_result, dict):
        return []
    results = auto_highlights_result.get("results")
    if not isinstance(results, list):
        return []

    highlights: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        phrase = str(item.get("text") or "").strip()
        if not phrase:
            continue
        count = item.get("count")
        rank = item.get("rank")
        timestamps_raw = item.get("timestamps")
        timestamps: list[dict[str, float]] = []
        if isinstance(timestamps_raw, list):
            for ts in timestamps_raw:
                if not isinstance(ts, dict):
                    continue
                start_ms = ts.get("start")
                end_ms = ts.get("end")
                if not isinstance(start_ms, (int, float)) or not isinstance(end_ms, (int, float)):
                    continue
                timestamps.append(
                    {
                        "start": round(float(start_ms) / 1000.0, 3),
                        "end": round(float(end_ms) / 1000.0, 3),
                    }
                )

        highlight: dict[str, Any] = {
            "text": phrase,
            "count": int(count) if isinstance(count, int) else 0,
            "rank": float(rank) if isinstance(rank, (int, float)) else 0.0,
            "timestamps": timestamps,
        }
        highlights.append(highlight)

    highlights.sort(key=lambda h: (h.get("rank", 0.0), h.get("count", 0)), reverse=True)
    return highlights


def _clip_about_from_highlights(highlights: list[dict[str, Any]], max_items: int = 8) -> str:
    phrases = [str(h.get("text") or "").strip() for h in highlights if str(h.get("text") or "").strip()]
    if not phrases:
        return ""
    return ", ".join(phrases[:max_items])


async def _assemblyai_upload_audio(audio_bytes: bytes) -> str:
    base_url = settings.assemblyai_base_url.rstrip("/")
    url = f"{base_url}/v2/upload"
    headers = {
        **_assemblyai_headers(),
        "Content-Type": "application/octet-stream",
    }
    response = await asyncio.to_thread(
        _http_json_request,
        method="POST",
        url=url,
        headers=headers,
        payload=audio_bytes,
        timeout_s=120.0,
    )
    upload_url = response.get("upload_url")
    if not isinstance(upload_url, str) or not upload_url:
        raise RuntimeError("AssemblyAI upload succeeded but response did not contain upload_url")
    return upload_url


async def _assemblyai_submit_transcript(upload_url: str) -> dict[str, Any]:
    base_url = settings.assemblyai_base_url.rstrip("/")
    url = f"{base_url}/v2/transcript"
    payload = {
        "audio_url": upload_url,
        "speech_models": ["universal-3-pro", "universal-2"],
        "language_detection": True,
        "auto_highlights": True,
        "speaker_labels": True,
        "format_text": True,
        "punctuate": True,
    }
    response = await asyncio.to_thread(
        _http_json_request,
        method="POST",
        url=url,
        headers={**_assemblyai_headers(), "Content-Type": "application/json"},
        payload=json.dumps(payload).encode("utf-8"),
        timeout_s=60.0,
    )
    return response


async def _assemblyai_poll_transcript(transcript_id: str) -> dict[str, Any]:
    base_url = settings.assemblyai_base_url.rstrip("/")
    url = f"{base_url}/v2/transcript/{transcript_id}"
    deadline = time.perf_counter() + max(settings.assemblyai_poll_timeout_seconds, 1.0)
    attempt = 0

    while True:
        attempt += 1
        response = await asyncio.to_thread(
            _http_json_request,
            method="GET",
            url=url,
            headers=_assemblyai_headers(),
            payload=None,
            timeout_s=60.0,
        )
        status = response.get("status")
        if status == "completed":
            return response
        if status == "error":
            raise RuntimeError(f"AssemblyAI transcript failed: {response.get('error')}")
        if time.perf_counter() >= deadline:
            raise RuntimeError(
                f"AssemblyAI transcript polling timed out after {attempt} attempts "
                f"for transcript_id={transcript_id}"
            )
        await asyncio.sleep(max(settings.assemblyai_poll_interval_seconds, 0.25))


async def transcribe_clip_assemblyai(
    audio_bytes: bytes,
    suffix: str = ".m4a",
    clip_id: str | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    upload_url = await _assemblyai_upload_audio(audio_bytes)
    submitted = await _assemblyai_submit_transcript(upload_url)
    transcript_id = submitted.get("id")
    if not isinstance(transcript_id, str) or not transcript_id:
        raise RuntimeError("AssemblyAI submit response missing transcript id")
    completed = await _assemblyai_poll_transcript(transcript_id)
    elapsed = time.perf_counter() - t0

    segments = _segments_from_assemblyai(completed)
    key_phrase_highlights = _key_phrase_highlights_from_assemblyai(completed)
    clip_about = _clip_about_from_highlights(key_phrase_highlights)
    audio_duration = completed.get("audio_duration")
    language_code = completed.get("language_code")
    confidence = completed.get("confidence")

    logger.info(
        "[TRANSCRIBE] assemblyai duration_s=%.3f audio_bytes=%d transcript_id=%s status=%s segments=%d",
        elapsed,
        len(audio_bytes),
        transcript_id,
        completed.get("status"),
        len(segments),
    )

    return {
        "transcript": segments,
        "video_report": {
            "provider": "assemblyai",
            "transcript_id": transcript_id,
            "status": completed.get("status"),
            "duration_seconds": float(audio_duration) if isinstance(audio_duration, (int, float)) else None,
            "language_code": language_code if isinstance(language_code, str) else None,
            "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
            "text_length": len(str(completed.get("text") or "")),
            "key_phrases": key_phrase_highlights,
            "key_phrases_status": (
                completed.get("auto_highlights_result", {}).get("status")
                if isinstance(completed.get("auto_highlights_result"), dict)
                else None
            ),
            "key_phrases_count": len(key_phrase_highlights),
            "about": clip_about,
        },
        "meta": {
            "provider": "assemblyai",
            "clip_id": clip_id,
            "suffix": suffix,
            "audio_bytes": len(audio_bytes),
            "wall_s": elapsed,
            "assemblyai_base_url": settings.assemblyai_base_url,
            "assemblyai_upload_url": upload_url,
            "assemblyai_transcript_id": transcript_id,
            "duration_seconds": float(audio_duration) if isinstance(audio_duration, (int, float)) else None,
        },
    }


async def transcribe_upload_async(
    audio_bytes: bytes,
    suffix: str = ".m4a",
    clip_id: str | None = None,
) -> dict[str, Any]:
    provider = (settings.transcription_provider or "assemblyai").lower()
    if provider == "modal":
        return await transcribe_clip_modal(audio_bytes, suffix=suffix, clip_id=clip_id)
    if provider == "assemblyai":
        return await transcribe_clip_assemblyai(audio_bytes, suffix=suffix, clip_id=clip_id)
    raise RuntimeError(
        f"Unsupported TRANSCRIPTION_PROVIDER={provider!r}. Use 'assemblyai' or 'modal'."
    )


async def transcribe_upload_for_ingest(
    audio_bytes: bytes,
    *,
    extension: str,
    index: int,
    file_name: str | None,
    clip_correlation_id: str | None = None,
) -> dict[str, Any]:
    """Transcribe one ingest file using configured provider and log duration."""
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

"""WebSocket proxy for OpenAI Realtime API transcription sessions."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import ssl
import time
from typing import Any, cast

from fastapi import WebSocket, WebSocketDisconnect
from websockets import exceptions as wsexceptions
from websockets.asyncio.client import connect

from app.config import settings

logger = logging.getLogger(__name__)

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime"

ALLOWED_TRANSCRIBE_MODELS: frozenset[str] = frozenset(
    {
        "gpt-realtime-whisper",
        "gpt-4o-mini-transcribe",
        "gpt-4o-transcribe",
    }
)
DEFAULT_TRANSCRIBE_MODEL = "gpt-realtime-whisper"


def _outbound_ssl_context() -> ssl.SSLContext:
    """TLS context for wss:// to OpenAI. Prefer certifi; override with config when needed."""
    ca = settings.outbound_ssl_cafile
    if ca:
        p = os.path.realpath(str(ca).strip())
        if os.path.isfile(p):
            return ssl.create_default_context(cafile=p)
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _transcription_session_update_event(*, model: str) -> dict[str, Any]:
    """
    GA Realtime transcription sessions use `session.update` with nested audio config.
    The audio capture clients send 24 kHz mono PCM16 chunks as base64.
    """
    return {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000,
                    },
                    "transcription": {
                        "model": model,
                        "language": "en",
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                    },
                },
            },
        },
    }


def _error_text(payload: dict[str, Any]) -> str:
    err = payload.get("error")
    if isinstance(err, dict) and "message" in err:
        return str(err["message"])
    return str(payload)[:4000]


def _outgoing_for_openai_event(msg: dict[str, Any]) -> dict[str, Any] | None:
    t = msg.get("type")
    if not isinstance(t, str):
        return None
    ts = int(time.time() * 1000)

    if t == "conversation.item.input_audio_transcription.delta":
        return {
            "type": "delta",
            "text": msg.get("delta", "") if msg.get("delta") is not None else "",
            "item_id": msg.get("item_id"),
            "content_index": msg.get("content_index"),
            "server_received_ts": ts,
        }

    if t == "conversation.item.input_audio_transcription.completed":
        return {
            "type": "completed",
            "text": msg.get("transcript", "") if msg.get("transcript") is not None else "",
            "item_id": msg.get("item_id"),
            "content_index": msg.get("content_index"),
            "server_received_ts": ts,
        }

    if t == "input_audio_buffer.speech_started":
        return {
            "type": "speech_started",
            "item_id": msg.get("item_id"),
            "server_received_ts": ts,
        }

    if t == "input_audio_buffer.speech_stopped":
        return {
            "type": "speech_stopped",
            "item_id": msg.get("item_id"),
            "server_received_ts": ts,
        }

    if t == "error":
        return {
            "type": "error",
            "detail": _error_text(msg),
            "server_received_ts": ts,
            "raw": msg,
        }

    if t == "conversation.item.input_audio_transcription.failed":
        failed_err = msg.get("error")
        detail: str
        if isinstance(failed_err, dict) and "message" in failed_err:
            detail = str(failed_err["message"])
        else:
            detail = str(msg)[:2000]
        return {
            "type": "error",
            "detail": detail,
            "server_received_ts": ts,
            "raw": msg,
        }

    return None


async def stream_transcription(
    client_ws: WebSocket,
    *,
    model: str = DEFAULT_TRANSCRIBE_MODEL,
) -> None:
    if not settings.openai_api_key:
        await client_ws.send_text(
            json.dumps(
                {
                    "type": "error",
                    "detail": "OPENAI_API_KEY is not configured on the server.",
                }
            )
        )
        await client_ws.close(code=1011)
        return

    if model not in ALLOWED_TRANSCRIBE_MODELS:
        await client_ws.send_text(
            json.dumps(
                {
                    "type": "error",
                    "detail": f"Unsupported model: {model}",
                }
            )
        )
        await client_ws.close(code=1008)
        return

    additional_headers: list[tuple[str, str]] = [
        ("Authorization", f"Bearer {settings.openai_api_key}"),
    ]

    oai: Any
    try:
        oai = await connect(
            OPENAI_REALTIME_URL,
            additional_headers=additional_headers,
            open_timeout=30.0,
            max_size=16_777_216,
            ssl=_outbound_ssl_context(),
        )
    except (OSError, wsexceptions.InvalidMessage, wsexceptions.WebSocketException) as exc:
        logger.exception("Failed to connect to OpenAI Realtime: %s", exc)
        await client_ws.send_text(
            json.dumps(
                {
                    "type": "error",
                    "detail": f"Failed to connect to OpenAI: {exc}",
                }
            )
        )
        await client_ws.close(code=1011)
        return

    async def pump_client_to_openai() -> None:
        try:
            while True:
                data = await client_ws.receive_text()
                try:
                    message = json.loads(data)
                except json.JSONDecodeError as exc:  # pragma: no cover - client fault
                    logger.warning("[transcribe] Invalid JSON from client: %s", exc)
                    continue

                mtype = message.get("type")
                if mtype == "audio":
                    audio = message.get("audio")
                    if not isinstance(audio, str) or not audio:
                        continue
                    try:
                        await oai.send(
                            json.dumps(
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": audio,
                                }
                            )
                        )
                    except wsexceptions.ConnectionClosed:
                        return
                elif mtype == "commit":
                    try:
                        await oai.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    except wsexceptions.ConnectionClosed:
                        return
                elif mtype == "stop":
                    return
        except WebSocketDisconnect:
            logger.info("[transcribe] Client disconnected; stopping upstream pump.")
        except wsexceptions.ConnectionClosed:
            return

    async def pump_openai_to_client() -> None:
        try:
            while True:
                try:
                    raw: str | bytes = await oai.recv()
                except wsexceptions.ConnectionClosed as exc:  # pragma: no cover - network
                    logger.info(
                        "[transcribe] OpenAI Realtime connection closed: code=%s reason=%s",
                        exc.code,
                        exc.reason,
                    )
                    return

                if isinstance(raw, (bytes, bytearray)):
                    logger.debug("[transcribe] Dropping non-text frame from OpenAI (binary).")
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:  # pragma: no cover
                    continue

                if not isinstance(msg, dict):
                    continue

                out = _outgoing_for_openai_event(cast(dict[str, Any], msg))
                if out is None:
                    continue
                try:
                    await client_ws.send_text(json.dumps(out))
                except WebSocketDisconnect:
                    return
        except WebSocketDisconnect:
            return
        except wsexceptions.ConnectionClosed:
            return

    try:
        update_event = _transcription_session_update_event(model=model)
        await oai.send(json.dumps(update_event))
        await client_ws.send_text(
            json.dumps(
                {
                    "type": "session_ready",
                    "model": model,
                }
            )
        )
        pump1 = asyncio.create_task(pump_client_to_openai())
        pump2 = asyncio.create_task(pump_openai_to_client())
        done, pending = await asyncio.wait(
            {pump1, pump2},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in done:
            if t.cancelled():
                continue
            try:
                exc = t.exception()
            except (asyncio.InvalidStateError, asyncio.CancelledError):
                continue
            if exc is not None and not isinstance(
                exc,
                (WebSocketDisconnect, wsexceptions.ConnectionClosed),
            ):
                logger.debug("[transcribe] task ended: %s", exc)
        for t in pending:
            t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*pending, return_exceptions=True)
    except WebSocketDisconnect:
        logger.info("[transcribe] WebSocket client disconnected (outer).")
    except wsexceptions.ConnectionClosed:
        pass
    except Exception:
        logger.exception("[transcribe] Unhandled error in stream_transcription")
    finally:
        with contextlib.suppress(wsexceptions.ConnectionClosed, OSError, RuntimeError):
            await oai.close()
        with contextlib.suppress(
            OSError, RuntimeError, wsexceptions.ConnectionClosed, WebSocketDisconnect
        ):
            await client_ws.close()
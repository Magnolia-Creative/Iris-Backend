from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any, cast

from fastapi import WebSocket, WebSocketDisconnect
from websockets import exceptions as wsexceptions
from websockets.asyncio.client import connect

from app.config import settings
from app.agent.intent.editing.service import IntentCompilerService
from app.agent.intent.editing.models import IntentCompilerContext
from app.services.realtime_transcription import (
    DEFAULT_TRANSCRIBE_MODEL,
    OPENAI_REALTIME_URL,
    _outbound_ssl_context,
    _transcription_session_update_event,
)


logger = logging.getLogger(__name__)


async def stream_voice_intent(
    client_ws: WebSocket,
    *,
    context: IntentCompilerContext,
    transcription_model: str = DEFAULT_TRANSCRIBE_MODEL,
    compiler_service: IntentCompilerService | None = None,
) -> None:
    if not settings.openai_api_key:
        await client_ws.send_text(json.dumps({"type": "error", "detail": "OPENAI_API_KEY is not configured on the server."}))
        await client_ws.close(code=1011)
        return

    oai: Any
    try:
        oai = await connect(
            OPENAI_REALTIME_URL,
            additional_headers=[
                ("Authorization", f"Bearer {settings.openai_api_key}"),
            ],
            open_timeout=30.0,
            max_size=16_777_216,
            ssl=_outbound_ssl_context(),
        )
    except (OSError, wsexceptions.InvalidMessage, wsexceptions.WebSocketException) as exc:
        logger.exception("Failed to connect voice intent transcription upstream: %s", exc)
        await client_ws.send_text(json.dumps({"type": "error", "detail": f"Failed to connect to OpenAI: {exc}"}))
        await client_ws.close(code=1011)
        return

    transcript_parts: list[str] = []
    partial_parts: list[str] = []
    finalize_received = asyncio.Event()
    transcript_completed = asyncio.Event()
    stop_received = asyncio.Event()

    async def send(payload: dict[str, Any]) -> None:
        await client_ws.send_text(json.dumps(payload))

    async def pump_client_to_openai() -> None:
        try:
            while True:
                message = await client_ws.receive_json()
                message_type = message.get("type")
                if message_type == "audio":
                    audio = message.get("audio")
                    if isinstance(audio, str) and audio:
                        await oai.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio}))
                elif message_type in {"finalize", "commit"}:
                    with contextlib.suppress(wsexceptions.ConnectionClosed):
                        await oai.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    finalize_received.set()
                    return
                elif message_type == "stop":
                    stop_received.set()
                    return
        except WebSocketDisconnect:
            stop_received.set()
        except wsexceptions.ConnectionClosed:
            stop_received.set()

    async def pump_openai_to_client() -> None:
        try:
            while not stop_received.is_set():
                raw = await oai.recv()
                if isinstance(raw, (bytes, bytearray)):
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict):
                    continue
                await _handle_openai_event(cast(dict[str, Any], msg), send, transcript_parts, partial_parts, transcript_completed)
        except (WebSocketDisconnect, wsexceptions.ConnectionClosed):
            stop_received.set()

    try:
        await oai.send(json.dumps(_transcription_session_update_event(model=transcription_model)))
        await send({"type": "session_ready", "model": transcription_model})
        client_task = asyncio.create_task(pump_client_to_openai())
        upstream_task = asyncio.create_task(pump_openai_to_client())

        await asyncio.wait({client_task}, return_when=asyncio.ALL_COMPLETED)
        if stop_received.is_set() and not finalize_received.is_set():
            return

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(transcript_completed.wait(), timeout=8.0)

        upstream_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await upstream_task

        prompt = " ".join(part.strip() for part in transcript_parts if part.strip()).strip()
        if not prompt:
            prompt = " ".join(partial_parts).strip()
        if not prompt:
            await send({"type": "error", "detail": "No transcript was captured."})
            return

        service = compiler_service or IntentCompilerService()

        async def compiler_event(payload: dict[str, Any]) -> None:
            await send(payload)

        result = await service.compile_prompt(prompt=prompt, context=context, event_handler=compiler_event)
        await send({"type": "intent_result", "prompt": prompt, "result": _jsonable(result)})
    except Exception:
        logger.exception("[voice-intent] Unhandled voice intent websocket error")
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await send({"type": "error", "detail": "Voice intent processing failed; check server logs."})
    finally:
        with contextlib.suppress(wsexceptions.ConnectionClosed, OSError, RuntimeError):
            await oai.close()
        with contextlib.suppress(WebSocketDisconnect, OSError, RuntimeError):
            await client_ws.close()


async def _handle_openai_event(
    msg: dict[str, Any],
    send: Any,
    transcript_parts: list[str],
    partial_parts: list[str],
    transcript_completed: asyncio.Event,
) -> None:
    event_type = msg.get("type")
    if event_type == "conversation.item.input_audio_transcription.delta":
        text = msg.get("delta") or ""
        if text:
            partial_parts.append(str(text))
        await send({"type": "transcript_delta", "text": text})
    elif event_type == "conversation.item.input_audio_transcription.completed":
        text = str(msg.get("transcript") or "")
        if text:
            transcript_parts.append(text)
        await send({"type": "transcript_completed", "text": text})
        transcript_completed.set()
    elif event_type == "input_audio_buffer.speech_started":
        await send({"type": "speech_started"})
    elif event_type == "input_audio_buffer.speech_stopped":
        await send({"type": "speech_stopped"})
    elif event_type == "error":
        error = msg.get("error")
        detail = error.get("message") if isinstance(error, dict) else str(msg)[:2000]
        await send({"type": "error", "detail": detail, "raw": msg})


def _jsonable(value: Any) -> dict[str, Any]:
    return json.loads(value.model_dump_json(by_alias=True))


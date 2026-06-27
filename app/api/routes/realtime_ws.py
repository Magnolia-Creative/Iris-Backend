import json

from fastapi import APIRouter, Query, WebSocket
from pydantic import ValidationError

from app.agent.intent.editing.voice import stream_voice_intent
from app.api.schemas.intent import VoiceIntentStartPayload
from app.auth import require_clerk_websocket_user
from app.services.realtime_transcription import (
    ALLOWED_TRANSCRIBE_MODELS,
    DEFAULT_TRANSCRIBE_MODEL,
    stream_transcription,
)


router = APIRouter()


@router.websocket("/ws/transcribe")
async def transcribe_websocket(
    websocket: WebSocket,
    model: str = Query(default=DEFAULT_TRANSCRIBE_MODEL),
) -> None:
    await require_clerk_websocket_user(websocket)
    await websocket.accept()
    if model not in ALLOWED_TRANSCRIBE_MODELS:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "error",
                    "detail": f"Unsupported model: {model}. "
                    f"Allowed: {', '.join(sorted(ALLOWED_TRANSCRIBE_MODELS))}",
                }
            )
        )
        await websocket.close(code=1008)
        return
    await stream_transcription(websocket, model=model)


@router.websocket("/ws/intent/voice")
async def voice_intent_websocket(
    websocket: WebSocket,
    model: str = Query(default=DEFAULT_TRANSCRIBE_MODEL),
) -> None:
    await require_clerk_websocket_user(websocket)
    await websocket.accept()
    if model not in ALLOWED_TRANSCRIBE_MODELS:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "error",
                    "detail": f"Unsupported model: {model}. "
                    f"Allowed: {', '.join(sorted(ALLOWED_TRANSCRIBE_MODELS))}",
                }
            )
        )
        await websocket.close(code=1008)
        return
    try:
        raw_message = await websocket.receive_json()
        start_payload = VoiceIntentStartPayload.model_validate(raw_message)
    except ValidationError as exc:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "error",
                    "detail": "Invalid start payload. Use {'type':'start','context':...}.",
                    "errors": exc.errors(),
                }
            )
        )
        await websocket.close(code=1003)
        return

    await stream_voice_intent(
        websocket,
        context=start_payload.context,
        transcription_model=model,
    )

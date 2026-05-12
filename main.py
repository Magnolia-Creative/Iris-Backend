from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from decimal import Decimal
import json
import logging
from typing import Any
from typing import Literal

from fastapi import (
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Base, engine, get_db
from app.automake.runtime import (
    approve_session_timeline,
    get_session_state,
    resume_session_from_reprompt,
    run_session_until_pause,
    set_session_state,
)
from app.services.session_graph_state_store import get_persisted_session_graph_state
from app.services.session_debug_store import (
    build_session_debug_snapshot,
    initialize_session_debug,
    record_session_event,
)
from app.services.session_ingest import (
    cancel_clip_processing,
    create_agent_session,
    ingest_session_clips,
    process_project_clips,
)
from app.services.session_state_builder import build_initial_state_from_session_payload
from app.services.realtime_transcription import (
    ALLOWED_TRANSCRIBE_MODELS,
    DEFAULT_TRANSCRIBE_MODEL,
    stream_transcription,
)
from app.intent_compiler.llm import IntentCompilerService
from app.intent_compiler.models import IntentCompileRequest, IntentCompilerContext
from app.intent_compiler.runs import create_intent_run, delete_intent_run, get_intent_run
from app.intent_compiler.transcripts import hydrate_intent_transcript_context
from app.intent_compiler.voice import stream_voice_intent
from app.services.transcription import print_received_transcript, transcribe_upload_to_sentences
from app.services.transcript_store import get_persisted_session_data
from app.database import models  # noqa: F401


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


class ClipCreate(BaseModel):
    title: str | None = None
    file_name: str | None = None
    source_url: str | None = None
    mime_type: str | None = None
    codec: str | None = None
    frame_rate: Decimal | None = None
    duration_seconds: Decimal | None = None
    width: int | None = None
    height: int | None = None
    file_size_bytes: int | None = None
    language_code: str | None = None
    captured_at: datetime | None = None


class IngestCreate(BaseModel):
    project_name: str
    clip: ClipCreate
    transcript: dict[str, Any]
    summary: str


class AgentSessionCreatePayload(BaseModel):
    project_name: str | None = None
    session_name: str | None = None


class WebSocketSessionStartPayload(BaseModel):
    type: Literal["start_session"]
    user_prompt: str


class WebSocketRepromptPayload(BaseModel):
    type: Literal["reprompt"]
    prompt: str


class VoiceIntentStartPayload(BaseModel):
    type: Literal["start"]
    context: IntentCompilerContext


def _is_timeline_approval_message(prompt: str) -> bool:
    normalized = " ".join(prompt.lower().strip().split())
    rejection_markers = {"do not approve", "don't approve", "not approved", "reject", "decline"}
    if any(marker in normalized for marker in rejection_markers):
        return False

    if normalized in {"yes", "yep", "yeah", "ok", "okay"}:
        return True

    approval_markers = {"approve", "approved", "looks good", "go ahead", "ship it"}
    return any(marker in normalized for marker in approval_markers)


def _intent_context_log_summary(context: IntentCompilerContext) -> dict[str, Any]:
    return {
        "timeline_id": context.timelineId,
        "project_id": context.projectId,
        "session_id": context.sessionId,
        "selected_clip_id": context.selectedClipId,
        "selected_track_id": context.selectedTrackId,
        "clip_count": len(context.clipsById),
        "track_count": len(context.orderedClipIdsByTrackId),
        "transcript_context_count": len(context.transcriptContextsByClipId),
    }


def _intent_result_log_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_count": len(result.get("actions") or []),
        "experimental_effect_count": len(result.get("experimentalEffectOperations") or []),
        "warnings": result.get("warnings") or [],
        "needs_clarification": result.get("needsClarification"),
        "source": result.get("source"),
    }


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Keep simple table creation for local development; use Alembic for production migrations.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/db-health")
async def db_health(db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("SELECT 1"))
    return {"database": "ok", "result": result.scalar_one()}


@app.post("/transcriptions/sentences")
async def create_sentence_transcription(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio upload was empty.")

    suffix = ""
    if audio.filename and "." in audio.filename:
        suffix = f".{audio.filename.rsplit('.', 1)[-1]}"

    result = await transcribe_upload_to_sentences(audio_bytes, suffix=suffix or ".m4a")
    print_received_transcript(
        source="main_endpoint",
        transcript_id=result.get("transcript_id"),
        full_text=result.get("full_text"),
        segments=result.get("sentences"),
    )
    return result


@app.post("/sessions/upload")
async def create_session_from_upload(
    videos: list[UploadFile] = File(...),
    local_keys: list[str] = Form(..., alias="local_key"),
    db: AsyncSession = Depends(get_db),
    session_name: str | None = None,
):
    ingest_result = await ingest_session_clips(
        db,
        videos,
        local_keys=local_keys,
        session_name=session_name,
    )
    return {
        "session_id": ingest_result["session_id"],
        "session_name": ingest_result["session_name"],
        "session_status": ingest_result["session_status"],
        "project_id": ingest_result["project_id"],
        "project_name": ingest_result["project_name"],
        "uploaded_count": ingest_result["uploaded_count"],
        "pending_clip_count": ingest_result["pending_clip_count"],
        "settled_clip_count": ingest_result["settled_clip_count"],
        "ready_for_websocket": ingest_result["ready_for_websocket"],
        "videos": ingest_result["videos"],
    }


@app.post("/projects/agent-sessions")
async def create_project_agent_session(
    payload: AgentSessionCreatePayload = Body(default=AgentSessionCreatePayload()),
    db: AsyncSession = Depends(get_db),
):
    return await create_agent_session(
        db,
        session_name=payload.session_name,
        project_name=payload.project_name,
    )


@app.post("/intent-runs")
async def create_intent_run_endpoint(
    payload: IntentCompileRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    logger.info(
        "[intent-runs] Create requested prompt_chars=%s context=%s",
        len(payload.prompt),
        _intent_context_log_summary(payload.context),
    )
    context = await hydrate_intent_transcript_context(payload.context, db)
    logger.info(
        "[intent-runs] Context hydrated prompt_chars=%s context=%s",
        len(payload.prompt),
        _intent_context_log_summary(context),
    )
    run = create_intent_run(prompt=payload.prompt, context=context)
    websocket_url = str(request.url_for("intent_run_websocket", run_id=run.run_id)).replace(
        "http://",
        "ws://",
        1,
    ).replace("https://", "wss://", 1)
    logger.info(
        "[intent-runs] Created run=%s websocket_url=%s",
        run.run_id,
        websocket_url,
    )
    return {"run_id": run.run_id, "websocket_url": websocket_url}


@app.post("/projects/{project_id}/clips/process")
async def process_project_clip_batch(
    project_id: int,
    videos: list[UploadFile] = File(...),
    local_keys: list[str] = Form(..., alias="local_key"),
    session_id: int = Form(...),
    db: AsyncSession = Depends(get_db),
):
    return await process_project_clips(
        db,
        project_id=project_id,
        session_id=session_id,
        videos=videos,
        local_keys=local_keys,
    )


@app.get("/sessions/{session_id}")
async def get_session_status(
    session_id: int,
    db: AsyncSession = Depends(get_db),
):
    session_payload = await get_persisted_session_data(
        db=db,
        session_id=session_id,
        include_ingest_details=False,
    )
    if session_payload is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return session_payload


@app.get("/sessions/{session_id}/debug")
async def get_session_debug(
    session_id: int,
    db: AsyncSession = Depends(get_db),
):
    session_payload = await get_persisted_session_data(
        db=db,
        session_id=session_id,
        include_ingest_details=True,
    )
    if session_payload is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    persisted_graph_state = await get_persisted_session_graph_state(db=db, session_id=session_id)
    state = get_session_state(str(session_id)) or persisted_graph_state
    initialize_session_debug(str(session_id), state)
    return build_session_debug_snapshot(
        session_id=session_id,
        session_payload=session_payload,
        state=state,
    )


@app.delete("/projects/{project_id}/clips/{local_key}")
async def cancel_project_clip(
    project_id: int,
    local_key: str,
    session_id: int,
    db: AsyncSession = Depends(get_db),
):
    return await cancel_clip_processing(
        db,
        project_id=project_id,
        session_id=session_id,
        local_key=local_key,
    )


@app.websocket("/ws/sessions/{session_id}")
async def session_websocket(
    websocket: WebSocket,
    session_id: int,
    db: AsyncSession = Depends(get_db),
):
    await websocket.accept()

    async def send_event(payload: dict[str, Any]) -> None:
        tracked_payload = record_session_event(str(session_id), payload)
        await websocket.send_text(json.dumps(tracked_payload))

    try:
        raw_message = await websocket.receive_json()
        start_payload = WebSocketSessionStartPayload.model_validate(raw_message)
        persisted_session_data = await get_persisted_session_data(
            db=db,
            session_id=session_id,
            include_ingest_details=True,
        )
        if persisted_session_data is None:
            await send_event(
                {
                    "type": "error",
                    "session_id": session_id,
                    "detail": f"Session {session_id} not found.",
                }
            )
            await websocket.close(code=1008)
            return

        if not persisted_session_data.get("ready_for_websocket"):
            await send_event(
                {
                    "type": "error",
                    "session_id": session_id,
                    "detail": "Session clips are still processing. Wait for remote transcripts to finish before starting.",
                }
            )
            await websocket.close(code=1008)
            return

        initial_state = build_initial_state_from_session_payload(
            session_id=str(session_id),
            session_payload=persisted_session_data,
            user_prompt=start_payload.user_prompt,
        )
        persisted_graph_state = await get_persisted_session_graph_state(db=db, session_id=session_id)
        initialize_session_debug(
            str(session_id),
            persisted_graph_state or initial_state,
        )

        await send_event(
            {
                "type": "session_started",
                "session_id": session_id,
                "project_id": persisted_session_data["project_id"],
                "uploaded_count": persisted_session_data["uploaded_count"],
            }
        )
        if persisted_graph_state is not None:
            set_session_state(str(session_id), persisted_graph_state)
            if persisted_graph_state.get("waiting_for_user"):
                await send_event(
                    {
                        "type": "timeline_update",
                        "session_id": session_id,
                        "timeline": persisted_graph_state.get("timeline", []),
                        "timeline_notes": persisted_graph_state.get("timeline_notes", []),
                    }
                )
                await send_event(
                    {
                        "type": "waiting_for_user",
                        "session_id": session_id,
                        "project_id": persisted_graph_state.get("project_id"),
                    }
                )
            elif persisted_graph_state.get("next_action") == "finish":
                await send_event(
                    {
                        "type": "session_complete",
                        "session_id": session_id,
                        "project_id": persisted_graph_state.get("project_id"),
                        "timeline": persisted_graph_state.get("timeline", []),
                    }
                )
            else:
                await run_session_until_pause(
                    state=persisted_graph_state,
                    db=db,
                    event_handler=send_event,
                )
        else:
            await run_session_until_pause(
                state=initial_state,
                db=db,
                event_handler=send_event,
            )

        while True:
            raw_message = await websocket.receive_json()
            message_type = raw_message.get("type")
            if message_type == "reprompt":
                try:
                    reprompt_payload = WebSocketRepromptPayload.model_validate(raw_message)
                except ValidationError as exc:
                    logger.warning(
                        "[ws] Invalid reprompt payload for session=%s payload=%s",
                        session_id,
                        raw_message,
                    )
                    await send_event(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "detail": "Invalid reprompt payload. Use {'type':'reprompt','prompt':'...'}",
                            "errors": exc.errors(),
                        }
                    )
                    continue
                current_state = get_session_state(str(session_id))
                if (
                    current_state
                    and current_state.get("waiting_for_user")
                    and _is_timeline_approval_message(reprompt_payload.prompt)
                ):
                    await approve_session_timeline(
                        session_id=str(session_id),
                        db=db,
                        event_handler=send_event,
                    )
                    continue
                await resume_session_from_reprompt(
                    session_id=str(session_id),
                    prompt=reprompt_payload.prompt,
                    db=db,
                    event_handler=send_event,
                )
                continue

            if message_type == "done":
                await send_event({"type": "session_closed", "session_id": session_id})
                await websocket.close()
                return

            await send_event(
                {
                    "type": "error",
                    "session_id": session_id,
                    "detail": "Unsupported message type. Use reprompt or done.",
                }
            )
    except ValidationError as exc:
        logger.exception("[ws] Invalid websocket payload for session=%s", session_id)
        await send_event(
            {
                "type": "error",
                "session_id": session_id,
                "detail": "Invalid websocket payload.",
                "errors": exc.errors(),
            }
        )
        await websocket.close(code=1003)
    except WebSocketDisconnect:
        logger.info("[ws] Client disconnected session=%s", session_id)
    except Exception:
        logger.exception("[ws] Session websocket failed session=%s", session_id)
        await send_event(
            {
                "type": "error",
                "session_id": session_id,
                "detail": "Session websocket failed; check server logs for details.",
            }
        )
        await websocket.close(code=1011)


@app.websocket("/ws/intent-runs/{run_id}", name="intent_run_websocket")
async def intent_run_websocket(
    websocket: WebSocket,
    run_id: str,
) -> None:
    await websocket.accept()
    logger.info("[intent-runs] WebSocket accepted run=%s", run_id)
    run = get_intent_run(run_id)
    if run is None:
        logger.warning("[intent-runs] WebSocket run not found run=%s", run_id)
        await websocket.send_text(
            json.dumps(
                {
                    "type": "error",
                    "run_id": run_id,
                    "detail": f"Intent run {run_id} not found.",
                }
            )
        )
        await websocket.close(code=1008)
        return

    async def send_event(payload: dict[str, Any]) -> None:
        event_type = str(payload.get("type") or "<missing>")
        message = json.dumps({"run_id": run_id, **payload})
        logger.info(
            "[intent-runs] Sending event run=%s type=%s bytes=%s",
            run_id,
            event_type,
            len(message),
        )
        await websocket.send_text(message)

    try:
        logger.info(
            "[intent-runs] Compile starting run=%s prompt_chars=%s context=%s",
            run_id,
            len(run.prompt),
            _intent_context_log_summary(run.context),
        )
        await send_event({"type": "run_started", "prompt": run.prompt})
        service = IntentCompilerService()
        result = await service.compile_prompt(
            prompt=run.prompt,
            context=run.context,
            event_handler=send_event,
        )
        result_payload = json.loads(result.model_dump_json(by_alias=True))
        logger.info(
            "[intent-runs] Compile completed run=%s summary=%s result_json=%s",
            run_id,
            _intent_result_log_summary(result_payload),
            json.dumps(result_payload),
        )
        await send_event(
            {
                "type": "intent_result",
                "prompt": run.prompt,
                "result": result_payload,
            }
        )
        await websocket.close()
        logger.info("[intent-runs] WebSocket closed normally run=%s", run_id)
    except WebSocketDisconnect:
        logger.info("[intent-runs] Client disconnected run=%s", run_id)
    except Exception:
        logger.exception("[intent-runs] Intent run failed run=%s", run_id)
        with suppress(WebSocketDisconnect, OSError, RuntimeError):
            await send_event(
                {
                    "type": "error",
                    "detail": "Intent run failed; check server logs for details.",
                }
            )
        with suppress(WebSocketDisconnect, OSError, RuntimeError):
            await websocket.close(code=1011)
    finally:
        logger.info("[intent-runs] Deleting run=%s", run_id)
        delete_intent_run(run_id)


@app.websocket("/ws/transcribe")
async def transcribe_websocket(
    websocket: WebSocket,
    model: str = Query(default=DEFAULT_TRANSCRIBE_MODEL),
) -> None:
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


@app.websocket("/ws/intent/voice")
async def voice_intent_websocket(
    websocket: WebSocket,
    model: str = Query(default=DEFAULT_TRANSCRIBE_MODEL),
) -> None:
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

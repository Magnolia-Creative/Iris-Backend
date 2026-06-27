import json
import logging
from time import perf_counter
from typing import Any

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.automake.runtime import (
    approve_session_timeline,
    get_session_state,
    resume_session_from_reprompt,
    run_session_until_pause,
    set_session_state,
)
from app.agent.intent import IntentAgentResponse, run_intent_agent
from app.agent.intent.editing.voice import stream_voice_intent
from app.agent.intent.editing.transcripts import prepare_intent_transcript_context
from app.api.context import context_id
from app.api.intent_logging import intent_context_log_summary, intent_result_log_summary
from app.api.schemas.agent import (
    AgentRunCreatePayload,
    AutomakeAgentRunCreatePayload,
    IntentAgentRunCreatePayload,
)
from app.api.schemas.intent import VoiceIntentStartPayload
from app.api.schemas.sessions import WebSocketRepromptPayload, WebSocketSessionStartPayload
from app.api.session_messages import is_timeline_approval_message
from app.api.timing import elapsed_ms
from app.auth import (
    ClerkPrincipal,
    require_clerk_user,
    require_clerk_websocket_user,
    require_owned_project,
    require_owned_session,
)
from app.database import get_db
from app.services.realtime_transcription import (
    ALLOWED_TRANSCRIBE_MODELS,
    DEFAULT_TRANSCRIBE_MODEL,
    stream_transcription,
)
from app.services.session_debug_store import build_session_debug_snapshot, initialize_session_debug
from app.services.session_debug_store import record_session_event
from app.services.session_graph_state_store import get_persisted_session_graph_state
from app.services.session_state_builder import build_initial_state_from_session_payload
from app.services.session_ingest import create_agent_session, create_agent_session_for_project
from app.services.transcription import print_received_transcript, transcribe_upload_to_sentences
from app.services.transcript_store import (
    get_persisted_session_data,
    insert_sentence_upload_transcript,
)


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/agent/transcriptions/sentences")
async def create_sentence_transcription(
    audio: UploadFile = File(...),
    _: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio upload was empty.")

    suffix = ""
    if audio.filename and "." in audio.filename:
        suffix = f".{audio.filename.rsplit('.', 1)[-1]}"

    result = await transcribe_upload_to_sentences(audio_bytes, suffix=suffix or ".m4a")
    provider_transcript_id = result.get("transcript_id")
    db_transcript_id = await insert_sentence_upload_transcript(db, result)
    result["transcript_id"] = db_transcript_id
    meta = result.get("meta")
    meta_out = dict(meta) if isinstance(meta, dict) else {}
    meta_out["provider_transcript_id"] = provider_transcript_id
    result["meta"] = meta_out
    print_received_transcript(
        source="main_endpoint",
        transcript_id=db_transcript_id,
        full_text=result.get("full_text"),
        segments=result.get("sentences"),
    )
    return result


@router.post("/agent/runs")
async def create_agent_run(
    payload: AgentRunCreatePayload = Body(...),
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    if isinstance(payload, IntentAgentRunCreatePayload):
        return await _run_intent_agent(payload, principal=principal, db=db)
    if isinstance(payload, AutomakeAgentRunCreatePayload):
        return await _create_automake_agent_run(payload, principal=principal, db=db)
    raise HTTPException(status_code=422, detail="Unsupported agent run kind.")


async def _run_intent_agent(
    payload: IntentAgentRunCreatePayload,
    *,
    principal: ClerkPrincipal,
    db: AsyncSession,
) -> IntentAgentResponse:
    started_at = perf_counter()
    logger.info(
        "[intent-agent] Run requested user=%s prompt_chars=%s context=%s",
        principal.user_id,
        len(payload.prompt),
        intent_context_log_summary(payload.context),
    )
    project_id = context_id(payload.context.projectId)
    if project_id is not None:
        ownership_started_at = perf_counter()
        logger.info("[intent-agent] Project ownership check starting project_id=%s", project_id)
        await require_owned_project(db, project_id=project_id, principal=principal)
        logger.info(
            "[intent-agent] Project ownership check completed project_id=%s elapsed_ms=%s",
            project_id,
            elapsed_ms(ownership_started_at),
        )

    session_id = context_id(payload.context.sessionId)
    if session_id is not None:
        ownership_started_at = perf_counter()
        logger.info("[intent-agent] Session ownership check starting session_id=%s", session_id)
        await require_owned_session(db, session_id=session_id, principal=principal)
        logger.info(
            "[intent-agent] Session ownership check completed session_id=%s elapsed_ms=%s",
            session_id,
            elapsed_ms(ownership_started_at),
        )

    hydration_started_at = perf_counter()
    logger.info("[intent-agent] Context hydration starting prompt_chars=%s", len(payload.prompt))
    context, hydration_meta = await prepare_intent_transcript_context(
        prompt=payload.prompt,
        context=payload.context,
        db=db,
    )
    logger.info(
        "[intent-agent] Context hydrated prompt_chars=%s context=%s elapsed_ms=%s",
        len(payload.prompt),
        intent_context_log_summary(context, hydration=hydration_meta),
        elapsed_ms(hydration_started_at),
    )

    result = await run_intent_agent(
        payload,
        context=context,
        hydration_meta=hydration_meta,
    )
    result_payload = json.loads(result.model_dump_json(by_alias=True))
    logger.info(
        "[intent-agent] Run completed user=%s summary=%s workspace_id=%s elapsed_ms=%s",
        principal.user_id,
        intent_result_log_summary(result_payload.get("edit") or {}),
        result.ui.workspaceId,
        elapsed_ms(started_at),
    )
    return result


async def _create_automake_agent_run(
    payload: AutomakeAgentRunCreatePayload,
    *,
    principal: ClerkPrincipal,
    db: AsyncSession,
):
    if payload.project_id is not None:
        return await create_agent_session_for_project(
            db,
            owner_user_id=principal.user_id,
            project_id=payload.project_id,
            session_name=payload.session_name,
        )
    return await create_agent_session(
        db,
        owner_user_id=principal.user_id,
        session_name=payload.session_name,
        project_name=payload.project_name,
    )


@router.get("/agent/runs/{run_id}")
async def get_agent_run_status(
    run_id: int,
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    await require_owned_session(db, session_id=run_id, principal=principal)
    session_payload = await get_persisted_session_data(
        db=db,
        session_id=run_id,
        include_ingest_details=False,
    )
    if session_payload is None:
        raise HTTPException(status_code=404, detail=f"Agent run {run_id} not found.")
    return session_payload


@router.get("/agent/runs/{run_id}/debug")
async def get_agent_run_debug(
    run_id: int,
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    await require_owned_session(db, session_id=run_id, principal=principal)
    session_payload = await get_persisted_session_data(
        db=db,
        session_id=run_id,
        include_ingest_details=True,
    )
    if session_payload is None:
        raise HTTPException(status_code=404, detail=f"Agent run {run_id} not found.")

    persisted_graph_state = await get_persisted_session_graph_state(db=db, session_id=run_id)
    state = get_session_state(str(run_id)) or persisted_graph_state
    initialize_session_debug(str(run_id), state)
    return build_session_debug_snapshot(
        session_id=run_id,
        session_payload=session_payload,
        state=state,
    )


@router.websocket("/agent/runs/{run_id}/stream")
async def agent_run_websocket(
    websocket: WebSocket,
    run_id: int,
    db: AsyncSession = Depends(get_db),
):
    session_id = run_id
    websocket_started_at = perf_counter()
    logger.info("[agent/ws] Session websocket auth starting session=%s", session_id)
    principal = await require_clerk_websocket_user(websocket)
    logger.info(
        "[agent/ws] Session websocket ownership check starting session=%s user=%s",
        session_id,
        principal.user_id,
    )
    await require_owned_session(db, session_id=session_id, principal=principal)
    await websocket.accept()
    logger.info(
        "[agent/ws] Session websocket accepted session=%s user=%s",
        session_id,
        principal.user_id,
    )

    async def send_event(payload: dict[str, Any]) -> None:
        tracked_payload = record_session_event(str(session_id), payload)
        event_type = str(tracked_payload.get("type") or "<missing>")
        logger.info(
            "[agent/ws] Sending event session=%s type=%s keys=%s",
            session_id,
            event_type,
            sorted(tracked_payload),
        )
        await websocket.send_text(json.dumps(tracked_payload))

    try:
        raw_message = await websocket.receive_json()
        logger.info(
            "[agent/ws] Received initial payload session=%s type=%s keys=%s",
            session_id,
            raw_message.get("type") if isinstance(raw_message, dict) else None,
            sorted(raw_message) if isinstance(raw_message, dict) else None,
        )
        start_payload = WebSocketSessionStartPayload.model_validate(raw_message)
        persisted_started_at = perf_counter()
        logger.info("[agent/ws] Loading persisted session data session=%s", session_id)
        persisted_session_data = await get_persisted_session_data(
            db=db,
            session_id=session_id,
            include_ingest_details=True,
        )
        logger.info(
            "[agent/ws] Loaded persisted session data session=%s found=%s elapsed_ms=%s",
            session_id,
            persisted_session_data is not None,
            elapsed_ms(persisted_started_at),
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
        logger.info(
            "[agent/ws] Initial state built session=%s clips=%s prompt_chars=%s",
            session_id,
            len(initial_state.get("clips", [])),
            len(start_payload.user_prompt),
        )
        persisted_graph_started_at = perf_counter()
        logger.info("[agent/ws] Loading persisted graph state session=%s", session_id)
        persisted_graph_state = await get_persisted_session_graph_state(
            db=db,
            session_id=session_id,
        )
        logger.info(
            "[agent/ws] Loaded persisted graph state session=%s found=%s elapsed_ms=%s",
            session_id,
            persisted_graph_state is not None,
            elapsed_ms(persisted_graph_started_at),
        )
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
                logger.info("[agent/ws] Running persisted graph state session=%s", session_id)
                await run_session_until_pause(
                    state=persisted_graph_state,
                    db=db,
                    event_handler=send_event,
                )
        else:
            logger.info("[agent/ws] Running new graph state session=%s", session_id)
            await run_session_until_pause(
                state=initial_state,
                db=db,
                event_handler=send_event,
            )

        while True:
            raw_message = await websocket.receive_json()
            message_type = raw_message.get("type")
            logger.info(
                "[agent/ws] Received payload session=%s type=%s keys=%s",
                session_id,
                message_type,
                sorted(raw_message) if isinstance(raw_message, dict) else None,
            )
            if message_type == "reprompt":
                try:
                    reprompt_payload = WebSocketRepromptPayload.model_validate(raw_message)
                except ValidationError as exc:
                    logger.warning(
                        "[agent/ws] Invalid reprompt payload for session=%s payload=%s",
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
                    and is_timeline_approval_message(reprompt_payload.prompt)
                ):
                    await approve_session_timeline(
                        session_id=str(session_id),
                        db=db,
                        event_handler=send_event,
                    )
                    continue
                resume_started_at = perf_counter()
                logger.info(
                    "[agent/ws] Resuming from reprompt session=%s prompt_chars=%s",
                    session_id,
                    len(reprompt_payload.prompt),
                )
                await resume_session_from_reprompt(
                    session_id=str(session_id),
                    prompt=reprompt_payload.prompt,
                    db=db,
                    event_handler=send_event,
                )
                logger.info(
                    "[agent/ws] Reprompt completed session=%s elapsed_ms=%s",
                    session_id,
                    elapsed_ms(resume_started_at),
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
        logger.exception("[agent/ws] Invalid websocket payload for session=%s", session_id)
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
        logger.info(
            "[agent/ws] Client disconnected session=%s elapsed_ms=%s",
            session_id,
            elapsed_ms(websocket_started_at),
        )
    except Exception:
        logger.exception("[agent/ws] Session websocket failed session=%s", session_id)
        await send_event(
            {
                "type": "error",
                "session_id": session_id,
                "detail": "Session websocket failed; check server logs for details.",
            }
        )
        await websocket.close(code=1011)


@router.websocket("/agent/voice/transcribe")
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


@router.websocket("/agent/voice/intent")
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

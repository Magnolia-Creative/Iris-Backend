from contextlib import suppress
import json
import logging
from time import perf_counter
from typing import Any

from fastapi import (
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.app import create_app
from app.api.context import context_id as _context_id
from app.api.intent_logging import (
    intent_context_log_summary as _intent_context_log_summary,
    intent_result_log_summary as _intent_result_log_summary,
)
from app.api.schemas.intent import VoiceIntentStartPayload
from app.api.schemas.projects import (
    AgentSessionCreatePayload,
    AgentSessionForProjectCreatePayload,
    ProjectCreatePayload,
)
from app.api.schemas.sessions import WebSocketRepromptPayload, WebSocketSessionStartPayload
from app.api.session_messages import is_timeline_approval_message as _is_timeline_approval_message
from app.api.timing import elapsed_ms as _elapsed_ms
from app.auth import (
    ClerkPrincipal,
    require_clerk_user,
    require_clerk_websocket_user,
    require_owned_project,
    require_owned_session,
)
from app.database import get_db
from app.automake.runtime import (
    approve_session_timeline,
    get_session_state,
    resume_session_from_reprompt,
    run_session_until_pause,
    set_session_state,
)
from app.services.session_graph_state_store import get_persisted_session_graph_state
from app.services.session_debug_store import initialize_session_debug, record_session_event
from app.services.session_state_builder import build_initial_state_from_session_payload
from app.services.realtime_transcription import (
    ALLOWED_TRANSCRIBE_MODELS,
    DEFAULT_TRANSCRIBE_MODEL,
    stream_transcription,
)
from app.intent_compiler.llm import IntentCompilerService
from app.intent_compiler.models import IntentCompileRequest, IntentCompilerContext
from app.intent_compiler.runs import create_intent_run, delete_intent_run, get_intent_run
from app.intent_compiler.transcripts import prepare_intent_transcript_context
from app.intent_compiler.voice import stream_voice_intent
from app.ui_workspace.models import UIWorkspacePlanRequest, UIWorkspacePlanResponse
from app.ui_workspace.planner import UIWorkspacePlannerService
from app.services.transcript_store import (
    get_persisted_session_data,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


app = create_app()


@app.post("/intent-runs")
async def create_intent_run_endpoint(
    payload: IntentCompileRequest,
    request: Request,
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    started_at = perf_counter()
    logger.info(
        "[intent-runs] Create requested prompt_chars=%s context=%s",
        len(payload.prompt),
        _intent_context_log_summary(payload.context),
    )
    project_id = _context_id(payload.context.projectId)
    if project_id is not None:
        ownership_started_at = perf_counter()
        logger.info("[intent-runs] Project ownership check starting project_id=%s", project_id)
        await require_owned_project(db, project_id=project_id, principal=principal)
        logger.info(
            "[intent-runs] Project ownership check completed project_id=%s elapsed_ms=%s",
            project_id,
            _elapsed_ms(ownership_started_at),
        )
    session_id = _context_id(payload.context.sessionId)
    if session_id is not None:
        ownership_started_at = perf_counter()
        logger.info("[intent-runs] Session ownership check starting session_id=%s", session_id)
        await require_owned_session(db, session_id=session_id, principal=principal)
        logger.info(
            "[intent-runs] Session ownership check completed session_id=%s elapsed_ms=%s",
            session_id,
            _elapsed_ms(ownership_started_at),
        )
    hydration_started_at = perf_counter()
    logger.info("[intent-runs] Context hydration starting prompt_chars=%s", len(payload.prompt))
    context, hydration_meta = await prepare_intent_transcript_context(
        prompt=payload.prompt,
        context=payload.context,
        db=db,
    )
    logger.info(
        "[intent-runs] Context hydrated prompt_chars=%s context=%s elapsed_ms=%s",
        len(payload.prompt),
        _intent_context_log_summary(context, hydration=hydration_meta),
        _elapsed_ms(hydration_started_at),
    )
    run = await create_intent_run(
        owner_user_id=principal.user_id,
        prompt=payload.prompt,
        context=context,
    )
    websocket_url = str(request.url_for("intent_run_websocket", run_id=run.run_id)).replace(
        "http://",
        "ws://",
        1,
    ).replace("https://", "wss://", 1)
    logger.info(
        "[intent-runs] Created run=%s websocket_url=%s elapsed_ms=%s",
        run.run_id,
        websocket_url,
        _elapsed_ms(started_at),
    )
    return {"run_id": run.run_id, "websocket_url": websocket_url}


@app.post("/projects/{project_id}/ui-workspace-plan")
async def create_ui_workspace_plan(
    project_id: int,
    payload: UIWorkspacePlanRequest,
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    started_at = perf_counter()
    await require_owned_project(db, project_id=project_id, principal=principal)
    context_project_id = _context_id(payload.context.projectId)
    if context_project_id is not None and context_project_id != project_id:
        raise HTTPException(
            status_code=400,
            detail="context.projectId must match the route project_id when provided.",
        )
    session_id = _context_id(payload.context.sessionId)
    if session_id is not None:
        await require_owned_session(db, session_id=session_id, principal=principal)

    logger.info(
        "[ui-workspace] Plan requested project_id=%s prompt_chars=%s",
        project_id,
        len(payload.prompt),
    )
    service = UIWorkspacePlannerService(use_llm=False)
    plan = await service.plan(payload)
    logger.info(
        "[ui-workspace] Plan completed project_id=%s workspace_id=%s slices=%s elapsed_ms=%s",
        project_id,
        plan.workspaceId,
        len(plan.intentSlices),
        _elapsed_ms(started_at),
    )
    return UIWorkspacePlanResponse(plan=plan)


@app.websocket("/ws/sessions/{session_id}")
async def session_websocket(
    websocket: WebSocket,
    session_id: int,
    db: AsyncSession = Depends(get_db),
):
    websocket_started_at = perf_counter()
    logger.info("[ws] Session websocket auth starting session=%s", session_id)
    principal = await require_clerk_websocket_user(websocket)
    logger.info("[ws] Session websocket ownership check starting session=%s user=%s", session_id, principal.user_id)
    await require_owned_session(db, session_id=session_id, principal=principal)
    await websocket.accept()
    logger.info("[ws] Session websocket accepted session=%s user=%s", session_id, principal.user_id)

    async def send_event(payload: dict[str, Any]) -> None:
        tracked_payload = record_session_event(str(session_id), payload)
        event_type = str(tracked_payload.get("type") or "<missing>")
        logger.info(
            "[ws] Sending event session=%s type=%s keys=%s",
            session_id,
            event_type,
            sorted(tracked_payload),
        )
        await websocket.send_text(json.dumps(tracked_payload))

    try:
        raw_message = await websocket.receive_json()
        logger.info(
            "[ws] Received initial payload session=%s type=%s keys=%s",
            session_id,
            raw_message.get("type") if isinstance(raw_message, dict) else None,
            sorted(raw_message) if isinstance(raw_message, dict) else None,
        )
        start_payload = WebSocketSessionStartPayload.model_validate(raw_message)
        persisted_started_at = perf_counter()
        logger.info("[ws] Loading persisted session data session=%s", session_id)
        persisted_session_data = await get_persisted_session_data(
            db=db,
            session_id=session_id,
            include_ingest_details=True,
        )
        logger.info(
            "[ws] Loaded persisted session data session=%s found=%s elapsed_ms=%s",
            session_id,
            persisted_session_data is not None,
            _elapsed_ms(persisted_started_at),
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
            "[ws] Initial state built session=%s clips=%s prompt_chars=%s",
            session_id,
            len(initial_state.get("clips", [])),
            len(start_payload.user_prompt),
        )
        persisted_graph_started_at = perf_counter()
        logger.info("[ws] Loading persisted graph state session=%s", session_id)
        persisted_graph_state = await get_persisted_session_graph_state(db=db, session_id=session_id)
        logger.info(
            "[ws] Loaded persisted graph state session=%s found=%s elapsed_ms=%s",
            session_id,
            persisted_graph_state is not None,
            _elapsed_ms(persisted_graph_started_at),
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
                logger.info("[ws] Running persisted graph state session=%s", session_id)
                await run_session_until_pause(
                    state=persisted_graph_state,
                    db=db,
                    event_handler=send_event,
                )
        else:
            logger.info("[ws] Running new graph state session=%s", session_id)
            await run_session_until_pause(
                state=initial_state,
                db=db,
                event_handler=send_event,
            )

        while True:
            raw_message = await websocket.receive_json()
            message_type = raw_message.get("type")
            logger.info(
                "[ws] Received payload session=%s type=%s keys=%s",
                session_id,
                message_type,
                sorted(raw_message) if isinstance(raw_message, dict) else None,
            )
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
                resume_started_at = perf_counter()
                logger.info(
                    "[ws] Resuming from reprompt session=%s prompt_chars=%s",
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
                    "[ws] Reprompt completed session=%s elapsed_ms=%s",
                    session_id,
                    _elapsed_ms(resume_started_at),
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
        logger.info("[ws] Client disconnected session=%s elapsed_ms=%s", session_id, _elapsed_ms(websocket_started_at))
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
    websocket_started_at = perf_counter()
    logger.info("[intent-runs] WebSocket auth starting run=%s", run_id)
    principal = await require_clerk_websocket_user(websocket)
    await websocket.accept()
    logger.info("[intent-runs] WebSocket accepted run=%s user=%s", run_id, principal.user_id)
    lookup_started_at = perf_counter()
    logger.info("[intent-runs] Run lookup starting run=%s", run_id)
    run = await get_intent_run(run_id)
    logger.info(
        "[intent-runs] Run lookup completed run=%s found=%s elapsed_ms=%s",
        run_id,
        run is not None,
        _elapsed_ms(lookup_started_at),
    )
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
    if run.owner_user_id != principal.user_id:
        logger.warning("[intent-runs] WebSocket run not owned run=%s", run_id)
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
        compile_started_at = perf_counter()
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
        logger.info(
            "[intent-runs] WebSocket closed normally run=%s compile_elapsed_ms=%s total_elapsed_ms=%s",
            run_id,
            _elapsed_ms(compile_started_at),
            _elapsed_ms(websocket_started_at),
        )
    except WebSocketDisconnect:
        logger.info("[intent-runs] Client disconnected run=%s elapsed_ms=%s", run_id, _elapsed_ms(websocket_started_at))
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
        await delete_intent_run(run_id)


@app.websocket("/ws/transcribe")
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


@app.websocket("/ws/intent/voice")
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

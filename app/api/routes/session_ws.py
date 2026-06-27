import json
import logging
from time import perf_counter
from typing import Any

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.sessions import WebSocketRepromptPayload, WebSocketSessionStartPayload
from app.api.session_messages import is_timeline_approval_message
from app.api.timing import elapsed_ms
from app.auth import require_clerk_websocket_user, require_owned_session
from app.agent.automake.runtime import (
    approve_session_timeline,
    get_session_state,
    resume_session_from_reprompt,
    run_session_until_pause,
    set_session_state,
)
from app.database import get_db
from app.services.session_debug_store import initialize_session_debug, record_session_event
from app.services.session_graph_state_store import get_persisted_session_graph_state
from app.services.session_state_builder import build_initial_state_from_session_payload
from app.services.transcript_store import get_persisted_session_data


logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/sessions/{session_id}")
async def session_websocket(
    websocket: WebSocket,
    session_id: int,
    db: AsyncSession = Depends(get_db),
):
    websocket_started_at = perf_counter()
    logger.info("[ws] Session websocket auth starting session=%s", session_id)
    principal = await require_clerk_websocket_user(websocket)
    logger.info(
        "[ws] Session websocket ownership check starting session=%s user=%s",
        session_id,
        principal.user_id,
    )
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
            "[ws] Initial state built session=%s clips=%s prompt_chars=%s",
            session_id,
            len(initial_state.get("clips", [])),
            len(start_payload.user_prompt),
        )
        persisted_graph_started_at = perf_counter()
        logger.info("[ws] Loading persisted graph state session=%s", session_id)
        persisted_graph_state = await get_persisted_session_graph_state(
            db=db, session_id=session_id
        )
        logger.info(
            "[ws] Loaded persisted graph state session=%s found=%s elapsed_ms=%s",
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
        logger.info(
            "[ws] Client disconnected session=%s elapsed_ms=%s",
            session_id,
            elapsed_ms(websocket_started_at),
        )
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

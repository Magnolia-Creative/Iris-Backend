from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
import json
import logging
from typing import Any
from typing import Literal

from fastapi import Body, Depends, FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Base, engine, get_db
from app.graph.runtime import (
    approve_session_timeline,
    get_session_state,
    resume_session_from_reprompt,
    run_session_until_pause,
    set_session_state,
)
from app.services.session_graph_state_store import get_persisted_session_graph_state
from app.services.session_ingest import (
    cancel_clip_processing,
    create_agent_session,
    ingest_session_clips,
    process_project_clips,
)
from app.services.session_state_builder import build_initial_state_from_session_payload
from app.services.transcript_store import get_persisted_session_data
from app import models  # noqa: F401


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


def _is_timeline_approval_message(prompt: str) -> bool:
    normalized = " ".join(prompt.lower().strip().split())
    rejection_markers = {"do not approve", "don't approve", "not approved", "reject", "decline"}
    if any(marker in normalized for marker in rejection_markers):
        return False

    if normalized in {"yes", "yep", "yeah", "ok", "okay"}:
        return True

    approval_markers = {"approve", "approved", "looks good", "go ahead", "ship it"}
    return any(marker in normalized for marker in approval_markers)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Keep simple table creation for local development; use Alembic for production migrations.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/db-health")
async def db_health(db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("SELECT 1"))
    return {"database": "ok", "result": result.scalar_one()}


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
        await websocket.send_text(json.dumps(payload))

    try:
        raw_message = await websocket.receive_json()
        start_payload = WebSocketSessionStartPayload.model_validate(raw_message)
        persisted_session_data = await get_persisted_session_data(
            db=db,
            session_id=session_id,
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

        initial_state = build_initial_state_from_session_payload(
            session_id=str(session_id),
            session_payload=persisted_session_data,
            user_prompt=start_payload.user_prompt,
        )

        await send_event(
            {
                "type": "session_started",
                "session_id": session_id,
                "project_id": persisted_session_data["project_id"],
                "uploaded_count": persisted_session_data["uploaded_count"],
            }
        )
        persisted_graph_state = await get_persisted_session_graph_state(db=db, session_id=session_id)
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

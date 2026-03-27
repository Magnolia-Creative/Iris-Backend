from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
import json
import logging
from typing import Any
from typing import Literal

from fastapi import Depends, FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Base, engine, get_db
from app.graph.runtime import (
    approve_session_timeline,
    get_session_state,
    resume_session_from_reprompt,
    run_session_until_pause,
)
from app.services.session_ingest import ingest_session_clips
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
    db: AsyncSession = Depends(get_db),
    session_name: str | None = None,
):
    ingest_result = await ingest_session_clips(db, videos, session_name=session_name)
    return {
        "session_id": ingest_result["session_id"],
        "session_name": ingest_result["session_name"],
        "session_status": ingest_result["session_status"],
        "uploaded_count": ingest_result["uploaded_count"],
        "videos": ingest_result["videos"],
    }


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

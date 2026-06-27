from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import ClerkPrincipal, require_clerk_user, require_owned_session
from app.agent.automake.runtime import get_session_state
from app.database import get_db
from app.services.session_debug_store import build_session_debug_snapshot, initialize_session_debug
from app.services.session_graph_state_store import get_persisted_session_graph_state
from app.services.session_ingest import ingest_session_clips
from app.services.transcript_store import get_persisted_session_data


router = APIRouter()


@router.post("/sessions/upload")
async def create_session_from_upload(
    videos: list[UploadFile] = File(...),
    local_keys: list[str] = Form(..., alias="local_key"),
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
    session_name: str | None = None,
):
    ingest_result = await ingest_session_clips(
        db,
        videos,
        local_keys=local_keys,
        owner_user_id=principal.user_id,
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


@router.get("/sessions/{session_id}")
async def get_session_status(
    session_id: int,
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    await require_owned_session(db, session_id=session_id, principal=principal)
    session_payload = await get_persisted_session_data(
        db=db,
        session_id=session_id,
        include_ingest_details=False,
    )
    if session_payload is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return session_payload


@router.get("/sessions/{session_id}/debug")
async def get_session_debug(
    session_id: int,
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    await require_owned_session(db, session_id=session_id, principal=principal)
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

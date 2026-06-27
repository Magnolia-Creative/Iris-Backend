import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import (
    ClerkPrincipal,
    require_clerk_user,
    require_owned_project,
    require_owned_project_clip,
    require_owned_session,
)
from app.database import get_db
from app.services.session_ingest import cancel_clip_processing, process_project_clips
from app.services.transcript_store import get_persisted_project_data
from app.services.visual_frame_payload import build_visual_frames_by_local_key


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/projects/{project_id}/sources")
async def process_project_source_batch(
    project_id: int,
    videos: list[UploadFile] = File(...),
    local_keys: list[str] = Form(..., alias="local_key"),
    session_id: int | None = Form(None),
    visual_frame_manifest: str | None = Form(None),
    visual_frames: list[UploadFile] = File(default=[]),
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    logger.info(
        "[sources/process] http project_id=%s session_id=%s videos=%s manifest=%s visual_frame_parts=%s",
        project_id,
        session_id,
        len(videos),
        visual_frame_manifest is not None and bool(visual_frame_manifest.strip()),
        len(visual_frames),
    )
    await require_owned_project(db, project_id=project_id, principal=principal)
    if session_id is not None:
        await require_owned_session(db, session_id=session_id, principal=principal)
    visual_map = await build_visual_frames_by_local_key(
        manifest_raw=visual_frame_manifest,
        visual_frame_files=visual_frames,
    )
    return await process_project_clips(
        db,
        project_id=project_id,
        session_id=session_id,
        videos=videos,
        local_keys=local_keys,
        visual_frames_by_local_key=visual_map if visual_map else None,
    )


@router.get("/projects/{project_id}/sources")
async def get_project_sources(
    project_id: int,
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    await require_owned_project(db, project_id=project_id, principal=principal)
    payload = await get_persisted_project_data(db, project_id, include_ingest_details=False)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found.")
    return payload


@router.delete("/projects/{project_id}/sources/{local_key}")
async def cancel_project_source(
    project_id: int,
    local_key: str,
    session_id: int | None = Query(None),
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    await require_owned_project_clip(
        db,
        project_id=project_id,
        local_key=local_key,
        principal=principal,
    )
    if session_id is not None:
        await require_owned_session(db, session_id=session_id, principal=principal)
    return await cancel_clip_processing(
        db,
        project_id=project_id,
        local_key=local_key,
        session_id=session_id,
    )

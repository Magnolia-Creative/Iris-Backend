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
from app.services.transcript_store import get_clip_captions_payload, get_persisted_project_data
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


@router.get("/projects/{project_id}/sources/{local_key}/transcript")
async def get_project_source_transcript(
    project_id: int,
    local_key: str,
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    """Return transcript segments as `sentences` for a single project source."""
    logger.info(
        "[sources/transcript] request project_id=%s local_key=%s",
        project_id,
        local_key,
    )
    await require_owned_project(db, project_id=project_id, principal=principal)
    status, body = await get_clip_captions_payload(db, project_id=project_id, local_key=local_key)
    if status == "project_not_found":
        logger.warning(
            "[sources/transcript] project not found project_id=%s local_key=%s",
            project_id,
            local_key,
        )
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found.")
    if status == "clip_not_found":
        logger.warning(
            "[sources/transcript] source not found project_id=%s local_key=%s",
            project_id,
            local_key,
        )
        raise HTTPException(
            status_code=404,
            detail=f"No source with local_key={local_key!r} for project {project_id}.",
        )
    if status == "transcript_not_ready":
        logger.info(
            "[sources/transcript] transcript not ready project_id=%s local_key=%s body=%s",
            project_id,
            local_key,
            body,
        )
        raise HTTPException(
            status_code=409,
            detail="Transcript not available yet for this source.",
        )
    logger.info(
        "[sources/transcript] success project_id=%s local_key=%s clip_id=%s transcript_id=%s sentence_count=%s processing_status=%s",
        project_id,
        local_key,
        body.get("clip_id") if isinstance(body, dict) else None,
        body.get("transcript_id") if isinstance(body, dict) else None,
        len(body.get("sentences") or []) if isinstance(body, dict) else 0,
        body.get("processing_status") if isinstance(body, dict) else None,
    )
    return body


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

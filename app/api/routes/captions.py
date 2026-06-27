import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import ClerkPrincipal, require_clerk_user, require_owned_project
from app.database import get_db
from app.services.transcript_store import get_clip_captions_payload


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/captions")
async def get_captions_for_clip(
    project_id: int = Query(..., description="Project id"),
    local_key: str = Query(..., description="Clip local_key matching ingest"),
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    """Return transcript segments as `sentences` for a single clip (captions client)."""
    logger.info(
        "[captions] request project_id=%s local_key=%s",
        project_id,
        local_key,
    )
    await require_owned_project(db, project_id=project_id, principal=principal)
    status, body = await get_clip_captions_payload(db, project_id=project_id, local_key=local_key)
    if status == "project_not_found":
        logger.warning(
            "[captions] project not found project_id=%s local_key=%s",
            project_id,
            local_key,
        )
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found.")
    if status == "clip_not_found":
        logger.warning(
            "[captions] clip not found project_id=%s local_key=%s",
            project_id,
            local_key,
        )
        raise HTTPException(
            status_code=404,
            detail=f"No clip with local_key={local_key!r} for project {project_id}.",
        )
    if status == "transcript_not_ready":
        logger.info(
            "[captions] transcript not ready project_id=%s local_key=%s body=%s",
            project_id,
            local_key,
            body,
        )
        raise HTTPException(
            status_code=409,
            detail="Transcript not available yet for this clip.",
        )
    logger.info(
        "[captions] success project_id=%s local_key=%s clip_id=%s transcript_id=%s sentence_count=%s processing_status=%s",
        project_id,
        local_key,
        body.get("clip_id") if isinstance(body, dict) else None,
        body.get("transcript_id") if isinstance(body, dict) else None,
        len(body.get("sentences") or []) if isinstance(body, dict) else 0,
        body.get("processing_status") if isinstance(body, dict) else None,
    )
    return body

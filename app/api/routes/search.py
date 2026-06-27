import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.search import SearchRequest
from app.auth import ClerkPrincipal, require_clerk_user, require_owned_project
from app.database import get_db
from app.services.semantic_search_service import search_project_semantic
from app.services.transcript_search_service import search_project_transcript


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/projects/{project_id}/semantic-search")
async def semantic_search_project(
    project_id: int,
    payload: SearchRequest,
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    preview = (payload.query or "").strip().replace("\n", " ")[:120]
    logger.info(
        "[semantic_search] http project_id=%s limit=%s query_preview=%r",
        project_id,
        payload.limit,
        preview,
    )
    await require_owned_project(db, project_id=project_id, principal=principal)
    return await search_project_semantic(
        db,
        project_id=project_id,
        query=payload.query,
        limit=payload.limit,
    )


@router.post("/projects/{project_id}/transcript-search")
async def transcript_search_project(
    project_id: int,
    payload: SearchRequest,
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    preview = (payload.query or "").strip().replace("\n", " ")[:120]
    logger.info(
        "[transcript_search] http project_id=%s limit=%s query_preview=%r",
        project_id,
        payload.limit,
        preview,
    )
    await require_owned_project(db, project_id=project_id, principal=principal)
    return await search_project_transcript(
        db,
        project_id=project_id,
        query=payload.query,
        limit=payload.limit,
    )

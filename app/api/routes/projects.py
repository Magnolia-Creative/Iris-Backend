from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.projects import (
    AgentSessionCreatePayload,
    AgentSessionForProjectCreatePayload,
    ProjectCreatePayload,
)
from app.auth import ClerkPrincipal, require_clerk_user, require_owned_project
from app.database import get_db
from app.services.session_ingest import (
    create_agent_session,
    create_agent_session_for_project,
    create_project,
)
from app.services.transcript_store import get_persisted_project_data


router = APIRouter()


@router.post("/projects")
async def create_project_endpoint(
    payload: ProjectCreatePayload = Body(default=ProjectCreatePayload()),
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    return await create_project(db, owner_user_id=principal.user_id, name=payload.name)


@router.post("/projects/{project_id}/agent-sessions")
async def create_agent_session_for_existing_project(
    project_id: int,
    payload: AgentSessionForProjectCreatePayload = Body(
        default=AgentSessionForProjectCreatePayload()
    ),
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    return await create_agent_session_for_project(
        db,
        owner_user_id=principal.user_id,
        project_id=project_id,
        session_name=payload.session_name,
    )


@router.get("/projects/{project_id}/clips/status")
async def get_project_clips_status(
    project_id: int,
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    await require_owned_project(db, project_id=project_id, principal=principal)
    payload = await get_persisted_project_data(db, project_id, include_ingest_details=False)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found.")
    return payload


@router.post("/projects/agent-sessions")
async def create_project_agent_session(
    payload: AgentSessionCreatePayload = Body(default=AgentSessionCreatePayload()),
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    return await create_agent_session(
        db,
        owner_user_id=principal.user_id,
        session_name=payload.session_name,
        project_name=payload.project_name,
    )

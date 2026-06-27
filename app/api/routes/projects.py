from fastapi import APIRouter, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.projects import (
    ProjectCreatePayload,
)
from app.auth import ClerkPrincipal, require_clerk_user
from app.database import get_db
from app.services.session_ingest import create_project


router = APIRouter()


@router.post("/projects")
async def create_project_endpoint(
    payload: ProjectCreatePayload = Body(default=ProjectCreatePayload()),
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    return await create_project(db, owner_user_id=principal.user_id, name=payload.name)



from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.clerk import ClerkPrincipal
from app.database import models


def _not_found(resource: str, resource_id: int | str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{resource} {resource_id} not found.")


async def require_owned_project(
    db: AsyncSession,
    *,
    project_id: int,
    principal: ClerkPrincipal,
) -> models.Project:
    result = await db.execute(
        select(models.Project).where(
            models.Project.id == project_id,
            models.Project.clerk_user_id == principal.user_id,
        )
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise _not_found("Project", project_id)
    return project


async def require_owned_session(
    db: AsyncSession,
    *,
    session_id: int,
    principal: ClerkPrincipal,
) -> models.Session:
    result = await db.execute(
        select(models.Session).where(
            models.Session.id == session_id,
            models.Session.clerk_user_id == principal.user_id,
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise _not_found("Session", session_id)
    return session


async def require_owned_project_clip(
    db: AsyncSession,
    *,
    project_id: int,
    local_key: str,
    principal: ClerkPrincipal,
) -> models.Clip:
    await require_owned_project(db, project_id=project_id, principal=principal)
    result = await db.execute(
        select(models.Clip).where(
            models.Clip.project_id == project_id,
            models.Clip.local_key == local_key,
        )
    )
    clip = result.scalar_one_or_none()
    if clip is None:
        raise _not_found("Clip", local_key)
    return clip

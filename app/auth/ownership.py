import logging

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.clerk import ClerkPrincipal
from app.database import models


logger = logging.getLogger(__name__)


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
        logger.warning(
            "[auth] Ownership rejected resource=project project_id=%s user_id=%s",
            project_id,
            principal.user_id,
        )
        raise _not_found("Project", project_id)
    logger.info(
        "[auth] Ownership accepted resource=project project_id=%s user_id=%s",
        project_id,
        principal.user_id,
    )
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
        logger.warning(
            "[auth] Ownership rejected resource=session session_id=%s user_id=%s",
            session_id,
            principal.user_id,
        )
        raise _not_found("Session", session_id)
    logger.info(
        "[auth] Ownership accepted resource=session session_id=%s user_id=%s",
        session_id,
        principal.user_id,
    )
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
        logger.warning(
            "[auth] Ownership rejected resource=clip project_id=%s local_key=%s user_id=%s",
            project_id,
            local_key,
            principal.user_id,
        )
        raise _not_found("Clip", local_key)
    logger.info(
        "[auth] Ownership accepted resource=clip project_id=%s local_key=%s user_id=%s",
        project_id,
        local_key,
        principal.user_id,
    )
    return clip

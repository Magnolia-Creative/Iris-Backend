from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import models
from app.automake.state import SessionGraphState, normalize_session_graph_state


async def get_persisted_session_graph_state(
    db: AsyncSession, session_id: int
) -> SessionGraphState | None:
    result = await db.execute(
        select(models.Session.graph_state).where(models.Session.id == session_id)
    )
    graph_state = result.scalar_one_or_none()
    if not isinstance(graph_state, dict):
        return None
    return normalize_session_graph_state(graph_state, session_id=session_id)


async def persist_session_graph_state(
    db: AsyncSession,
    session_id: int,
    state: SessionGraphState,
) -> None:
    result = await db.execute(select(models.Session).where(models.Session.id == session_id))
    session = result.scalar_one_or_none()
    if session is None:
        raise KeyError(f"Unknown session_id: {session_id}")

    session.graph_state = dict(normalize_session_graph_state(state, session_id=session_id))
    await db.commit()

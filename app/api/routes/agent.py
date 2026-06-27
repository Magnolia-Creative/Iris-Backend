import json
import logging
from time import perf_counter

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.automake.runtime import get_session_state
from app.agent.intent import IntentAgentResponse, run_intent_agent
from app.agent.intent.editing.transcripts import prepare_intent_transcript_context
from app.api.context import context_id
from app.api.intent_logging import intent_context_log_summary, intent_result_log_summary
from app.api.schemas.agent import (
    AgentRunCreatePayload,
    AutomakeAgentRunCreatePayload,
    IntentAgentRunCreatePayload,
)
from app.api.timing import elapsed_ms
from app.auth import ClerkPrincipal, require_clerk_user, require_owned_project, require_owned_session
from app.database import get_db
from app.services.session_debug_store import build_session_debug_snapshot, initialize_session_debug
from app.services.session_graph_state_store import get_persisted_session_graph_state
from app.services.session_ingest import create_agent_session, create_agent_session_for_project
from app.services.transcript_store import get_persisted_session_data


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/agent/runs")
async def create_agent_run(
    payload: AgentRunCreatePayload = Body(...),
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    if isinstance(payload, IntentAgentRunCreatePayload):
        return await _run_intent_agent(payload, principal=principal, db=db)
    if isinstance(payload, AutomakeAgentRunCreatePayload):
        return await _create_automake_agent_run(payload, principal=principal, db=db)
    raise HTTPException(status_code=422, detail="Unsupported agent run kind.")


async def _run_intent_agent(
    payload: IntentAgentRunCreatePayload,
    *,
    principal: ClerkPrincipal,
    db: AsyncSession,
) -> IntentAgentResponse:
    started_at = perf_counter()
    logger.info(
        "[intent-agent] Run requested user=%s prompt_chars=%s context=%s",
        principal.user_id,
        len(payload.prompt),
        intent_context_log_summary(payload.context),
    )
    project_id = context_id(payload.context.projectId)
    if project_id is not None:
        ownership_started_at = perf_counter()
        logger.info("[intent-agent] Project ownership check starting project_id=%s", project_id)
        await require_owned_project(db, project_id=project_id, principal=principal)
        logger.info(
            "[intent-agent] Project ownership check completed project_id=%s elapsed_ms=%s",
            project_id,
            elapsed_ms(ownership_started_at),
        )

    session_id = context_id(payload.context.sessionId)
    if session_id is not None:
        ownership_started_at = perf_counter()
        logger.info("[intent-agent] Session ownership check starting session_id=%s", session_id)
        await require_owned_session(db, session_id=session_id, principal=principal)
        logger.info(
            "[intent-agent] Session ownership check completed session_id=%s elapsed_ms=%s",
            session_id,
            elapsed_ms(ownership_started_at),
        )

    hydration_started_at = perf_counter()
    logger.info("[intent-agent] Context hydration starting prompt_chars=%s", len(payload.prompt))
    context, hydration_meta = await prepare_intent_transcript_context(
        prompt=payload.prompt,
        context=payload.context,
        db=db,
    )
    logger.info(
        "[intent-agent] Context hydrated prompt_chars=%s context=%s elapsed_ms=%s",
        len(payload.prompt),
        intent_context_log_summary(context, hydration=hydration_meta),
        elapsed_ms(hydration_started_at),
    )

    result = await run_intent_agent(
        payload,
        context=context,
        hydration_meta=hydration_meta,
    )
    result_payload = json.loads(result.model_dump_json(by_alias=True))
    logger.info(
        "[intent-agent] Run completed user=%s summary=%s workspace_id=%s elapsed_ms=%s",
        principal.user_id,
        intent_result_log_summary(result_payload.get("edit") or {}),
        result.ui.workspaceId,
        elapsed_ms(started_at),
    )
    return result


async def _create_automake_agent_run(
    payload: AutomakeAgentRunCreatePayload,
    *,
    principal: ClerkPrincipal,
    db: AsyncSession,
):
    if payload.project_id is not None:
        return await create_agent_session_for_project(
            db,
            owner_user_id=principal.user_id,
            project_id=payload.project_id,
            session_name=payload.session_name,
        )
    return await create_agent_session(
        db,
        owner_user_id=principal.user_id,
        session_name=payload.session_name,
        project_name=payload.project_name,
    )


@router.get("/agent/runs/{run_id}")
async def get_agent_run_status(
    run_id: int,
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    await require_owned_session(db, session_id=run_id, principal=principal)
    session_payload = await get_persisted_session_data(
        db=db,
        session_id=run_id,
        include_ingest_details=False,
    )
    if session_payload is None:
        raise HTTPException(status_code=404, detail=f"Agent run {run_id} not found.")
    return session_payload


@router.get("/agent/runs/{run_id}/debug")
async def get_agent_run_debug(
    run_id: int,
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
    await require_owned_session(db, session_id=run_id, principal=principal)
    session_payload = await get_persisted_session_data(
        db=db,
        session_id=run_id,
        include_ingest_details=True,
    )
    if session_payload is None:
        raise HTTPException(status_code=404, detail=f"Agent run {run_id} not found.")

    persisted_graph_state = await get_persisted_session_graph_state(db=db, session_id=run_id)
    state = get_session_state(str(run_id)) or persisted_graph_state
    initialize_session_debug(str(run_id), state)
    return build_session_debug_snapshot(
        session_id=run_id,
        session_payload=session_payload,
        state=state,
    )

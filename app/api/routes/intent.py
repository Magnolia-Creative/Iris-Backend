import json
import logging
from time import perf_counter

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.intent import IntentAgentRequest, IntentAgentResponse, run_intent_agent
from app.agent.intent.editing.transcripts import prepare_intent_transcript_context
from app.api.context import context_id
from app.api.intent_logging import intent_context_log_summary, intent_result_log_summary
from app.api.timing import elapsed_ms
from app.auth import ClerkPrincipal, require_clerk_user, require_owned_project, require_owned_session
from app.database import get_db


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/agent/intent", response_model=IntentAgentResponse)
async def run_intent_agent_endpoint(
    payload: IntentAgentRequest,
    principal: ClerkPrincipal = Depends(require_clerk_user),
    db: AsyncSession = Depends(get_db),
):
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

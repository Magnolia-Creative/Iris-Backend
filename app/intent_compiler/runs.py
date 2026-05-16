from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from app.intent_compiler.models import IntentCompilerContext
from app.services.transcript_cache import get_redis_client


INTENT_RUN_TTL_SECONDS = 15 * 60


@dataclass(frozen=True)
class IntentRun:
    run_id: str
    prompt: str
    context: IntentCompilerContext
    created_at: datetime


def _intent_run_key(run_id: str) -> str:
    return f"intent-run:{run_id}"


async def create_intent_run(*, prompt: str, context: IntentCompilerContext) -> IntentRun:
    run = IntentRun(
        run_id=str(uuid4()),
        prompt=prompt,
        context=context,
        created_at=datetime.now(tz=UTC),
    )
    redis = get_redis_client()
    await redis.set(
        _intent_run_key(run.run_id),
        json.dumps(
            {
                "run_id": run.run_id,
                "prompt": run.prompt,
                "context": run.context.model_dump(mode="json", by_alias=True),
                "created_at": run.created_at.isoformat(),
            }
        ),
        ex=INTENT_RUN_TTL_SECONDS,
    )
    return run


async def get_intent_run(run_id: str) -> IntentRun | None:
    redis = get_redis_client()
    payload = await redis.get(_intent_run_key(run_id))
    if payload is None:
        return None
    data = json.loads(payload)
    return IntentRun(
        run_id=str(data["run_id"]),
        prompt=str(data["prompt"]),
        context=IntentCompilerContext.model_validate(data["context"]),
        created_at=datetime.fromisoformat(str(data["created_at"])),
    )


async def delete_intent_run(run_id: str) -> None:
    redis = get_redis_client()
    await redis.delete(_intent_run_key(run_id))


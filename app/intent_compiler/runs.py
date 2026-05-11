from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from app.intent_compiler.models import IntentCompilerContext


@dataclass(frozen=True)
class IntentRun:
    run_id: str
    prompt: str
    context: IntentCompilerContext
    created_at: datetime


_runs: dict[str, IntentRun] = {}


def create_intent_run(*, prompt: str, context: IntentCompilerContext) -> IntentRun:
    run = IntentRun(
        run_id=str(uuid4()),
        prompt=prompt,
        context=context,
        created_at=datetime.now(tz=UTC),
    )
    _runs[run.run_id] = run
    return run


def get_intent_run(run_id: str) -> IntentRun | None:
    return _runs.get(run_id)


def delete_intent_run(run_id: str) -> None:
    _runs.pop(run_id, None)


from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any

from app.agent.intent.editing.graph import build_editing_intent_graph
from app.agent.intent.models import (
    IntentAgentMeta,
    IntentAgentRequest,
    IntentAgentResponse,
    IntentAgentTiming,
)
from app.agent.intent.ui.graph import build_ui_intent_graph
from app.agent.intent.ui.planner import IntentUIPlannerService
from app.agent.intent.editing.models import IntentCompilerContext
from app.agent.intent.editing.service import IntentCompilerService


def _elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


async def run_intent_agent(
    request: IntentAgentRequest,
    *,
    context: IntentCompilerContext | None = None,
    hydration_meta: dict[str, Any] | None = None,
    intent_compiler_service: IntentCompilerService | None = None,
    ui_intent_planner_service: Any | None = None,
    editing_graph: Any | None = None,
    ui_graph: Any | None = None,
) -> IntentAgentResponse:
    started_at = perf_counter()
    effective_context = context or request.context
    compiler = intent_compiler_service or IntentCompilerService()
    planner = ui_intent_planner_service or IntentUIPlannerService()
    editing_workflow = editing_graph or build_editing_intent_graph()
    ui_workflow = ui_graph or build_ui_intent_graph()

    edit_state, ui_state = await asyncio.gather(
        editing_workflow.ainvoke(
            {
                "prompt": request.prompt,
                "context": effective_context,
            },
            config={"configurable": {"intent_compiler_service": compiler}},
        ),
        ui_workflow.ainvoke(
            {
                "prompt": request.prompt,
                "context": effective_context,
                "editor_context": request.editorContext,
                "current_workspace_id": request.currentWorkspaceId,
            },
            config={"configurable": {"ui_intent_planner_service": planner}},
        ),
    )

    edit_result = edit_state.get("result")
    ui_plan = ui_state.get("plan")
    if edit_result is None:
        raise RuntimeError("Intent editing graph did not return an edit result.")
    if ui_plan is None:
        raise RuntimeError("Intent UI graph did not return a UI plan.")

    return IntentAgentResponse(
        edit=edit_result,
        ui=ui_plan,
        meta=IntentAgentMeta(
            hydration=hydration_meta or {},
            edit_events=edit_state.get("events") or [],
            ui_events=ui_state.get("events") or [],
            timings=[
                IntentAgentTiming(
                    branch="edit",
                    elapsed_ms=int(edit_state.get("elapsed_ms") or 0),
                ),
                IntentAgentTiming(
                    branch="ui",
                    elapsed_ms=int(ui_state.get("elapsed_ms") or 0),
                ),
                IntentAgentTiming(branch="total", elapsed_ms=_elapsed_ms(started_at)),
            ],
        ),
    )


class IntentAgentService:
    def __init__(
        self,
        *,
        intent_compiler_service: IntentCompilerService | None = None,
        ui_intent_planner_service: Any | None = None,
        editing_graph: Any | None = None,
        ui_graph: Any | None = None,
    ) -> None:
        self.intent_compiler_service = intent_compiler_service or IntentCompilerService()
        self.ui_intent_planner_service = ui_intent_planner_service or IntentUIPlannerService()
        self.editing_graph = editing_graph
        self.ui_graph = ui_graph

    async def run(
        self,
        request: IntentAgentRequest,
        *,
        context: IntentCompilerContext | None = None,
        hydration_meta: dict[str, Any] | None = None,
    ) -> IntentAgentResponse:
        return await run_intent_agent(
            request,
            context=context,
            hydration_meta=hydration_meta,
            intent_compiler_service=self.intent_compiler_service,
            ui_intent_planner_service=self.ui_intent_planner_service,
            editing_graph=self.editing_graph,
            ui_graph=self.ui_graph,
        )

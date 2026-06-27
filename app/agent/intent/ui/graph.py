from functools import lru_cache
from time import perf_counter
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.agent.intent.ui.models import IntentUIPlanRequest, UIIntentGraphState
from app.agent.intent.ui.planner import IntentUIPlannerService


def _elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


def _get_configurable(config: RunnableConfig | None) -> dict[str, Any]:
    if not config:
        return {}
    configurable = config.get("configurable")
    return configurable if isinstance(configurable, dict) else {}


async def plan_ui_node(
    state: UIIntentGraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    planner = _get_configurable(config).get("ui_intent_planner_service")
    if planner is None:
        planner = IntentUIPlannerService()

    events = [{"type": "ui_planner_started", "status": "Planning workspace."}]
    plan = await planner.plan(
        IntentUIPlanRequest(
            prompt=state["prompt"],
            context=state["context"],
            editorContext=state.get("editor_context"),
            intentResult=state.get("intent_result"),
            currentWorkspaceId=state.get("current_workspace_id"),
        )
    )
    events.append(
        {
            "type": "ui_planner_completed",
            "workspace_id": plan.workspaceId,
            "intent_slices": len(plan.intentSlices),
        }
    )
    return {
        "plan": plan,
        "events": events,
        "elapsed_ms": _elapsed_ms(started_at),
    }


@lru_cache(maxsize=1)
def build_ui_intent_graph():
    graph = StateGraph(UIIntentGraphState)
    graph.add_node("plan_ui", plan_ui_node)
    graph.add_edge(START, "plan_ui")
    graph.add_edge("plan_ui", END)
    return graph.compile()

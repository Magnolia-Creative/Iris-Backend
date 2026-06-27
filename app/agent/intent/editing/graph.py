from functools import lru_cache
from time import perf_counter
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.agent.intent.editing.models import EditingIntentGraphState
from app.agent.intent.editing.service import IntentCompilerService


def _elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


def _get_configurable(config: RunnableConfig | None) -> dict[str, Any]:
    if not config:
        return {}
    configurable = config.get("configurable")
    return configurable if isinstance(configurable, dict) else {}


async def compile_edit_node(
    state: EditingIntentGraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    compiler = _get_configurable(config).get("intent_compiler_service")
    if compiler is None:
        compiler = IntentCompilerService()

    events: list[dict[str, Any]] = []

    async def collect_event(payload: dict[str, Any]) -> None:
        events.append(dict(payload))

    result = await compiler.compile_prompt(
        prompt=state["prompt"],
        context=state["context"],
        event_handler=collect_event,
    )
    return {
        "result": result,
        "events": events,
        "elapsed_ms": _elapsed_ms(started_at),
    }


@lru_cache(maxsize=1)
def build_editing_intent_graph():
    graph = StateGraph(EditingIntentGraphState)
    graph.add_node("compile_edit", compile_edit_node)
    graph.add_edge(START, "compile_edit")
    graph.add_edge("compile_edit", END)
    return graph.compile()

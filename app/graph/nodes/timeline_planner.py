import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from app.config import settings
from app.graph.state import SessionGraphState, TimelinePlannerOutput
from app.services.transcript_cache import get_cached_transcript


logger = logging.getLogger(__name__)


def _trace(message: str) -> None:
    print(f"[TRACE][timeline_planner] {message}", flush=True)


def _get_configurable(config: RunnableConfig | None) -> dict[str, Any]:
    if not config:
        return {}
    configurable = config.get("configurable")
    return configurable if isinstance(configurable, dict) else {}


def _get_llm(config: RunnableConfig | None) -> Any:
    llm = _get_configurable(config).get("llm")
    if llm is not None:
        return llm
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=0,
    )


async def _emit_event(
    config: RunnableConfig | None,
    *,
    event_type: str,
    node: str,
    payload: dict[str, Any] | None = None,
) -> None:
    callback = _get_configurable(config).get("event_handler")
    if callback is None:
        return
    await callback({"type": event_type, "node": node, "payload": payload or {}})


async def _timeline_context(state: SessionGraphState) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    cleanup_plan = state.get("cleanup_plan") or {}
    trim_suggestions = cleanup_plan.get("trim_suggestions") or []
    for clip in state.get("clips", []):
        cache_key = clip.get("transcript_cache_key")
        transcript_excerpt = ""
        if cache_key:
            cached = await get_cached_transcript(cache_key)
            if isinstance(cached, dict):
                transcript_excerpt = str(cached.get("full_text") or "")[:1500]
        context.append(
            {
                "clip_id": clip.get("clip_id"),
                "summary": clip.get("summary"),
                "metadata": clip.get("metadata", {}),
                "transcript_excerpt": transcript_excerpt,
                "trim_suggestions": [
                    suggestion
                    for suggestion in trim_suggestions
                    if suggestion.get("clip_id") == clip.get("clip_id")
                ],
            }
        )
    return context


async def timeline_planner_node(
    state: SessionGraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    node_name = "timeline_planner"
    logger.info("[%s] Starting session=%s", node_name, state.get("session_id"))
    _trace(
        f"start session={state.get('session_id')} prior_timeline_entries={len(state.get('timeline', []))}"
    )
    await _emit_event(config, event_type="node_start", node=node_name)

    llm = _get_llm(config).with_structured_output(TimelinePlannerOutput)
    result = await llm.ainvoke(
        [
            (
                "system",
                "You are the editorial timeline planner for an interactive video editing workflow. "
                "Build a coherent sequence using the current edit plan, cleanup guidance, and "
                "available transcript context. Return strict structured output only.",
            ),
            (
                "human",
                "Construct the next proposed timeline.\n"
                f"User prompt: {state.get('user_prompt', '')}\n"
                f"Edit plan: {json.dumps(state.get('edit_plan'))}\n"
                f"Cleanup plan: {json.dumps(state.get('cleanup_plan'))}\n"
                f"Prior timeline: {json.dumps(state.get('timeline'))}\n"
                f"Notes: {json.dumps(state.get('notes', []))}\n"
                f"Clip context: {json.dumps(await _timeline_context(state))}\n"
                "Produce the proposed sequence, choosing clips, in/out points, and rationale.",
            ),
        ]
    )
    _trace(
        "thinking="
        + json.dumps(
            {
                "timeline": [entry.model_dump() for entry in result.timeline],
                "timeline_notes": result.timeline_notes,
            }
        )
    )

    logger.info(
        "[%s] Completed session=%s timeline_entries=%d",
        node_name,
        state.get("session_id"),
        len(result.timeline),
    )
    await _emit_event(
        config,
        event_type="node_complete",
        node=node_name,
        payload={"timeline_entries": len(result.timeline)},
    )
    _trace(f"complete timeline_entries={len(result.timeline)}")
    return {
        "timeline": [entry.model_dump() for entry in result.timeline],
        "timeline_notes": result.timeline_notes,
        "next_action": None,
    }

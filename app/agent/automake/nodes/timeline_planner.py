import json
import logging
from time import perf_counter
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from app.config import settings
from app.agent.automake.state import SessionGraphState, TimelinePlannerOutput
from app.services.transcript_cache import get_cached_transcript


logger = logging.getLogger(__name__)


def _trace(message: str) -> None:
    print(f"[TRACE][timeline_planner] {message}", flush=True)


def _elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


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
    started_at = perf_counter()
    context: list[dict[str, Any]] = []
    cleanup_plan = state.get("cleanup_plan") or {}
    trim_suggestions = cleanup_plan.get("trim_suggestions") or []
    for index, clip in enumerate(state.get("clips", []), start=1):
        cache_key = clip.get("transcript_cache_key")
        transcript_excerpt = ""
        if cache_key:
            cache_started_at = perf_counter()
            logger.info(
                "[timeline_planner] Transcript cache fetch starting session=%s clip_id=%s key=%s",
                state.get("session_id"),
                clip.get("clip_id"),
                cache_key,
            )
            cached = await get_cached_transcript(cache_key)
            logger.info(
                "[timeline_planner] Transcript cache fetch completed session=%s clip_id=%s hit=%s elapsed_ms=%s",
                state.get("session_id"),
                clip.get("clip_id"),
                isinstance(cached, dict),
                _elapsed_ms(cache_started_at),
            )
            if isinstance(cached, dict):
                transcript_excerpt = str(cached.get("full_text") or "")
        context.append(
            {
                "source_order": index,
                "clip_id": clip.get("clip_id"),
                "local_key": clip.get("local_key"),
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
    logger.info(
        "[timeline_planner] Timeline context built session=%s clips=%s elapsed_ms=%s",
        state.get("session_id"),
        len(context),
        _elapsed_ms(started_at),
    )
    return context


async def timeline_planner_node(
    state: SessionGraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    started_at = perf_counter()
    node_name = "timeline_planner"
    logger.info("[%s] Starting session=%s", node_name, state.get("session_id"))
    _trace(
        f"start session={state.get('session_id')} prior_timeline_entries={len(state.get('timeline', []))}"
    )
    await _emit_event(
        config,
        event_type="node_start",
        node=node_name,
        payload={"status_message": "Applying finishing touches."},
    )

    llm = _get_llm(config).with_structured_output(TimelinePlannerOutput)
    context_started_at = perf_counter()
    clip_context = await _timeline_context(state)
    logger.info(
        "[%s] LLM invoke starting session=%s clip_context=%s context_elapsed_ms=%s",
        node_name,
        state.get("session_id"),
        len(clip_context),
        _elapsed_ms(context_started_at),
    )
    llm_started_at = perf_counter()
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
                f"Iteration count: {state.get('iteration_count', 0)}\n"
                f"Edit plan: {json.dumps(state.get('edit_plan'))}\n"
                f"Cleanup plan: {json.dumps(state.get('cleanup_plan'))}\n"
                f"Prior timeline: {json.dumps(state.get('timeline'))}\n"
                f"Notes: {json.dumps(state.get('notes', []))}\n"
                f"Clip context: {json.dumps(clip_context)}\n"
                "If iteration count > 0, determine whether the prompt requests a full overhaul or "
                "a targeted adjustment of the prior timeline. Default to targeted adjustment unless "
                "the user explicitly requests replacing everything. For targeted adjustments, keep "
                "unchanged timeline entries from Prior timeline and modify only the requested part.\n"
                "Each clip in Clip context may include a local_key. When you use a clip in the "
                "timeline, preserve that clip's local_key on the matching timeline entry so clip_id "
                "and local_key stay aligned.\n"
                "Clip context is listed in source_order. Interpret user references like 'first clip', "
                "'second clip', 'last clip', or similar positional language against that source_order. "
                "If the user asks for edits on a specific clip by position, apply those edits to that "
                "same clip and keep that same relative order in the assembled timeline unless the user "
                "explicitly asks you to reorder clips.\n"
                "Produce the proposed sequence, choosing clips, in/out points, and rationale.",
            ),
        ]
    )
    logger.info(
        "[%s] LLM invoke completed session=%s elapsed_ms=%s",
        node_name,
        state.get("session_id"),
        _elapsed_ms(llm_started_at),
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
        "[%s] Completed session=%s timeline_entries=%d elapsed_ms=%s",
        node_name,
        state.get("session_id"),
        len(result.timeline),
        _elapsed_ms(started_at),
    )
    await _emit_event(
        config,
        event_type="node_complete",
        node=node_name,
        payload={
            "timeline_entries": len(result.timeline),
            "timeline": [entry.model_dump() for entry in result.timeline],
            "timeline_notes": result.timeline_notes,
            "status_message": "Timeline updated.",
        },
    )
    _trace(f"complete timeline_entries={len(result.timeline)}")
    return {
        "timeline": [entry.model_dump() for entry in result.timeline],
        "timeline_notes": result.timeline_notes,
        "next_action": None,
        "status_message": "Timeline updated.",
        "status_details": {
            "node": node_name,
            "timeline_entries": len(result.timeline),
        },
    }

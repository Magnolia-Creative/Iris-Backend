import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from app.config import settings
from app.graph.state import ClipCleanupOutput, SessionGraphState
from app.services.transcript_cache import get_cached_transcript


logger = logging.getLogger(__name__)


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


async def _clip_context(state: SessionGraphState) -> list[dict[str, Any]]:
    clip_context: list[dict[str, Any]] = []
    for clip in state.get("clips", []):
        transcript_excerpt = ""
        cache_key = clip.get("transcript_cache_key")
        if cache_key:
            cached = await get_cached_transcript(cache_key)
            if isinstance(cached, dict):
                transcript_excerpt = str(cached.get("full_text") or "")[:1200]

        clip_context.append(
            {
                "clip_id": clip.get("clip_id"),
                "summary": clip.get("summary"),
                "metadata": clip.get("metadata", {}),
                "transcript_excerpt": transcript_excerpt,
            }
        )
    return clip_context


async def clip_cleanup_node(
    state: SessionGraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    node_name = "clip_cleanup"
    logger.info("[%s] Starting session=%s", node_name, state.get("session_id"))
    await _emit_event(config, event_type="node_start", node=node_name)

    llm = _get_llm(config).with_structured_output(ClipCleanupOutput)
    result = await llm.ainvoke(
        [
            (
                "system",
                "You are a semantic clip cleanup node for an interactive video editing system. "
                "Decide which clips are relevant, which to drop, and which trim ranges to keep. "
                "Return strict structured output only.",
            ),
            (
                "human",
                "Use the user's prompt, the current edit plan, and the available clip context.\n"
                f"User prompt: {state.get('user_prompt', '')}\n"
                f"Edit plan: {json.dumps(state.get('edit_plan'))}\n"
                f"Prior cleanup plan: {json.dumps(state.get('cleanup_plan'))}\n"
                f"Clip context: {json.dumps(await _clip_context(state))}\n"
                "Return selected clips, dropped clips, trim suggestions, and concise cleanup notes.",
            ),
        ]
    )

    cleanup_plan = result.model_dump()
    edit_plan = dict(state.get("edit_plan") or {})
    if result.selected_clip_ids:
        edit_plan["target_clips"] = result.selected_clip_ids
    notes = list(state.get("notes", []))
    notes.extend(result.cleanup_notes)

    logger.info(
        "[%s] Completed session=%s selected=%d dropped=%d",
        node_name,
        state.get("session_id"),
        len(result.selected_clip_ids),
        len(result.dropped_clip_ids),
    )
    await _emit_event(
        config,
        event_type="node_complete",
        node=node_name,
        payload={
            "selected_clip_ids": result.selected_clip_ids,
            "dropped_clip_ids": result.dropped_clip_ids,
        },
    )
    return {
        "cleanup_plan": cleanup_plan,
        "edit_plan": edit_plan,
        "notes": notes,
        "next_action": None,
    }

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from app.config import settings
from app.graph.state import DecisionAgentOutput, SessionGraphState


logger = logging.getLogger(__name__)


def _trace(message: str) -> None:
    print(f"[TRACE][decision_agent] {message}", flush=True)


def _get_configurable(config: RunnableConfig | None) -> dict[str, Any]:
    if not config:
        return {}
    configurable = config.get("configurable")
    return configurable if isinstance(configurable, dict) else {}


def _get_llm(config: RunnableConfig | None) -> Any:
    configurable = _get_configurable(config)
    llm = configurable.get("llm")
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
    await callback(
        {
            "type": event_type,
            "node": node,
            "payload": payload or {},
        }
    )


def _clip_digest(state: SessionGraphState) -> list[dict[str, Any]]:
    digest: list[dict[str, Any]] = []
    for clip in state.get("clips", []):
        digest.append(
            {
                "clip_id": clip.get("clip_id"),
                "summary": clip.get("summary"),
                "metadata": clip.get("metadata", {}),
                "transcript_cached": clip.get("transcript_cached", False),
            }
        )
    return digest


async def decision_agent_node(
    state: SessionGraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    node_name = "decision_agent"
    logger.info("[%s] Starting session=%s", node_name, state.get("session_id"))
    _trace(
        f"start session={state.get('session_id')} iteration={state.get('iteration_count', 0)} "
        f"clips={len(state.get('clips', []))}"
    )
    await _emit_event(config, event_type="node_start", node=node_name)

    llm = _get_llm(config).with_structured_output(DecisionAgentOutput)
    force_reconsider = bool(state.get("force_reconsider"))
    messages = [
        (
            "system",
            "You are the orchestration node for an interactive video-editing workflow. "
            "Decide only the next workflow action. You must not directly clean clips, "
            "fetch transcripts, or build the final timeline. Return strict structured output.",
        ),
        (
            "human",
            "Inspect the current workflow state and decide the next action.\n"
            f"User prompt: {state.get('user_prompt', '')}\n"
            f"Iteration count: {state.get('iteration_count', 0)}\n"
            f"Current clips: {json.dumps(_clip_digest(state))}\n"
            f"Existing retrieval plan: {json.dumps(state.get('retrieval_plan'))}\n"
            f"Existing edit plan: {json.dumps(state.get('edit_plan'))}\n"
            f"Existing cleanup plan: {json.dumps(state.get('cleanup_plan'))}\n"
            f"Existing timeline: {json.dumps(state.get('timeline'))}\n"
            f"Timeline notes: {json.dumps(state.get('timeline_notes'))}\n"
            f"Force reconsider: {force_reconsider}\n"
            f"Notes: {json.dumps(state.get('notes', []))}\n"
            f"Errors: {json.dumps(state.get('errors', []))}\n"
            "Use one of these next_action values only: hydrate_transcripts, clip_cleanup, "
            "timeline_planner, finish.\n"
            "When Iteration count > 0, treat Existing timeline/cleanup/edit plans as the current "
            "draft baseline. Determine whether the user's prompt implies: (a) full overhaul of the "
            "draft, or (b) targeted updates to only specific parts.\n"
            "Default to targeted updates unless the user explicitly asks to replace everything. "
            "For targeted updates, preserve unaffected prior sections and choose the next action "
            "that edits only the requested portion.\n"
            "Choose hydrate_transcripts when clip summaries are not enough and transcript "
            "detail is needed for better editorial decisions. Choose clip_cleanup when "
            "semantic filtering or trim suggestions are needed. Choose timeline_planner "
            "when there is enough context to assemble the timeline. Choose finish only "
            "if the workflow should stop without further changes.\n"
            "If Force reconsider is true, you must not choose finish on this turn; "
            "select the best non-finish action to reevaluate the user's latest request.",
        ),
    ]
    result = await llm.ainvoke(messages)
    next_action = result.next_action
    if force_reconsider and next_action == "finish":
        next_action = "timeline_planner"
        result.reasoning_notes.append(
            "Forced reconsider was active, so finish was overridden to timeline_planner."
        )
    notes = list(state.get("notes", []))
    notes.extend(result.reasoning_notes)
    _trace(
        "thinking="
        + json.dumps(
            {
                "next_action": next_action,
                "retrieval_plan": result.retrieval_plan.model_dump(),
                "edit_plan": result.edit_plan.model_dump(),
                "reasoning_notes": result.reasoning_notes,
            }
        )
    )
    logger.info(
        "[%s] Completed session=%s next_action=%s",
        node_name,
        state.get("session_id"),
        next_action,
    )
    await _emit_event(
        config,
        event_type="node_complete",
        node=node_name,
        payload={"next_action": next_action},
    )
    _trace(f"complete next_action={next_action}")
    return {
        "next_action": next_action,
        "retrieval_plan": result.retrieval_plan.model_dump(),
        "edit_plan": result.edit_plan.model_dump(),
        "force_reconsider": False,
        "notes": notes,
    }

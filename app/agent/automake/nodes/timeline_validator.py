import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agent.automake.state import SessionGraphState, TimelineValidationResult, ValidatedTimelineEntry


logger = logging.getLogger(__name__)


def _trace(message: str) -> None:
    print(f"[TRACE][timeline_validator] {message}", flush=True)


def _get_configurable(config: RunnableConfig | None) -> dict[str, Any]:
    if not config:
        return {}
    configurable = config.get("configurable")
    return configurable if isinstance(configurable, dict) else {}


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


def _clip_duration_lookup(state: SessionGraphState) -> dict[str, float | None]:
    lookup: dict[str, float | None] = {}
    for clip in state.get("clips", []):
        metadata = clip.get("metadata") or {}
        duration = metadata.get("duration_seconds")
        lookup[str(clip.get("clip_id"))] = float(duration) if isinstance(duration, (int, float)) else None
    return lookup


def _clip_local_key_lookup(state: SessionGraphState) -> dict[str, str | None]:
    lookup: dict[str, str | None] = {}
    for clip in state.get("clips", []):
        local_key = clip.get("local_key")
        lookup[str(clip.get("clip_id"))] = local_key if isinstance(local_key, str) else None
    return lookup


def _validate_timeline(state: SessionGraphState) -> TimelineValidationResult:
    valid_clip_ids = {str(clip.get("clip_id")) for clip in state.get("clips", [])}
    duration_lookup = _clip_duration_lookup(state)
    local_key_lookup = _clip_local_key_lookup(state)
    errors: list[str] = []
    notes: list[str] = []
    normalized: list[ValidatedTimelineEntry] = []

    for index, entry in enumerate(state.get("timeline", []), start=1):
        clip_id = str(entry.get("clip_id"))
        if clip_id not in valid_clip_ids:
            errors.append(f"Timeline entry {index} references unknown clip_id {clip_id}.")
            continue

        in_sec = entry.get("in_sec")
        out_sec = entry.get("out_sec")
        if not isinstance(in_sec, (int, float)) or not isinstance(out_sec, (int, float)):
            errors.append(f"Timeline entry {index} has non-numeric in/out bounds.")
            continue

        normalized_in = round(float(in_sec), 3)
        normalized_out = round(float(out_sec), 3)
        if normalized_in >= normalized_out:
            errors.append(f"Timeline entry {index} has invalid bounds: in_sec >= out_sec.")
            continue

        duration = duration_lookup.get(clip_id)
        if duration is not None and normalized_out > duration:
            errors.append(
                f"Timeline entry {index} exceeds clip duration for clip_id {clip_id}."
            )
            continue
        if duration is None:
            notes.append(f"No duration metadata found for clip {clip_id}; bounds accepted as-is.")

        entry_local_key = entry.get("local_key")
        normalized_local_key = (
            entry_local_key
            if isinstance(entry_local_key, str)
            else local_key_lookup.get(clip_id)
        )

        normalized.append(
            ValidatedTimelineEntry(
                clip_id=clip_id,
                local_key=normalized_local_key,
                in_sec=normalized_in,
                out_sec=normalized_out,
                rationale=str(entry.get("rationale") or ""),
            )
        )

    return TimelineValidationResult(
        is_valid=bool(normalized) and not errors,
        normalized_timeline=normalized,
        validation_notes=notes,
        validation_errors=errors,
    )


async def timeline_validator_node(
    state: SessionGraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    node_name = "timeline_validator"
    logger.info("[%s] Starting session=%s", node_name, state.get("session_id"))
    _trace(
        f"start session={state.get('session_id')} proposed_entries={len(state.get('timeline', []))}"
    )
    await _emit_event(
        config,
        event_type="node_start",
        node=node_name,
        payload={"status_message": "Checking timeline."},
    )

    result = _validate_timeline(state)
    notes = list(state.get("notes", []))
    notes.extend(result.validation_notes)
    errors = list(state.get("errors", []))
    errors.extend(result.validation_errors)
    _trace(
        f"thinking is_valid={result.is_valid} "
        f"normalized_entries={len(result.normalized_timeline)} "
        f"errors={result.validation_errors}"
    )

    logger.info(
        "[%s] Completed session=%s valid=%s entries=%d",
        node_name,
        state.get("session_id"),
        result.is_valid,
        len(result.normalized_timeline),
    )
    await _emit_event(
        config,
        event_type="node_complete",
        node=node_name,
        payload={
            "is_valid": result.is_valid,
            "normalized_timeline": [entry.model_dump() for entry in result.normalized_timeline],
            "validation_notes": result.validation_notes,
            "validation_errors": result.validation_errors,
            "status_message": (
                "Timeline ready for review."
                if result.is_valid
                else "Refining timeline."
            ),
        },
    )
    _trace(f"complete waiting_for_user={result.is_valid}")
    return {
        "timeline": [entry.model_dump() for entry in result.normalized_timeline],
        "waiting_for_user": result.is_valid,
        "notes": notes,
        "errors": errors,
        "next_action": "finish" if result.is_valid else "timeline_planner",
        "status_message": (
            "Timeline ready for review."
            if result.is_valid
            else "Refining timeline."
        ),
        "status_details": {
            "node": node_name,
            "is_valid": result.is_valid,
            "validation_errors": result.validation_errors,
        },
    }

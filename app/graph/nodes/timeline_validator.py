import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.graph.state import SessionGraphState, TimelineValidationResult, ValidatedTimelineEntry


logger = logging.getLogger(__name__)


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


def _validate_timeline(state: SessionGraphState) -> TimelineValidationResult:
    valid_clip_ids = {str(clip.get("clip_id")) for clip in state.get("clips", [])}
    duration_lookup = _clip_duration_lookup(state)
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

        normalized.append(
            ValidatedTimelineEntry(
                clip_id=clip_id,
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
    await _emit_event(config, event_type="node_start", node=node_name)

    result = _validate_timeline(state)
    notes = list(state.get("notes", []))
    notes.extend(result.validation_notes)
    errors = list(state.get("errors", []))
    errors.extend(result.validation_errors)

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
            "validation_errors": result.validation_errors,
        },
    )
    return {
        "timeline": [entry.model_dump() for entry in result.normalized_timeline],
        "waiting_for_user": result.is_valid,
        "notes": notes,
        "errors": errors,
        "next_action": "finish" if result.is_valid else "timeline_planner",
    }

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.state import SessionGraphState
from app.services.transcript_cache import cache_transcript, is_transcript_cached
from app.services.transcript_store import get_transcript_payload


logger = logging.getLogger(__name__)


def _trace(message: str) -> None:
    print(f"[TRACE][hydrate_transcripts] {message}", flush=True)


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


def _get_db(config: RunnableConfig | None) -> AsyncSession:
    db = _get_configurable(config).get("db")
    if not isinstance(db, AsyncSession):
        raise RuntimeError("Async DB session is required for transcript hydration")
    return db


async def hydrate_transcripts_node(
    state: SessionGraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    node_name = "hydrate_transcripts"
    logger.info("[%s] Starting session=%s", node_name, state.get("session_id"))
    _trace(f"start session={state.get('session_id')}")
    await _emit_event(
        config,
        event_type="node_start",
        node=node_name,
        payload={"status_message": "Analyzing audio."},
    )

    db = _get_db(config)
    session_id = state["session_id"]
    requested_clip_ids = set((state.get("retrieval_plan") or {}).get("clip_ids") or [])
    _trace(f"requested_clip_ids={sorted(requested_clip_ids)}")
    updated_clips: list[dict[str, Any]] = []
    hydrated_clip_ids: list[str] = []
    notes = list(state.get("notes", []))

    for clip in state.get("clips", []):
        clip_id = clip.get("clip_id")
        if clip_id not in requested_clip_ids:
            updated_clips.append(dict(clip))
            continue

        clip_copy = dict(clip)
        cache_key = clip_copy.get("transcript_cache_key")
        if cache_key and await is_transcript_cached(cache_key):
            clip_copy["transcript_cached"] = True
            updated_clips.append(clip_copy)
            continue

        transcript_id = clip_copy.get("transcript_id")
        if not isinstance(transcript_id, int):
            updated_clips.append(clip_copy)
            continue

        transcript_payload = await get_transcript_payload(db, transcript_id)
        if transcript_payload is None:
            updated_clips.append(clip_copy)
            continue

        cache_key = await cache_transcript(session_id, clip_id, transcript_payload)
        clip_copy["transcript_cached"] = True
        clip_copy["transcript_cache_key"] = cache_key
        updated_clips.append(clip_copy)
        hydrated_clip_ids.append(clip_id)

    if hydrated_clip_ids:
        notes.append(f"Hydrated transcripts for clips: {', '.join(hydrated_clip_ids)}")
    _trace(f"hydrated_clip_ids={hydrated_clip_ids}")

    logger.info(
        "[%s] Completed session=%s hydrated=%d",
        node_name,
        session_id,
        len(hydrated_clip_ids),
    )
    await _emit_event(
        config,
        event_type="node_complete",
        node=node_name,
        payload={
            "requested_clip_ids": sorted(requested_clip_ids),
            "hydrated_clip_ids": hydrated_clip_ids,
            "status_message": "Audio analysis complete.",
        },
    )
    _trace("complete")
    return {
        "clips": updated_clips,
        "notes": notes,
        "next_action": None,
        "status_message": "Audio analysis complete.",
        "status_details": {
            "node": node_name,
            "hydrated_clip_ids": hydrated_clip_ids,
            "requested_clip_ids": sorted(requested_clip_ids),
        },
    }

import logging
from time import perf_counter
from typing import Any

from langchain_core.runnables import RunnableConfig
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.automake.state import SessionGraphState
from app.services.transcript_cache import cache_transcript, is_transcript_cached
from app.services.transcript_store import get_transcript_payload


logger = logging.getLogger(__name__)


def _trace(message: str) -> None:
    print(f"[TRACE][hydrate_transcripts] {message}", flush=True)


def _elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


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
    started_at = perf_counter()
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
    logger.info(
        "[%s] Requested clips session=%s requested_clip_ids=%s total_clips=%s",
        node_name,
        session_id,
        sorted(requested_clip_ids),
        len(state.get("clips", [])),
    )
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
        if cache_key:
            cache_started_at = perf_counter()
            logger.info(
                "[%s] Cache check starting session=%s clip_id=%s key=%s",
                node_name,
                session_id,
                clip_id,
                cache_key,
            )
            cached = await is_transcript_cached(cache_key)
            logger.info(
                "[%s] Cache check completed session=%s clip_id=%s cached=%s elapsed_ms=%s",
                node_name,
                session_id,
                clip_id,
                cached,
                _elapsed_ms(cache_started_at),
            )
            if cached:
                clip_copy["transcript_cached"] = True
                updated_clips.append(clip_copy)
                continue

        transcript_id = clip_copy.get("transcript_id")
        if not isinstance(transcript_id, int):
            logger.warning(
                "[%s] Missing numeric transcript_id session=%s clip_id=%s transcript_id=%r",
                node_name,
                session_id,
                clip_id,
                transcript_id,
            )
            updated_clips.append(clip_copy)
            continue

        fetch_started_at = perf_counter()
        logger.info(
            "[%s] DB transcript fetch starting session=%s clip_id=%s transcript_id=%s",
            node_name,
            session_id,
            clip_id,
            transcript_id,
        )
        transcript_payload = await get_transcript_payload(db, transcript_id)
        logger.info(
            "[%s] DB transcript fetch completed session=%s clip_id=%s transcript_id=%s found=%s elapsed_ms=%s",
            node_name,
            session_id,
            clip_id,
            transcript_id,
            transcript_payload is not None,
            _elapsed_ms(fetch_started_at),
        )
        if transcript_payload is None:
            updated_clips.append(clip_copy)
            continue

        cache_started_at = perf_counter()
        logger.info(
            "[%s] Cache write starting session=%s clip_id=%s transcript_id=%s",
            node_name,
            session_id,
            clip_id,
            transcript_id,
        )
        cache_key = await cache_transcript(session_id, clip_id, transcript_payload)
        logger.info(
            "[%s] Cache write completed session=%s clip_id=%s key=%s elapsed_ms=%s",
            node_name,
            session_id,
            clip_id,
            cache_key,
            _elapsed_ms(cache_started_at),
        )
        clip_copy["transcript_cached"] = True
        clip_copy["transcript_cache_key"] = cache_key
        updated_clips.append(clip_copy)
        hydrated_clip_ids.append(clip_id)

    if hydrated_clip_ids:
        notes.append(f"Hydrated transcripts for clips: {', '.join(hydrated_clip_ids)}")
    _trace(f"hydrated_clip_ids={hydrated_clip_ids}")

    logger.info(
        "[%s] Completed session=%s hydrated=%d elapsed_ms=%s",
        node_name,
        session_id,
        len(hydrated_clip_ids),
        _elapsed_ms(started_at),
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

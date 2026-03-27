import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from app.config import settings
from app.graph.state import ClipCleanupOutput, SessionGraphState
from app.services.transcript_cache import get_cached_transcript


logger = logging.getLogger(__name__)
_PROMPT_ENTITY_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "has",
    "have",
    "include",
    "interview",
    "only",
    "podcast",
    "specifically",
    "that",
    "the",
    "this",
    "today",
    "want",
    "with",
    "you",
}


def _trace(message: str) -> None:
    print(f"[TRACE][clip_cleanup] {message}", flush=True)


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


def _prompt_entity_terms(prompt: str) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z'-]{1,}", prompt):
        normalized = token.strip("'").lower()
        if len(normalized) < 3 or normalized in _PROMPT_ENTITY_STOPWORDS:
            continue
        if normalized not in terms:
            terms.append(normalized)
    return terms


def _extract_prompt_entity_hits(
    transcript_payload: dict[str, Any], entity_terms: list[str]
) -> list[dict[str, Any]]:
    if not entity_terms:
        return []
    segments = transcript_payload.get("segments")
    if not isinstance(segments, list):
        return []

    hits: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, float, float, str]] = set()
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_text = str(segment.get("text") or "").strip()
        if not segment_text:
            continue
        start = segment.get("start")
        end = segment.get("end")
        start_sec = float(start) if isinstance(start, (int, float)) else None
        end_sec = float(end) if isinstance(end, (int, float)) else None

        for term in entity_terms:
            if not re.search(rf"\b{re.escape(term)}\b", segment_text, flags=re.IGNORECASE):
                continue
            key = (term, start_sec or -1.0, end_sec or -1.0, segment_text)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            hits.append(
                {
                    "term": term,
                    "start": start_sec,
                    "end": end_sec,
                    "text": segment_text,
                }
            )
    return hits


def _transcript_excerpt_with_hits(
    transcript_payload: dict[str, Any], *, entity_hits: list[dict[str, Any]]
) -> str:
    base_excerpt = str(transcript_payload.get("full_text") or "")
    if not entity_hits:
        return base_excerpt

    hit_lines: list[str] = []
    for hit in entity_hits[:12]:
        start = f"{hit['start']:.3f}" if isinstance(hit.get("start"), float) else "?"
        end = f"{hit['end']:.3f}" if isinstance(hit.get("end"), float) else "?"
        term = str(hit.get("term") or "")
        text = str(hit.get("text") or "")
        hit_lines.append(f"- {term} [{start}-{end}]: {text}")
    hit_block = "Prompt entity transcript hits:\n" + "\n".join(hit_lines)
    return f"{base_excerpt}\n\n{hit_block}".strip()


async def _clip_context(state: SessionGraphState) -> list[dict[str, Any]]:
    clip_context: list[dict[str, Any]] = []
    prompt_entities = _prompt_entity_terms(state.get("user_prompt", ""))
    for clip in state.get("clips", []):
        transcript_excerpt = ""
        prompt_entity_hits: list[dict[str, Any]] = []
        cache_key = clip.get("transcript_cache_key")
        if cache_key:
            cached = await get_cached_transcript(cache_key)
            if isinstance(cached, dict):
                prompt_entity_hits = _extract_prompt_entity_hits(cached, prompt_entities)
                transcript_excerpt = _transcript_excerpt_with_hits(
                    cached, entity_hits=prompt_entity_hits
                )

        clip_context.append(
            {
                "clip_id": clip.get("clip_id"),
                "summary": clip.get("summary"),
                "metadata": clip.get("metadata", {}),
                "transcript_excerpt": transcript_excerpt,
                "prompt_entity_hits": prompt_entity_hits[:20],
            }
        )
    return clip_context


async def clip_cleanup_node(
    state: SessionGraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    node_name = "clip_cleanup"
    logger.info("[%s] Starting session=%s", node_name, state.get("session_id"))
    _trace(
        f"start session={state.get('session_id')} prompt={state.get('user_prompt', '')!r}"
    )
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
                f"Iteration count: {state.get('iteration_count', 0)}\n"
                f"Edit plan: {json.dumps(state.get('edit_plan'))}\n"
                f"Prior cleanup plan: {json.dumps(state.get('cleanup_plan'))}\n"
                f"Prior timeline: {json.dumps(state.get('timeline'))}\n"
                f"Clip context: {json.dumps(await _clip_context(state))}\n"
                "Treat prompt_entity_hits as high-signal evidence for speaker/name matches when present.\n"
                "If iteration count > 0, treat prior plans/timeline as the baseline draft and infer "
                "whether the user requested a full replacement or a partial update. Default to partial "
                "update unless full replacement is explicit. For partial updates, preserve unaffected "
                "clips/ranges and only change selections/trims for the specifically requested portion.\n"
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
    _trace(
        "thinking="
        + json.dumps(
            {
                "selected_clip_ids": result.selected_clip_ids,
                "dropped_clip_ids": result.dropped_clip_ids,
                "trim_suggestions": [s.model_dump() for s in result.trim_suggestions],
                "cleanup_notes": result.cleanup_notes,
            }
        )
    )

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
    _trace(
        f"complete selected={len(result.selected_clip_ids)} dropped={len(result.dropped_clip_ids)}"
    )
    return {
        "cleanup_plan": cleanup_plan,
        "edit_plan": edit_plan,
        "notes": notes,
        "next_action": None,
    }

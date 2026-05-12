from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.intent_compiler.models import (
    ClipTranscriptContext,
    IntentCompilerContext,
    TranscriptPauseRange,
    TranscriptWord,
)
from app.intent_compiler.transcript_phrases import collect_phrase_matches
from app.intent_compiler.transcript_preflight import (
    needs_phrase_matching,
    needs_transcript_hydration,
    resolve_transcript_target_clip_ids,
)
from app.services.transcript_cache import cache_transcript, get_cached_transcript
from app.services.transcript_store import get_transcript_payload


async def hydrate_intent_transcript_context(
    context: IntentCompilerContext,
    db: AsyncSession,
) -> IntentCompilerContext:
    if not context.transcriptContextsByClipId:
        return context

    hydrated: dict[str, ClipTranscriptContext] = {}
    for clip_id, transcript_ref in context.transcriptContextsByClipId.items():
        payload: dict[str, Any] | None = None
        cache_key = transcript_ref.cacheKey
        if cache_key:
            payload = await get_cached_transcript(cache_key)

        transcript_id = _int_transcript_id(transcript_ref.transcriptId)
        if payload is None and transcript_id is not None:
            payload = await get_transcript_payload(db, transcript_id)
            if payload is not None:
                cache_key = await cache_transcript(_cache_session_id(context), clip_id, payload)

        hydrated[clip_id] = (
            _context_from_payload(
                clip_id=clip_id,
                transcript_id=transcript_ref.transcriptId,
                cache_key=cache_key,
                payload=payload,
            )
            if payload is not None
            else transcript_ref
        )

    return context.model_copy(update={"transcriptContextsByClipId": hydrated})


async def prepare_intent_transcript_context(
    *,
    prompt: str,
    context: IntentCompilerContext,
    db: AsyncSession,
) -> tuple[IntentCompilerContext, dict[str, Any]]:
    """Decide whether transcript data is needed, hydrate only target clips, attach phrase matches."""
    meta: dict[str, Any] = {
        "transcript_preflight": "skipped",
        "transcript_refs_in_request": len(context.transcriptContextsByClipId),
        "transcript_target_clip_count": 0,
        "hydrated_word_total": 0,
        "pause_range_total": 0,
        "phrase_match_total": 0,
        "skip_reason": None,
    }

    if not needs_transcript_hydration(prompt):
        meta["transcript_preflight"] = "skipped_no_signal"
        meta["skip_reason"] = "no_transcript_intent_signal"
        return context, meta

    if not context.transcriptContextsByClipId:
        meta["transcript_preflight"] = "skipped_empty_refs"
        meta["skip_reason"] = "transcript_intent_but_no_clip_refs"
        return context, meta

    target_ids = resolve_transcript_target_clip_ids(context, prompt)
    filtered = {
        clip_id: ref for clip_id, ref in context.transcriptContextsByClipId.items() if clip_id in target_ids
    }
    if not filtered:
        meta["transcript_preflight"] = "skipped_no_target_refs"
        meta["skip_reason"] = "no_transcript_ref_for_resolved_targets"
        return context, meta

    slim = context.model_copy(update={"transcriptContextsByClipId": filtered})
    hydrated = await hydrate_intent_transcript_context(slim, db)

    if needs_phrase_matching(prompt):
        hydrated = _attach_phrase_matches(prompt, hydrated)

    meta["transcript_preflight"] = "hydrated"
    meta["transcript_target_clip_count"] = len(filtered)
    for tctx in hydrated.transcriptContextsByClipId.values():
        meta["hydrated_word_total"] += len(tctx.words)
        meta["pause_range_total"] += len(tctx.pauseRanges)
        meta["phrase_match_total"] += len(tctx.phraseMatches)

    return hydrated, meta


def _attach_phrase_matches(prompt: str, context: IntentCompilerContext) -> IntentCompilerContext:
    updated: dict[str, ClipTranscriptContext] = {}
    for clip_id, tctx in context.transcriptContextsByClipId.items():
        if not tctx.words:
            updated[clip_id] = tctx
            continue
        matches = collect_phrase_matches(prompt, tctx.words)
        updated[clip_id] = tctx.model_copy(update={"phraseMatches": matches})
    return context.model_copy(update={"transcriptContextsByClipId": updated})


def _context_from_payload(
    *,
    clip_id: str,
    transcript_id: int | str | None,
    cache_key: str | None,
    payload: dict[str, Any],
) -> ClipTranscriptContext:
    words = _timed_words(payload)
    return ClipTranscriptContext(
        clipId=clip_id,
        transcriptId=transcript_id,
        cacheKey=cache_key,
        fullText=str(payload.get("full_text") or ""),
        words=words,
        pauseRanges=_pause_ranges_from_words(words),
    )


def _timed_words(transcript_payload: dict[str, Any]) -> list[TranscriptWord]:
    segments = transcript_payload.get("segments")
    if not isinstance(segments, list):
        return []

    words: list[TranscriptWord] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        raw_words = segment.get("words")
        if not isinstance(raw_words, list):
            continue
        for raw_word in raw_words:
            if not isinstance(raw_word, dict):
                continue
            word = str(raw_word.get("word") or "").strip()
            start = _seconds_to_us(raw_word.get("start"))
            end = _seconds_to_us(raw_word.get("end"))
            if not word or start is None or end is None or end <= start:
                continue
            words.append(TranscriptWord(word=word, startUs=start, endUs=end))
    return words


def _pause_ranges_from_words(
    words: list[TranscriptWord],
    *,
    min_gap_us: int = 600_000,
    max_items: int = 20,
) -> list[TranscriptPauseRange]:
    pauses: list[TranscriptPauseRange] = []
    for previous_word, current_word in zip(words, words[1:], strict=False):
        gap = current_word.startUs - previous_word.endUs
        if gap < min_gap_us:
            continue
        pauses.append(
            TranscriptPauseRange(
                startUs=previous_word.endUs,
                endUs=current_word.startUs,
                durationUs=gap,
                beforeWord=previous_word.word,
                afterWord=current_word.word,
            )
        )
        if len(pauses) >= max_items:
            break
    return pauses


def _seconds_to_us(value: Any) -> int | None:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value) * 1_000_000)


def _int_transcript_id(value: int | str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _cache_session_id(context: IntentCompilerContext) -> str:
    return str(context.sessionId or context.projectId or context.timelineId)

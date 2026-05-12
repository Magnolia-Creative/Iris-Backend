"""Deterministic rules for when intent compilation should hydrate transcript data."""

from __future__ import annotations

import re

from app.intent_compiler.models import IntentCompilerContext

_DEAD_SPACE = re.compile(
    r"\b(dead\s*space|silence|silent|pause|pauses|gap|gaps)\b",
    re.IGNORECASE,
)
_FILLER = re.compile(
    r"\b(ums?|uhs?|filler|disfluency|stutter|false\s+start|mistake)\b",
    re.IGNORECASE,
)
_WHERE_SAY = re.compile(r"\b(where|when)\s+i\s+say\b", re.IGNORECASE)
_REMOVE_PART = re.compile(r"\b(remove|cut)\s+(out\s+)?(the\s+)?(part\s+)?where\b", re.IGNORECASE)
_QUOTED = re.compile(r'["\']')


def needs_transcript_hydration(prompt: str) -> bool:
    text = (prompt or "").strip()
    if not text:
        return False
    if _DEAD_SPACE.search(text):
        return True
    if _FILLER.search(text):
        return True
    if _WHERE_SAY.search(text) or _REMOVE_PART.search(text):
        return True
    if _QUOTED.search(text):
        return True
    return False


def needs_phrase_matching(prompt: str) -> bool:
    text = (prompt or "").strip()
    if not text:
        return False
    if _WHERE_SAY.search(text) or _REMOVE_PART.search(text):
        return True
    if '"' in text or "'" in text:
        return True
    return False


def current_clip_at_playhead_id(context: IntentCompilerContext) -> str | None:
    if context.playheadTimeUs is None:
        return None
    track_ids: list[str] = []
    if context.selectedTrackId:
        track_ids.append(context.selectedTrackId)
    track_ids.extend(tid for tid in sorted(context.orderedClipIdsByTrackId) if tid not in track_ids)
    for track_id in track_ids:
        for clip_id in context.orderedClipIdsByTrackId.get(track_id, []):
            clip = context.clip(clip_id)
            if clip and clip.timelineRange.start <= context.playheadTimeUs < clip.timelineRange.end:
                return clip.clipId
    return None


def resolve_transcript_target_clip_ids(context: IntentCompilerContext, prompt: str) -> set[str]:
    """Clip IDs that should receive hydrated transcript context for this prompt."""
    refs = context.transcriptContextsByClipId
    if not refs:
        return set()

    lowered = prompt.lower()
    broad_markers = (
        "all clips",
        "every clip",
        "each clip",
        "whole timeline",
        "entire timeline",
        "all the clips",
    )
    if any(marker in lowered for marker in broad_markers):
        return set(refs.keys())

    targets: set[str] = set()
    sel = context.selectedClipId
    if sel and sel in refs:
        targets.add(sel)
    cur = current_clip_at_playhead_id(context)
    if cur and cur in refs:
        targets.add(cur)
    if targets:
        return targets
    return set(refs.keys())

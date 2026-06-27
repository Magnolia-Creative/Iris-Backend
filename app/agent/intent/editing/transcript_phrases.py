"""Phrase extraction and deterministic transcript word-window matching."""

from __future__ import annotations

import re

from app.agent.intent.editing.models import TranscriptPhraseMatch, TranscriptWord

_PHRASE_PAD_US = 120_000

_QUOTED_DOUBLE = re.compile(r'"([^"]{2,200})"')
_QUOTED_SINGLE = re.compile(r"'([^']{2,200})'")
_TRAILING_PUNCT = re.compile(r"[.,!?;:]+$")


def _norm_token(word: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (word or "").lower())


def extract_phrase_queries(prompt: str) -> list[str]:
    text = (prompt or "").strip()
    if not text:
        return []

    candidates: list[str] = []
    for pattern in (_QUOTED_DOUBLE, _QUOTED_SINGLE):
        for match in pattern.finditer(text):
            chunk = match.group(1).strip()
            if len(chunk) >= 2:
                candidates.append(_TRAILING_PUNCT.sub("", chunk))

    tail_clause = re.compile(
        r"\b(?:where|when)\s+i\s+say\s+(.{2,200}?)(?:[.?!]|$)",
        re.IGNORECASE | re.DOTALL,
    )
    for match in tail_clause.finditer(text):
        chunk = match.group(1).strip()
        chunk = _TRAILING_PUNCT.sub("", chunk)
        if len(chunk) >= 2:
            candidates.append(chunk)

    remove_where = re.compile(
        r"\b(?:remove|cut)\s+(?:out\s+)?(?:the\s+)?(?:part\s+)?where\s+(?:i\s+say\s+)?(.{2,200}?)(?:[.?!]|$)",
        re.IGNORECASE | re.DOTALL,
    )
    for match in remove_where.finditer(text):
        chunk = match.group(1).strip()
        chunk = _TRAILING_PUNCT.sub("", chunk)
        if len(chunk) >= 2:
            candidates.append(chunk)

    seen: set[str] = set()
    ordered: list[str] = []
    for phrase in candidates:
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(phrase)
    return ordered


def find_phrase_time_ranges(words: list[TranscriptWord], phrase: str) -> list[TranscriptPhraseMatch]:
    query_tokens = [_norm_token(part) for part in phrase.split() if _norm_token(part)]
    if not query_tokens or not words:
        return []

    word_tokens = [_norm_token(w.word) for w in words]
    n_tokens, window = len(word_tokens), len(query_tokens)
    if window == 0 or n_tokens < window:
        return []

    matches: list[TranscriptPhraseMatch] = []
    for start_idx in range(0, n_tokens - window + 1):
        if word_tokens[start_idx : start_idx + window] != query_tokens:
            continue
        span = words[start_idx : start_idx + window]
        start_us = max(0, span[0].startUs - _PHRASE_PAD_US)
        end_us = span[-1].endUs + _PHRASE_PAD_US
        matches.append(TranscriptPhraseMatch(phrase=phrase, startUs=start_us, endUs=end_us))
    return matches


def collect_phrase_matches(prompt: str, words: list[TranscriptWord]) -> list[TranscriptPhraseMatch]:
    all_matches: list[TranscriptPhraseMatch] = []
    for phrase in extract_phrase_queries(prompt):
        all_matches.extend(find_phrase_time_ranges(words, phrase))
    return _dedupe_phrase_matches(all_matches)


def _dedupe_phrase_matches(matches: list[TranscriptPhraseMatch]) -> list[TranscriptPhraseMatch]:
    if not matches:
        return []
    ordered = sorted(matches, key=lambda item: (item.startUs, item.endUs))
    merged: list[TranscriptPhraseMatch] = []
    for match in ordered:
        if not merged:
            merged.append(match)
            continue
        last = merged[-1]
        if match.startUs <= last.endUs:
            if match.endUs > last.endUs:
                merged[-1] = TranscriptPhraseMatch(
                    phrase=last.phrase,
                    startUs=last.startUs,
                    endUs=max(last.endUs, match.endUs),
                )
        else:
            merged.append(match)
    return merged


def transcript_operation_wants_phrase_fallback(text: str | None) -> bool:
    return bool(extract_phrase_queries(text or "")) or bool(
        re.search(r"\b(where|when)\s+i\s+say\b", text or "", re.IGNORECASE)
    )

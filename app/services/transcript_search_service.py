"""Project-scoped literal search over sentence transcript segments (JSONB `segments` only)."""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import models
from app.domains.search import SearchMatch, search_response_dict
from app.services.semantic_constants import RESULTS_LIMIT

logger = logging.getLogger(__name__)


def _normalize_words(text: str) -> str:
    return " ".join(
        w
        for w in re.sub(r"[^a-zA-Z0-9]+", " ", text.lower()).split()
        if w
    )


def _segment_time_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _segment_bounds(seg: dict[str, Any]) -> tuple[float, float] | None:
    start = _segment_time_seconds(seg.get("start"))
    end = _segment_time_seconds(seg.get("end"))
    if start is None or end is None or end < start:
        return None
    return (start, end)


def _confidence_for_match(*, query_norm: str, query_terms: set[str], segment_text: str) -> float:
    """Higher = better. Exact phrase (substring) beats token-only overlap."""
    seg_norm = _normalize_words(segment_text)
    if not seg_norm or not query_norm:
        return 0.0

    if query_norm in seg_norm:
        return 100.0 + min(len(query_norm), 80) * 0.01

    seg_terms = set(seg_norm.split())
    shared = query_terms & seg_terms
    if not shared:
        return 0.0
    return 10.0 + (len(shared) / max(len(query_terms), 1)) * 5.0


async def search_project_transcript(
    db: AsyncSession,
    *,
    project_id: int,
    query: str,
    limit: int | None = None,
) -> dict[str, Any]:
    lim = limit if limit is not None and limit > 0 else RESULTS_LIMIT
    qstrip = (query or "").strip()
    if not qstrip:
        logger.info("[transcript_search] empty query project_id=%s", project_id)
        return search_response_dict(matches=[], query=query)

    query_norm = _normalize_words(qstrip)
    query_terms = set(query_norm.split()) if query_norm else set()
    if not query_terms:
        return search_response_dict(matches=[], query=query)

    stmt = (
        select(
            models.Clip.id,
            models.Clip.local_key,
            models.Clip.file_name,
            models.Transcript.transcript["segments"].label("segments"),
        )
        .join(models.Transcript, models.Transcript.clip_id == models.Clip.id)
        .where(models.Clip.project_id == project_id)
        .order_by(models.Clip.id)
    )
    result = await db.execute(stmt)
    rows = result.all()

    raw_hits: list[tuple[float, int, float, SearchMatch]] = []
    for clip_id, local_key, file_name, segments in rows:
        if not isinstance(segments, list):
            continue
        lk = str(local_key) if local_key is not None else ""
        fn = str(file_name) if file_name is not None else None
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            text = seg.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            bounds = _segment_bounds(seg)
            if bounds is None:
                continue
            start_s, end_s = bounds
            conf = _confidence_for_match(
                query_norm=query_norm, query_terms=query_terms, segment_text=text
            )
            if conf <= 0:
                continue
            raw_hits.append(
                (
                    -conf,
                    int(clip_id),
                    start_s,
                    SearchMatch(
                        clip_id=int(clip_id),
                        local_key=lk,
                        file_name=fn,
                        start_time_seconds=start_s,
                        end_time_seconds=end_s,
                        confidence=conf,
                        source="audio",
                        match_text=text.strip(),
                    ),
                )
            )

    raw_hits.sort(key=lambda t: (t[0], t[1], t[2]))
    matches: list[SearchMatch] = [t[3] for t in raw_hits[:lim]]

    logger.info(
        "[transcript_search] project_id=%s query_len=%s limit=%s hits=%s returned=%s",
        project_id,
        len(qstrip),
        lim,
        len(raw_hits),
        len(matches),
    )
    return search_response_dict(matches=matches, query=query)

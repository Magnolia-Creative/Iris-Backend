"""Merge adjacent chunk hits (ported from Iris-Main TemporalRangeScorer)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.services.semantic_constants import CHUNK_MERGE_MINIMUM_SCORE, RANGE_MERGE_GAP_SECONDS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkHit:
    clip_id: int
    local_key: str
    file_name: str | None
    modality: str
    chunk_index: int
    start_time_seconds: float
    end_time_seconds: float
    center_time_seconds: float
    score: float


@dataclass(frozen=True)
class MergedRange:
    clip_id: int
    local_key: str
    file_name: str | None
    start_time_seconds: float
    end_time_seconds: float
    confidence: float
    source: str


def _build_merged_group(
    *,
    clip_id: int,
    local_key: str,
    file_name: str | None,
    modality: str,
    start_time_seconds: float,
    end_time_seconds: float,
    chunk_hits: list[ChunkHit],
) -> MergedRange:
    score_sum = sum(h.score for h in chunk_hits)
    score_count = len(chunk_hits)
    peak = max(h.score for h in chunk_hits)
    average = score_sum / max(score_count, 1)
    confidence = (average * 0.65) + (peak * 0.35)
    # Confidence blends mean similarity with peak chunk match so a single
    # strong hit still ranks well alongside broader multi-chunk agreement.
    return MergedRange(
        clip_id=clip_id,
        local_key=local_key,
        file_name=file_name,
        start_time_seconds=start_time_seconds,
        end_time_seconds=end_time_seconds,
        confidence=confidence,
        source=modality,
    )


def _log_merged_range(
    mr: MergedRange,
    chunk_hits: list[ChunkHit],
    *,
    project_id: int | None,
    range_closure: str,
) -> None:
    """Explain one merged timeline span: which chunks fused, why, and how confidence is derived."""
    score_sum = sum(h.score for h in chunk_hits)
    n = len(chunk_hits)
    avg = score_sum / max(n, 1)
    peak = max(h.score for h in chunk_hits)
    ctx = f"project_id={project_id} " if project_id is not None else ""
    chunk_bits = "; ".join(
        f"idx={h.chunk_index} t={h.start_time_seconds:.2f}-{h.end_time_seconds:.2f}s sim={h.score:.3f}"
        for h in chunk_hits
    )
    logger.info(
        "[chunk_range_merge] %sclip_id=%s local_key=%s modality=%s closure=%s | "
        "merged_wall_time=%.2f-%.2f s n_chunks=%s confidence=%.4f avg_sim=%.4f peak_sim=%.4f "
        "(confidence = 0.65 * avg_sim + 0.35 * peak_sim)",
        ctx,
        mr.clip_id,
        mr.local_key,
        mr.source,
        range_closure,
        mr.start_time_seconds,
        mr.end_time_seconds,
        n,
        mr.confidence,
        avg,
        peak,
    )
    logger.info("[chunk_range_merge] %schunks_detail clip_id=%s | %s", ctx, mr.clip_id, chunk_bits)


def merge_chunk_hits(
    hits: list[ChunkHit],
    *,
    max_gap_seconds: float = RANGE_MERGE_GAP_SECONDS,
    minimum_score: float = CHUNK_MERGE_MINIMUM_SCORE,
    project_id: int | None = None,
) -> list[MergedRange]:
    ctx = f"project_id={project_id} " if project_id is not None else ""
    logger.info(
        "[chunk_range_merge] %sstart raw_hits=%s thresholds: minimum_score=%.4f (drop at <=) "
        "max_gap_seconds=%.4f - within each clip, walk chunks in time order and extend one range "
        "while (next.start - range_end) <= max_gap; else start a new range. "
        "Embedding model supplies per-chunk similarity scores; merge only uses time + those scores.",
        ctx,
        len(hits),
        minimum_score,
        max_gap_seconds,
    )

    eligible = [h for h in hits if h.score > minimum_score]
    dropped = len(hits) - len(eligible)
    if dropped:
        logger.info(
            "[chunk_range_merge] %sscore_filter dropped=%s kept=%s "
            "(only chunks with similarity score strictly above minimum_score are merged)",
            ctx,
            dropped,
            len(eligible),
        )
    if not eligible:
        logger.info(
            "[chunk_range_merge] %sno eligible chunks after score filter; returning no merged ranges",
            ctx,
        )
        return []

    grouped: dict[int, list[ChunkHit]] = {}
    for h in eligible:
        grouped.setdefault(h.clip_id, []).append(h)

    logger.info(
        "[chunk_range_merge] %sgroup_by_clip clips_with_hits=%s (each clip processed on its own timeline)",
        ctx,
        len(grouped),
    )

    merged: list[MergedRange] = []
    for clip_id, clip_hits in grouped.items():
        sorted_hits = sorted(clip_hits, key=lambda h: h.start_time_seconds)
        if not sorted_hits:
            continue

        logger.info(
            "[chunk_range_merge] %sclip_id=%s timeline_walk eligible_chunks=%s "
            "(sorted by start_time; embedding model scores each chunk vs query — merge only uses time gaps + scores)",
            ctx,
            clip_id,
            len(sorted_hits),
        )

        first = sorted_hits[0]
        range_start = first.start_time_seconds
        range_end = first.end_time_seconds
        local_key = first.local_key
        file_name = first.file_name
        modality = first.modality
        chunk_hits = [first]

        def close_range(closure: str) -> None:
            mr = _build_merged_group(
                clip_id=clip_id,
                local_key=local_key,
                file_name=file_name,
                modality=modality,
                start_time_seconds=range_start,
                end_time_seconds=range_end,
                chunk_hits=chunk_hits,
            )
            merged.append(mr)
            _log_merged_range(
                mr,
                chunk_hits,
                project_id=project_id,
                range_closure=closure,
            )

        for current in sorted_hits[1:]:
            gap = current.start_time_seconds - range_end
            if gap <= max_gap_seconds:
                logger.debug(
                    "[chunk_range_merge] %sclip_id=%s extend_range: gap=%.4fs <= max_gap=%.4fs "
                    "append chunk_idx=%s (ranges grow along the clip when neighbors are close in time)",
                    ctx,
                    clip_id,
                    gap,
                    max_gap_seconds,
                    current.chunk_index,
                )
                range_end = max(range_end, current.end_time_seconds)
                chunk_hits.append(current)
            else:
                logger.debug(
                    "[chunk_range_merge] %sclip_id=%s split_range: gap=%.4fs > max_gap=%.4fs "
                    "flush current range; next range starts at chunk_idx=%s",
                    ctx,
                    clip_id,
                    gap,
                    max_gap_seconds,
                    current.chunk_index,
                )
                close_range("time_gap_exceeded")
                range_start = current.start_time_seconds
                range_end = current.end_time_seconds
                local_key = current.local_key
                file_name = current.file_name
                modality = current.modality
                chunk_hits = [current]

        close_range("end_of_sorted_hits")

    merged.sort(key=lambda m: m.confidence, reverse=True)
    logger.info(
        "[chunk_range_merge] %sdone merged_ranges=%s ordered_by=confidence_desc "
        "(cross-clip ordering is only for presentation; merge logic is per-clip)",
        ctx,
        len(merged),
    )
    return merged

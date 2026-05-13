"""Merge adjacent chunk hits (ported from Iris-Main TemporalRangeScorer).

Spike-aware: within each time-contiguous component, when score spread is wide
enough, keep only chunks whose similarity is near the local peak (floor =
peak - spread * max_normalized_drop), so weak shoulders before/after a spike
are not chained into the same range. Optional boundary refinement interpolates
the crossing time between chunk centers and the floor threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.services.semantic_constants import (
    CHUNK_MERGE_MINIMUM_SCORE,
    RANGE_MERGE_GAP_SECONDS,
    RANGE_MERGE_MAX_NORMALIZED_DROP,
    RANGE_MERGE_MIN_CLIP_SCORE_SPREAD,
    RANGE_MERGE_MIN_SCORE_RATIO_AFTER_FIRST_CHUNK,
)

logger = logging.getLogger(__name__)

_SCORE_INTERP_EPS = 1e-9


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


def _dedupe_hits_by_chunk_index(clip_hits: list[ChunkHit]) -> list[ChunkHit]:
    """Keep one row per chunk_index (highest similarity), e.g. duplicate NN rows per modality."""
    best_by_idx: dict[int, ChunkHit] = {}
    for h in clip_hits:
        prev = best_by_idx.get(h.chunk_index)
        if prev is None or h.score > prev.score:
            best_by_idx[h.chunk_index] = h
    return list(best_by_idx.values())


def _split_time_components(
    sorted_hits: list[ChunkHit],
    *,
    max_gap_seconds: float,
) -> list[list[ChunkHit]]:
    """Split into maximal time-contiguous runs (gap = next.start - prev.end)."""
    if not sorted_hits:
        return []
    out: list[list[ChunkHit]] = [[sorted_hits[0]]]
    for h in sorted_hits[1:]:
        gap = h.start_time_seconds - out[-1][-1].end_time_seconds
        if gap > max_gap_seconds:
            out.append([h])
        else:
            out[-1].append(h)
    return out


def _interp_rising_threshold_time(prev_hit: ChunkHit, hi_hit: ChunkHit, floor: float) -> float | None:
    """Time where linear score between chunk centers crosses ``floor`` (rising edge)."""
    t0, s0 = prev_hit.center_time_seconds, prev_hit.score
    t1, s1 = hi_hit.center_time_seconds, hi_hit.score
    if s1 - s0 <= _SCORE_INTERP_EPS:
        return None
    if not (s0 < floor <= s1):
        return None
    return t0 + (floor - s0) / (s1 - s0) * (t1 - t0)


def _interp_falling_threshold_time(hi_hit: ChunkHit, next_hit: ChunkHit, floor: float) -> float | None:
    """Time where linear score between chunk centers crosses ``floor`` (falling edge)."""
    t0, s0 = hi_hit.center_time_seconds, hi_hit.score
    t1, s1 = next_hit.center_time_seconds, next_hit.score
    if s0 - s1 <= _SCORE_INTERP_EPS:
        return None
    if not (s1 < floor <= s0):
        return None
    return t0 + (floor - s0) / (s1 - s0) * (t1 - t0)


def _refine_island_wall_times(
    component: list[ChunkHit],
    island_hits: list[ChunkHit],
    floor: float,
) -> tuple[float, float]:
    """Union of island windows, optionally tightened using floor crossing vs neighbors."""
    if not island_hits:
        raise ValueError("island_hits must be non-empty")

    first = island_hits[0]
    last = island_hits[-1]
    raw_start = min(h.start_time_seconds for h in island_hits)
    raw_end = max(h.end_time_seconds for h in island_hits)

    try:
        i_first = component.index(first)
    except ValueError:
        i_first = -1
    try:
        i_last = component.index(last)
    except ValueError:
        i_last = -1

    start_t = raw_start
    if i_first > 0:
        prev_hit = component[i_first - 1]
        t_rise = _interp_rising_threshold_time(prev_hit, first, floor)
        if t_rise is not None:
            # Crossing should land at or after the first spike window starts.
            start_t = max(raw_start, t_rise)

    end_t = raw_end
    if i_last >= 0 and i_last + 1 < len(component):
        next_hit = component[i_last + 1]
        t_fall = _interp_falling_threshold_time(last, next_hit, floor)
        if t_fall is not None:
            end_t = min(raw_end, t_fall)

    if end_t < start_t:
        end_t = raw_end
        start_t = raw_start
    return start_t, end_t


def _prune_tail_by_score_ratio_to_first_chunk(
    seg_hits: list[ChunkHit],
    *,
    min_ratio: float,
) -> list[ChunkHit]:
    """Keep the first chunk (by start time), then only further chunks while score/first > min_ratio.

    On first failure, omit that chunk and all later chunks in this segment (truncate the tail).
    """
    if len(seg_hits) <= 1:
        return seg_hits
    ordered = sorted(seg_hits, key=lambda h: h.start_time_seconds)
    anchor = ordered[0].score
    if anchor <= _SCORE_INTERP_EPS:
        return ordered
    out: list[ChunkHit] = [ordered[0]]
    for h in ordered[1:]:
        if h.score / anchor > min_ratio:
            out.append(h)
        else:
            break
    return out


def _segment_by_score_floor(
    component: list[ChunkHit],
    *,
    floor: float,
) -> list[tuple[str, list[ChunkHit]]]:
    """Split into alternating time-contiguous runs above vs below the peak-relative floor.

    Below-floor runs are still returned as separate ranges (weak tails / shoulders) so
    behavior matches the prior drop-split tests while spike islands stay tight.
    """
    segments: list[tuple[str, list[ChunkHit]]] = []
    cur: list[ChunkHit] = []
    cur_kind: str | None = None
    for h in component:
        kind = "high" if h.score >= floor else "low"
        if not cur:
            cur = [h]
            cur_kind = kind
        elif kind == cur_kind:
            cur.append(h)
        else:
            if cur_kind is not None:
                segments.append((cur_kind, cur))
            cur = [h]
            cur_kind = kind
    if cur and cur_kind is not None:
        segments.append((cur_kind, cur))
    return segments


def _merge_time_component(
    component: list[ChunkHit],
    *,
    clip_id: int,
    max_normalized_drop_vs_clip_spread: float,
    min_clip_score_spread: float,
    min_score_ratio_after_first_chunk: float,
    project_id: int | None,
    ctx: str,
) -> list[MergedRange]:
    """Merge one time-contiguous chunk list into one or more MergedRange rows."""
    scores = [h.score for h in component]
    comp_min = min(scores)
    comp_max = max(scores)
    comp_spread = comp_max - comp_min
    use_peak_floor = comp_spread >= min_clip_score_spread

    logger.info(
        "[chunk_range_merge] %sclip_id=%s time_component chunks=%s comp_score_min=%.4f "
        "comp_score_max=%.4f comp_spread=%.4f peak_floor_mode=%s",
        ctx,
        clip_id,
        len(component),
        comp_min,
        comp_max,
        comp_spread,
        use_peak_floor,
    )

    first = component[0]
    local_key = first.local_key
    file_name = first.file_name
    modality = first.modality

    out: list[MergedRange] = []

    if not use_peak_floor:
        pruned = _prune_tail_by_score_ratio_to_first_chunk(
            component,
            min_ratio=min_score_ratio_after_first_chunk,
        )
        rs = min(h.start_time_seconds for h in pruned)
        re = max(h.end_time_seconds for h in pruned)
        mr = _build_merged_group(
            clip_id=clip_id,
            local_key=local_key,
            file_name=file_name,
            modality=modality,
            start_time_seconds=rs,
            end_time_seconds=re,
            chunk_hits=pruned,
        )
        out.append(mr)
        _log_merged_range(
            mr,
            pruned,
            project_id=project_id,
            range_closure="component_low_spread_merge",
        )
        return out

    floor = comp_max - comp_spread * max_normalized_drop_vs_clip_spread
    logger.info(
        "[chunk_range_merge] %sclip_id=%s peak_relative_floor=%.4f "
        "(peak - comp_spread * max_normalized_drop_vs_clip_spread)",
        ctx,
        clip_id,
        floor,
    )

    segments = _segment_by_score_floor(component, floor=floor)
    if not segments:
        rs = min(h.start_time_seconds for h in component)
        re = max(h.end_time_seconds for h in component)
        mr = _build_merged_group(
            clip_id=clip_id,
            local_key=local_key,
            file_name=file_name,
            modality=modality,
            start_time_seconds=rs,
            end_time_seconds=re,
            chunk_hits=component,
        )
        out.append(mr)
        _log_merged_range(
            mr,
            component,
            project_id=project_id,
            range_closure="component_peak_floor_fallback",
        )
        return out

    for kind, seg_hits in segments:
        pruned = _prune_tail_by_score_ratio_to_first_chunk(
            seg_hits,
            min_ratio=min_score_ratio_after_first_chunk,
        )
        if kind == "high":
            rs, re = _refine_island_wall_times(component, pruned, floor)
            closure = "peak_relative_floor_island"
            if rs != min(h.start_time_seconds for h in pruned) or re != max(
                h.end_time_seconds for h in pruned
            ):
                closure = "peak_relative_floor_island_interpolated"
        else:
            rs = min(h.start_time_seconds for h in pruned)
            re = max(h.end_time_seconds for h in pruned)
            closure = "below_peak_floor_segment"
        if len(pruned) < len(seg_hits):
            closure = f"{closure}_ratio_tail_pruned"
        mr = _build_merged_group(
            clip_id=clip_id,
            local_key=local_key,
            file_name=file_name,
            modality=modality,
            start_time_seconds=rs,
            end_time_seconds=re,
            chunk_hits=pruned,
        )
        out.append(mr)
        _log_merged_range(mr, pruned, project_id=project_id, range_closure=closure)

    return out


def merge_chunk_hits(
    hits: list[ChunkHit],
    *,
    max_gap_seconds: float = RANGE_MERGE_GAP_SECONDS,
    minimum_score: float = CHUNK_MERGE_MINIMUM_SCORE,
    max_normalized_drop_vs_clip_spread: float = RANGE_MERGE_MAX_NORMALIZED_DROP,
    min_clip_score_spread: float = RANGE_MERGE_MIN_CLIP_SCORE_SPREAD,
    min_score_ratio_after_first_chunk: float = RANGE_MERGE_MIN_SCORE_RATIO_AFTER_FIRST_CHUNK,
    dedupe_by_chunk_index: bool = True,
    project_id: int | None = None,
) -> list[MergedRange]:
    ctx = f"project_id={project_id} " if project_id is not None else ""
    logger.info(
        "[chunk_range_merge] %sstart raw_hits=%s thresholds: minimum_score=%.4f (drop at <=) "
        "max_gap_seconds=%.4f min_clip_score_spread=%.4f max_normalized_drop_vs_clip_spread=%.4f "
        "dedupe_by_chunk_index=%s min_score_ratio_after_first_chunk=%.4f (strict >) - per clip: dedupe, "
        "sort by time, split by time gap; within each time component, if comp_spread >= min_clip_score_spread, "
        "keep contiguous chunks with score >= peak - comp_spread * max_normalized_drop (else merge whole "
        "component); optional boundary interpolation at floor crossings; then drop tail chunks whose score "
        "divided by the segment's first chunk (time order) is not strictly above min_score_ratio_after_first_chunk.",
        ctx,
        len(hits),
        minimum_score,
        max_gap_seconds,
        min_clip_score_spread,
        max_normalized_drop_vs_clip_spread,
        dedupe_by_chunk_index,
        min_score_ratio_after_first_chunk,
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
        pre_dedupe = len(clip_hits)
        if dedupe_by_chunk_index:
            clip_hits = _dedupe_hits_by_chunk_index(clip_hits)
            removed = pre_dedupe - len(clip_hits)
            if removed:
                logger.info(
                    "[chunk_range_merge] %sclip_id=%s dedupe_by_chunk_index removed=%s kept=%s",
                    ctx,
                    clip_id,
                    removed,
                    len(clip_hits),
                )

        sorted_hits = sorted(clip_hits, key=lambda h: h.start_time_seconds)
        if not sorted_hits:
            continue

        clip_scores = [h.score for h in sorted_hits]
        clip_min = min(clip_scores)
        clip_max = max(clip_scores)
        clip_spread = clip_max - clip_min
        logger.info(
            "[chunk_range_merge] %sclip_id=%s timeline eligible_chunks=%s "
            "clip_score_min=%.4f clip_score_max=%.4f clip_spread=%.4f (per-time-component spread used for merge)",
            ctx,
            clip_id,
            len(sorted_hits),
            clip_min,
            clip_max,
            clip_spread,
        )

        components = _split_time_components(sorted_hits, max_gap_seconds=max_gap_seconds)
        logger.info(
            "[chunk_range_merge] %sclip_id=%s time_components=%s (max_gap_seconds=%.4f)",
            ctx,
            clip_id,
            len(components),
            max_gap_seconds,
        )

        for comp in components:
            merged.extend(
                _merge_time_component(
                    comp,
                    clip_id=clip_id,
                    max_normalized_drop_vs_clip_spread=max_normalized_drop_vs_clip_spread,
                    min_clip_score_spread=min_clip_score_spread,
                    min_score_ratio_after_first_chunk=min_score_ratio_after_first_chunk,
                    project_id=project_id,
                    ctx=ctx,
                )
            )

    merged.sort(key=lambda m: m.confidence, reverse=True)
    logger.info(
        "[chunk_range_merge] %sdone merged_ranges=%s ordered_by=confidence_desc "
        "(cross-clip ordering is only for presentation; merge logic is per-clip)",
        ctx,
        len(merged),
    )
    return merged

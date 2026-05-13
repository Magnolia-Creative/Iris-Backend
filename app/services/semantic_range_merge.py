"""Merge adjacent chunk hits (ported from Iris-Main TemporalRangeScorer)."""

from __future__ import annotations

from dataclasses import dataclass

from app.services.semantic_constants import CHUNK_MERGE_MINIMUM_SCORE, RANGE_MERGE_GAP_SECONDS


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
    return MergedRange(
        clip_id=clip_id,
        local_key=local_key,
        file_name=file_name,
        start_time_seconds=start_time_seconds,
        end_time_seconds=end_time_seconds,
        confidence=confidence,
        source=modality,
    )


def merge_chunk_hits(
    hits: list[ChunkHit],
    *,
    max_gap_seconds: float = RANGE_MERGE_GAP_SECONDS,
    minimum_score: float = CHUNK_MERGE_MINIMUM_SCORE,
) -> list[MergedRange]:
    eligible = [h for h in hits if h.score > minimum_score]
    if not eligible:
        return []

    grouped: dict[int, list[ChunkHit]] = {}
    for h in eligible:
        grouped.setdefault(h.clip_id, []).append(h)

    merged: list[MergedRange] = []
    for clip_id, clip_hits in grouped.items():
        sorted_hits = sorted(clip_hits, key=lambda h: h.start_time_seconds)
        if not sorted_hits:
            continue

        first = sorted_hits[0]
        range_start = first.start_time_seconds
        range_end = first.end_time_seconds
        score_sum = first.score
        score_count = 1
        peak = first.score
        local_key = first.local_key
        file_name = first.file_name
        modality = first.modality
        chunk_hits = [first]

        for current in sorted_hits[1:]:
            gap = current.start_time_seconds - range_end
            if gap <= max_gap_seconds:
                range_end = max(range_end, current.end_time_seconds)
                score_sum += current.score
                score_count += 1
                peak = max(peak, current.score)
                chunk_hits.append(current)
            else:
                merged.append(
                    _build_merged_group(
                        clip_id=clip_id,
                        local_key=local_key,
                        file_name=file_name,
                        modality=modality,
                        start_time_seconds=range_start,
                        end_time_seconds=range_end,
                        chunk_hits=chunk_hits,
                    )
                )
                range_start = current.start_time_seconds
                range_end = current.end_time_seconds
                score_sum = current.score
                score_count = 1
                peak = current.score
                local_key = current.local_key
                file_name = current.file_name
                modality = current.modality
                chunk_hits = [current]

        merged.append(
            _build_merged_group(
                clip_id=clip_id,
                local_key=local_key,
                file_name=file_name,
                modality=modality,
                start_time_seconds=range_start,
                end_time_seconds=range_end,
                chunk_hits=chunk_hits,
            )
        )

    merged.sort(key=lambda m: m.confidence, reverse=True)
    return merged

"""Unit tests for semantic chunk hit merging."""

from __future__ import annotations

from app.services.semantic_range_merge import ChunkHit, merge_chunk_hits


def _hit(
    clip_id: int,
    chunk_index: int,
    start: float,
    end: float,
    score: float,
    *,
    modality: str = "audio",
) -> ChunkHit:
    return ChunkHit(
        clip_id=clip_id,
        local_key="k",
        file_name=None,
        modality=modality,
        chunk_index=chunk_index,
        start_time_seconds=start,
        end_time_seconds=end,
        center_time_seconds=(start + end) / 2,
        score=score,
    )


def test_dedupe_same_chunk_index_keeps_highest_score():
    hits = [
        _hit(1, 0, 0.0, 4.0, 0.40),
        _hit(1, 0, 0.0, 4.0, 0.65),
    ]
    merged = merge_chunk_hits(hits, project_id=None)
    assert len(merged) == 1
    assert merged[0].start_time_seconds == 0.0
    assert merged[0].end_time_seconds == 4.0
    assert merged[0].confidence > 0.6


def test_score_drop_splits_chain_despite_time_gap_ok():
    hits = [
        _hit(1, 0, 0.0, 4.0, 0.62),
        _hit(1, 1, 3.5, 7.5, 0.60),
        _hit(1, 2, 7.0, 11.0, 0.50),
    ]
    merged = merge_chunk_hits(hits)
    assert len(merged) == 2
    merged_by_start = sorted(merged, key=lambda m: m.start_time_seconds)
    # High segment ends before the weak tail thanks to floor + interpolated fall.
    assert merged_by_start[0].end_time_seconds < merged_by_start[1].start_time_seconds + 1e-6
    assert merged_by_start[0].end_time_seconds < 7.5
    assert merged_by_start[1].start_time_seconds == 7.0
    assert merged_by_start[0].source == "audio"


def test_similar_scores_remain_one_range():
    hits = [
        _hit(1, 0, 0.0, 4.0, 0.52),
        _hit(1, 1, 3.5, 7.5, 0.50),
        _hit(1, 2, 7.0, 11.0, 0.51),
    ]
    merged = merge_chunk_hits(hits)
    assert len(merged) == 1


def test_time_gap_still_splits():
    hits = [
        _hit(1, 0, 0.0, 4.0, 0.90),
        _hit(1, 1, 10.0, 14.0, 0.88),
    ]
    merged = merge_chunk_hits(hits)
    assert len(merged) == 2


def test_absolute_drop_triggers_split():
    hits = [
        _hit(1, 0, 0.0, 4.0, 0.70),
        _hit(1, 1, 3.5, 7.5, 0.58),
    ]
    merged = merge_chunk_hits(hits)
    assert len(merged) == 2


def test_ratio_drop_triggers_split_high_peak():
    hits = [
        _hit(1, 0, 0.0, 4.0, 0.80),
        _hit(1, 1, 3.5, 7.5, 0.69),
    ]
    merged = merge_chunk_hits(hits)
    assert len(merged) == 2


def test_low_clip_spread_skips_score_tier_split():
    """Narrow NN band: spread below floor so only time-gap rules apply."""
    hits = [
        _hit(1, 0, 0.0, 4.0, 0.72),
        _hit(1, 1, 3.5, 7.5, 0.68),
    ]
    merged = merge_chunk_hits(hits)
    assert len(merged) == 1


def test_identical_scores_merge_when_time_gap_ok():
    hits = [
        _hit(1, 0, 0.0, 4.0, 0.55),
        _hit(1, 1, 3.5, 7.5, 0.55),
        _hit(1, 2, 7.0, 11.0, 0.55),
    ]
    merged = merge_chunk_hits(hits)
    assert len(merged) == 1


def test_custom_normalized_threshold_allows_merge():
    hits = [
        _hit(1, 0, 0.0, 4.0, 0.80),
        _hit(1, 1, 3.5, 7.5, 0.69),
    ]
    merged = merge_chunk_hits(
        hits,
        max_normalized_drop_vs_clip_spread=1.05,
    )
    assert len(merged) == 1


def test_waving_flag_like_scores_peak_island_top_ranked():
    """Regression: weak shoulders/tails must not merge into the dominant spike range."""
    hits = [
        _hit(1, 0, 0.0, 4.0, 0.508),
        _hit(1, 1, 3.5, 7.5, 0.514),
        _hit(1, 2, 7.0, 11.0, 0.596),
        _hit(1, 3, 10.5, 14.5, 0.612),
        _hit(1, 4, 14.0, 18.0, 0.594),
        _hit(1, 5, 17.5, 21.5, 0.509),
        _hit(1, 6, 21.0, 25.0, 0.517),
        _hit(1, 7, 21.33, 25.33, 0.524),
    ]
    merged = merge_chunk_hits(hits, project_id=None)
    assert len(merged) == 3
    top = max(merged, key=lambda m: m.confidence)
    # Not the old 0–18s single-range anchor at t=0.
    assert top.start_time_seconds >= 7.0
    assert top.start_time_seconds >= 7.8
    assert top.end_time_seconds <= 18.0
    assert top.end_time_seconds <= 17.0
    assert top.end_time_seconds - top.start_time_seconds < 11.0
    assert top.confidence >= 0.58



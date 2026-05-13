"""Unit tests for semantic search result selection."""

from __future__ import annotations

from app.services.semantic_range_merge import MergedRange
from app.services.semantic_search_service import _select_top_matches_by_confidence_ratio


def _range(idx: int, confidence: float) -> MergedRange:
    return MergedRange(
        clip_id=idx,
        local_key=f"k{idx}",
        file_name=None,
        start_time_seconds=float(idx),
        end_time_seconds=float(idx + 1),
        confidence=confidence,
        source="visual_frame",
    )


def test_select_top_matches_drops_final_results_below_confidence_ratio():
    matches = [
        _range(1, 0.6070337464842742),
        _range(2, 0.5326880075946838),
    ]

    selected = _select_top_matches_by_confidence_ratio(matches, limit=5)

    assert len(selected) == 1
    assert selected[0].clip_id == 1
    assert matches[1].confidence / matches[0].confidence < 0.92


def test_select_top_matches_keeps_near_top_results_until_limit():
    matches = [
        _range(1, 1.00),
        _range(2, 0.93),
        _range(3, 0.91),
    ]

    selected = _select_top_matches_by_confidence_ratio(matches, limit=2)

    assert [m.clip_id for m in selected] == [1, 2]

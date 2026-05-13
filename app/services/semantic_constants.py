"""Semantic chunking constants aligned with Iris-Main Features/Semantic."""

CHUNK_DURATION_SECONDS = 4.0
CHUNK_OVERLAP_SECONDS = 0.5
CHUNK_STRIDE_SECONDS = max(0.1, CHUNK_DURATION_SECONDS - CHUNK_OVERLAP_SECONDS)
CHUNK_TOP_K = 10
CHUNK_MERGE_MINIMUM_SCORE = 0.15
RANGE_MERGE_GAP_SECONDS = 0.6
# When extending a range, do not absorb a chunk much weaker than the best chunk
# already in that range (in addition to the time-gap rule).
RANGE_MERGE_MAX_SCORE_DROP_FROM_PEAK = 0.10
RANGE_MERGE_MIN_SCORE_RATIO_OF_PEAK = 0.88
RESULTS_LIMIT = 3

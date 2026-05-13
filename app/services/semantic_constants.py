"""Semantic chunking constants aligned with Iris-Main Features/Semantic."""

CHUNK_DURATION_SECONDS = 4.0
CHUNK_OVERLAP_SECONDS = 0.5
CHUNK_STRIDE_SECONDS = max(0.1, CHUNK_DURATION_SECONDS - CHUNK_OVERLAP_SECONDS)
CHUNK_TOP_K = 10
CHUNK_MERGE_MINIMUM_SCORE = 0.15
RANGE_MERGE_GAP_SECONDS = 0.6
# Score-tier split uses this clip's min/max similarity among eligible chunks (after dedupe).
# Spread below this skips score-based splitting (only time-gap splits), so near-flat NN scores
# do not fragment ranges.
RANGE_MERGE_MIN_CLIP_SCORE_SPREAD = 0.08
# If (range_peak - next_chunk_score) / clip_score_spread exceeds this, start a new range.
# Scale-free within each clip: adapts when similarities are compressed (e.g. all high) vs wide.
RANGE_MERGE_MAX_NORMALIZED_DROP = 0.35
# After the first chunk in a merged span (time order), each next chunk is kept only while
# its score divided by the first chunk's score stays strictly above this ratio (weak tails drop off).
RANGE_MERGE_MIN_SCORE_RATIO_AFTER_FIRST_CHUNK = 0.92
RESULTS_LIMIT = 3

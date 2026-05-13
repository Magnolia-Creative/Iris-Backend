"""4s windows with 0.5s overlap (3.5s stride), matching SemanticSearchPipeline.chunkPoints."""

from app.services.semantic_constants import CHUNK_DURATION_SECONDS, CHUNK_STRIDE_SECONDS


def chunk_time_windows(duration_seconds: float) -> list[tuple[float, float, float]]:
    """Return (start, end, center) for each chunk window."""
    if duration_seconds <= 0:
        return []

    chunk_duration = min(CHUNK_DURATION_SECONDS, duration_seconds)
    if chunk_duration <= 0:
        return []

    if duration_seconds <= chunk_duration:
        return [
            (
                0.0,
                duration_seconds,
                duration_seconds / 2.0,
            )
        ]

    stride = CHUNK_STRIDE_SECONDS
    starts: list[float] = []
    current_start = 0.0
    while current_start + chunk_duration < duration_seconds:
        starts.append(current_start)
        current_start += stride

    final_start = max(0.0, duration_seconds - chunk_duration)
    if starts:
        if abs(starts[-1] - final_start) > 0.001:
            starts.append(final_start)
    else:
        starts.append(final_start)

    out: list[tuple[float, float, float]] = []
    for start in starts:
        end = min(start + chunk_duration, duration_seconds)
        center = min(duration_seconds, start + (end - start) / 2.0)
        out.append((start, end, center))
    return out

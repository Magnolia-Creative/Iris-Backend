from app.services.chunk_time_windows import chunk_time_windows
from app.services.semantic_constants import CHUNK_DURATION_SECONDS, CHUNK_STRIDE_SECONDS


def test_chunk_time_windows_short_clip():
    w = chunk_time_windows(2.0)
    assert len(w) == 1
    start, end, center = w[0]
    assert start == 0.0
    assert end == 2.0
    assert center == 1.0


def test_chunk_time_windows_stride_and_overlap():
    # 10s duration: multiple 4s windows with 3.5s stride
    w = chunk_time_windows(10.0)
    assert w[0][0] == 0.0
    assert abs(w[0][1] - min(CHUNK_DURATION_SECONDS, 10.0)) < 0.001
    if len(w) > 1:
        assert abs(w[1][0] - CHUNK_STRIDE_SECONDS) < 0.001

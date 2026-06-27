from time import perf_counter


def elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)

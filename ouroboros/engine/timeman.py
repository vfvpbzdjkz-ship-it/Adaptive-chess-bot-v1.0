"""Clock → simulation budget mapping."""
import time
import collections
import logging

log = logging.getLogger(__name__)

_WINDOW = 20  # rolling window for sims/sec measurement


class TimeManager:
    def __init__(self, base_sims: int = 96):
        self.base_sims = base_sims
        self._timings: collections.deque = collections.deque(maxlen=_WINDOW)
        self._sims_per_sec: float = 1.0  # initial pessimistic estimate

    def record(self, sims: int, elapsed: float) -> None:
        if elapsed > 0.01:
            rate = sims / elapsed
            self._timings.append(rate)
            self._sims_per_sec = sum(self._timings) / len(self._timings)

    def budget_sims(
        self,
        remaining_ms: int,
        increment_ms: int,
        cap_sims: int,
        book_speed: bool = False,
    ) -> int:
        if book_speed:
            return max(1, cap_sims // 16)

        remaining_s = remaining_ms / 1000.0
        increment_s = increment_ms / 1000.0

        # Never flag
        if remaining_s < 10.0:
            think_s = 0.5
        else:
            target_s = remaining_s / 30.0 + 0.8 * increment_s
            hard_ceil = 0.20 * remaining_s
            think_s = max(0.1, min(target_s, hard_ceil))

        sims = int(think_s * max(self._sims_per_sec, 1.0))
        sims = max(1, min(sims, cap_sims))
        return sims


class Timer:
    def __init__(self):
        self._start = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start

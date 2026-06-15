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
        # Initial guess until the first move is timed. Starting too low wastes the
        # generous early-game budget of a classical clock on a near-empty search.
        self._sims_per_sec: float = 25.0

    @property
    def sims_per_sec(self) -> float:
        return self._sims_per_sec

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
    ) -> tuple[int, float]:
        """Return (n_sims, think_seconds)."""
        if book_speed:
            return max(1, cap_sims // 16), 0.1

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
        # When we have real time to think, never search trivially few nodes; the
        # per-move wall-clock deadline (set by the caller) still cuts it short.
        floor = min(cap_sims, 64) if remaining_s >= 30.0 else 1
        sims = max(floor, min(sims, cap_sims))
        return sims, think_s


class Timer:
    def __init__(self):
        self._start = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start

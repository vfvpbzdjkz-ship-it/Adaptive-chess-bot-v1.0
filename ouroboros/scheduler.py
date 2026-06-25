"""Hourly Lichess / self-play mode scheduler."""
import logging
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)

SLOT_SECONDS = 3600  # one hour per mode

_lock = threading.Lock()
_lichess_active: bool = True
_switch_at: float = 0.0


def is_lichess_active() -> bool:
    with _lock:
        return _lichess_active


class PlayScheduler:
    """Alternates between Lichess-play and self-play-only every hour.

    During Lichess hours the matchmaker sends challenges and the event loop
    accepts incoming ones. During self-play hours the matchmaker is stopped
    and incoming challenges are declined so the training loop runs unthrottled.
    """

    def __init__(self, matchmaker) -> None:
        self._matchmaker = matchmaker
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Non-blocking lock: if a custom challenge is already in flight, skip the new one.
        self._challenge_lock = threading.Lock()

    def start(self) -> None:
        global _lichess_active, _switch_at
        with _lock:
            _lichess_active = True
            _switch_at = time.time() + SLOT_SECONDS
            sw = _switch_at
        self._notify_viewer("lichess", sw)
        self._matchmaker.start()   # begin in Lichess mode immediately
        log.info("PlayScheduler started — switching mode every %d min", SLOT_SECONDS // 60)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def force_one_game(self) -> None:
        """Immediately trigger one challenge cycle regardless of current mode."""
        threading.Thread(
            target=self._matchmaker.challenge_once, daemon=True
        ).start()
        log.info("Force-one-game requested")

    def challenge_with_options(self, username: str, time_limit: int, increment: int) -> None:
        """Challenge with user-specified opponent and/or time control from the GUI."""
        if not self._challenge_lock.acquire(blocking=False):
            log.debug("Custom challenge skipped — one already in flight")
            return
        log.info("Custom challenge: opp=%r tc=%d+%d", username or "(auto)", time_limit, increment)

        def _run():
            try:
                if username:
                    self._matchmaker.challenge_specific(username, time_limit, increment)
                else:
                    self._matchmaker.challenge_once_with_tc(time_limit, increment)
            finally:
                self._challenge_lock.release()

        threading.Thread(target=_run, daemon=True).start()

    def _loop(self) -> None:
        global _lichess_active, _switch_at
        while not self._stop.wait(SLOT_SECONDS):
            try:
                with _lock:
                    _lichess_active = not _lichess_active
                    new_active = _lichess_active
                    _switch_at = time.time() + SLOT_SECONDS
                    sw = _switch_at

                log.info("Mode → %s", "LICHESS" if new_active else "SELFPLAY")
                self._notify_viewer("lichess" if new_active else "selfplay", sw)

                if new_active:
                    self._matchmaker.start()
                else:
                    self._matchmaker.stop()
            except Exception as e:
                log.error("PlayScheduler error: %s", e)

    @staticmethod
    def _notify_viewer(mode: str, switch_at: float) -> None:
        try:
            from ouroboros.web_viewer import update_play_mode
            update_play_mode(mode, switch_at)
        except Exception:
            pass

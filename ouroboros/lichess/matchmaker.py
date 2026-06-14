"""Auto-challenge online bots for training games."""
import logging
import random
import threading
import time
from typing import Optional

from ouroboros.lichess.client import LichessClient
from ouroboros.persistence import get_db

log = logging.getLogger(__name__)

CHALLENGE_INTERVAL = 180  # seconds between challenges (max N/hr enforced)
MAX_PER_HOUR = 10
MAX_RETRIES = 5  # how many different bots to try per cycle before giving up


def _set_next_challenge(ts: float) -> None:
    try:
        from ouroboros.web_viewer import update_next_challenge
        update_next_challenge(ts)
    except Exception:
        pass


class Matchmaker:
    def __init__(self, client: LichessClient, cfg: dict, on_game_start=None):
        self.client = client
        self.cfg = cfg
        self.on_game_start = on_game_start
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._challenges_this_hour = 0
        self._hour_start = time.time()

    def start(self) -> None:
        if not self.cfg.get("matchmaker_enabled", True):
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("Matchmaker started")
        _set_next_challenge(time.time() + CHALLENGE_INTERVAL)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _loop(self) -> None:
        while not self._stop_event.wait(CHALLENGE_INTERVAL):
            try:
                self._cycle()
            except Exception as e:
                log.error("Matchmaker cycle error (will retry next interval): %s", e, exc_info=True)
            _set_next_challenge(time.time() + CHALLENGE_INTERVAL)

    def _cycle(self) -> None:
        """One challenge attempt cycle."""
        # Reset hourly counter
        if time.time() - self._hour_start >= 3600:
            self._challenges_this_hour = 0
            self._hour_start = time.time()

        max_per_hour = self.cfg.get("matchmaker_max_per_hour", MAX_PER_HOUR)
        if self._challenges_this_hour >= max_per_hour:
            return

        t_limit = self.cfg.get("matchmaker_time", 5)
        t_inc = self.cfg.get("matchmaker_increment", 3)
        tried: set[str] = set()

        for attempt in range(MAX_RETRIES):
            target = self._pick_target(exclude=tried)
            if not target:
                break

            username = target.get("id", "")
            if not username:
                break
            tried.add(username)

            label = f" [retry {attempt}]" if attempt else ""
            log.info("Matchmaker: challenging %s (%d+%d)%s", username, t_limit, t_inc, label)

            try:
                result = self.client.challenge_player(username, t_limit, t_inc, rated=False)
                if result is not None:
                    self._challenges_this_hour += 1
                    break
                log.debug("Challenge to %s returned no data; trying another bot", username)
            except Exception as e:
                log.debug("Challenge to %s failed (%s); trying another bot", username, e)

    def _pick_target(self, exclude: set[str] | None = None) -> Optional[dict]:
        """Pick a bot to challenge. Prefer revenge targets; skip already-tried ones."""
        exclude = exclude or set()

        # Revenge targets first (bots that have beaten us more than we've beaten them)
        try:
            with get_db() as conn:
                rows = conn.execute(
                    "SELECT username FROM opponents "
                    "WHERE is_bot=1 AND losses_vs_us > wins_vs_us "
                    "ORDER BY losses_vs_us-wins_vs_us DESC LIMIT 10",
                ).fetchall()
            revenge_targets = [r["username"] for r in rows if r["username"] not in exclude]
        except Exception as e:
            log.debug("DB revenge-target lookup failed: %s", e)
            revenge_targets = []

        if revenge_targets:
            return {"id": random.choice(revenge_targets)}

        # Fall back to a random online bot, excluding already-tried ones
        try:
            bots = []
            for bot in self.client.get_online_bots():
                if bot.get("id", "") not in exclude:
                    bots.append(bot)
                if len(bots) >= 50:
                    break
            if not bots:
                return None
            return random.choice(bots)
        except Exception as e:
            log.debug("Failed to get online bots: %s", e)
            return None

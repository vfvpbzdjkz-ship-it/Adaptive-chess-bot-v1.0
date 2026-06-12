"""Auto-challenge online bots for training games."""
import logging
import random
import threading
import time
from typing import Optional

from ouroboros.lichess.client import LichessClient
from ouroboros.persistence import get_db

log = logging.getLogger(__name__)

CHALLENGE_INTERVAL = 300  # seconds between challenges (max N/hr enforced)
MAX_PER_HOUR = 10


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

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _loop(self) -> None:
        while not self._stop_event.wait(CHALLENGE_INTERVAL):
            # Reset hourly counter
            if time.time() - self._hour_start >= 3600:
                self._challenges_this_hour = 0
                self._hour_start = time.time()

            max_per_hour = self.cfg.get("matchmaker_max_per_hour", MAX_PER_HOUR)
            if self._challenges_this_hour >= max_per_hour:
                continue

            target = self._pick_target()
            if not target:
                continue

            username = target.get("id", "")
            if not username:
                continue

            t_limit = self.cfg.get("matchmaker_time", 5)
            t_inc = self.cfg.get("matchmaker_increment", 3)
            log.info("Matchmaker: challenging %s (%d+%d)", username, t_limit, t_inc)

            try:
                self.client.challenge_player(username, t_limit, t_inc, rated=False)
                self._challenges_this_hour += 1
            except Exception as e:
                log.debug("Challenge to %s failed: %s", username, e)

    def _pick_target(self) -> Optional[dict]:
        """Pick a bot to challenge. Prefer bots with negative score against us."""
        # Get revenge targets first
        with get_db() as conn:
            rows = conn.execute(
                "SELECT username FROM opponents WHERE is_bot=1 AND losses_vs_us > wins_vs_us ORDER BY losses_vs_us-wins_vs_us DESC LIMIT 5",
            ).fetchall()
            revenge_targets = [r["username"] for r in rows]

        if revenge_targets:
            target_name = random.choice(revenge_targets)
            return {"id": target_name}

        # Otherwise pick a random online bot
        try:
            bots = []
            for bot in self.client.get_online_bots():
                bots.append(bot)
                if len(bots) >= 50:
                    break
            if not bots:
                return None
            return random.choice(bots)
        except Exception as e:
            log.debug("Failed to get online bots: %s", e)
            return None

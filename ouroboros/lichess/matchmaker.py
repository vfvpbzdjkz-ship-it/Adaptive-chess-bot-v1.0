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
        self._lock = threading.Lock()
        self._challenges_this_hour = 0
        self._hour_start = time.time()
        self._in_game = False   # set True while a Lichess game is active

    def set_in_game(self, active: bool) -> None:
        """Call with True when a game starts, False when it ends."""
        with self._lock:
            self._in_game = active

    def start(self) -> None:
        if not self.cfg.get("matchmaker_enabled", True):
            return
        if self._thread and self._thread.is_alive():
            log.debug("Matchmaker already running; skipping duplicate start")
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

    def challenge_once(self) -> None:
        """Run exactly one challenge cycle immediately (used by force-game button)."""
        try:
            self._cycle()
        except Exception as e:
            log.error("challenge_once error: %s", e, exc_info=True)

    def challenge_once_with_tc(self, time_limit: int, increment: int) -> None:
        """One auto-pick challenge cycle with a custom time control."""
        try:
            self._cycle(time_limit_override=time_limit, increment_override=increment)
        except Exception as e:
            log.error("challenge_once_with_tc error: %s", e, exc_info=True)

    def challenge_specific(self, username: str, time_limit: int, increment: int) -> None:
        """Directly challenge a named opponent with the given time control."""
        with self._lock:
            if self._in_game:
                log.debug("challenge_specific: skipping — game in progress")
                return
        log.info("Challenging %s (%d+%d) [user-specified]", username, time_limit, increment)
        try:
            result = self.client.challenge_player(username, time_limit, increment, rated=False)
            if result:
                with self._lock:
                    self._challenges_this_hour += 1
            else:
                log.debug("Challenge to %s returned no data", username)
        except Exception as e:
            log.error("challenge_specific to %s failed: %s", username, e)

    def _loop(self) -> None:
        while not self._stop_event.wait(CHALLENGE_INTERVAL):
            try:
                self._cycle()
            except Exception as e:
                log.error("Matchmaker cycle error (will retry next interval): %s", e, exc_info=True)
            _set_next_challenge(time.time() + CHALLENGE_INTERVAL)

    def _cycle(self, time_limit_override=None, increment_override=None) -> None:
        """One challenge attempt cycle."""
        with self._lock:
            if self._in_game:
                log.debug("Matchmaker: skipping challenge -- game in progress")
                return
            if time.time() - self._hour_start >= 3600:
                self._challenges_this_hour = 0
                self._hour_start = time.time()
            max_per_hour = self.cfg.get("matchmaker_max_per_hour", MAX_PER_HOUR)
            if self._challenges_this_hour >= max_per_hour:
                return

        t_limit = time_limit_override if time_limit_override is not None else self.cfg.get("matchmaker_time", 10)
        t_inc = increment_override if increment_override is not None else self.cfg.get("matchmaker_increment", 5)
        tried: set[str] = set()

        for attempt in range(MAX_RETRIES):
            target = self._pick_target(exclude=tried, time_limit=t_limit)
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
                    with self._lock:
                        self._challenges_this_hour += 1
                    break
                log.debug("Challenge to %s returned no data; trying another bot", username)
            except Exception as e:
                log.debug("Challenge to %s failed (%s); trying another bot", username, e)

    def _pick_target(self, exclude: set[str] | None = None, time_limit: int | None = None) -> Optional[dict]:
        """Pick a bot to challenge. Prefer revenge targets; filter out very strong bots."""
        exclude = exclude or set()

        # wins_vs_us = times opponent beat us; losses_vs_us = times we beat opponent
        # Revenge = bots that have beaten us more than we've beaten them
        try:
            with get_db() as conn:
                rows = conn.execute(
                    "SELECT username FROM opponents "
                    "WHERE is_bot=1 AND wins_vs_us > losses_vs_us "
                    "ORDER BY wins_vs_us-losses_vs_us DESC LIMIT 10",
                ).fetchall()
            revenge_targets = [r["username"] for r in rows if r["username"] not in exclude]
        except Exception as e:
            log.debug("DB revenge-target lookup failed: %s", e)
            revenge_targets = []

        if revenge_targets:
            return {"id": random.choice(revenge_targets)}

        # ELO-aware random bot selection: prefer bots ≤ 1800 ELO (skip Stockfish-tier)
        t_limit = time_limit if time_limit is not None else self.cfg.get("matchmaker_time", 10)
        tc_key = "classical" if t_limit > 8 else "blitz" if t_limit > 3 else "bullet"
        try:
            preferred: list[dict] = []
            fallback: list[dict] = []
            for bot in self.client.get_online_bots():
                bid = bot.get("id", "")
                if not bid or bid in exclude:
                    continue
                perfs = bot.get("perfs", {})
                bot_elo = perfs.get(tc_key, {}).get("rating", None)
                if bot_elo is None or bot_elo <= 1800:
                    preferred.append(bot)
                fallback.append(bot)
                if len(fallback) >= 100:
                    break
            pool = preferred if preferred else fallback
            if not pool:
                return None
            return random.choice(pool)
        except Exception as e:
            log.debug("Failed to get online bots: %s", e)
            return None

"""Account event loop: challenges, game starts, reconnects."""
import logging
import threading
import time
from typing import Optional

from ouroboros.lichess.client import LichessClient
from ouroboros.engine.network import OuroborosNet

log = logging.getLogger(__name__)


def _check_revenge(opp_username: str) -> bool:
    """Return True if the opponent has beaten us more than we've beaten them."""
    try:
        from ouroboros.persistence import get_db
        with get_db() as conn:
            row = conn.execute(
                "SELECT wins_vs_us, losses_vs_us FROM opponents WHERE username=?",
                (opp_username,),
            ).fetchone()
            if row:
                # wins_vs_us = opponent's wins against us (our losses)
                # losses_vs_us = opponent's losses against us (our wins)
                return int(row["wins_vs_us"]) > int(row["losses_vs_us"])
    except Exception:
        pass
    return False


def _push_record_to_viewer() -> None:
    """Query cumulative results from the DB and push them to the web viewer."""
    try:
        from ouroboros.persistence import get_db
        from ouroboros.web_viewer import update_record
        with get_db() as conn:
            row = conn.execute(
                "SELECT "
                "SUM(CASE WHEN result='win'  THEN 1 ELSE 0 END) AS wins, "
                "SUM(CASE WHEN result='loss' THEN 1 ELSE 0 END) AS losses, "
                "SUM(CASE WHEN result='draw' THEN 1 ELSE 0 END) AS draws "
                "FROM games",
            ).fetchone()
            if row:
                update_record(
                    int(row["wins"]   or 0),
                    int(row["losses"] or 0),
                    int(row["draws"]  or 0),
                )
    except Exception:
        pass


def _should_accept(challenge: dict, cfg: dict) -> tuple[bool, str]:
    variant = challenge.get("variant", {}).get("key", "")
    if variant != "standard":
        return False, "variant"

    speed = challenge.get("speed", "")
    time_control = challenge.get("timeControl", {})
    limit = time_control.get("limit", 300)
    increment = time_control.get("increment", 0)

    if speed == "bullet" and not cfg.get("accept_bullet", False):
        if cfg.get("device", "cpu") == "cpu":
            return False, "tooFast"

    accept_speeds = set()
    if cfg.get("accept_blitz", True):
        accept_speeds.update(["blitz"])
    if cfg.get("accept_rapid", True):
        accept_speeds.update(["rapid"])
    if cfg.get("accept_classical", True):
        accept_speeds.update(["classical"])
    if cfg.get("accept_bullet", False):
        accept_speeds.update(["bullet"])

    if speed not in accept_speeds and speed != "correspondence":
        return False, "tooFast"

    rated = challenge.get("rated", False)
    if rated and not cfg.get("accept_rated", True):
        return False, "casual"
    if not rated and not cfg.get("accept_casual", True):
        return False, "rated"

    return True, ""


class EventLoop:
    def __init__(
        self,
        client: LichessClient,
        net: OuroborosNet,
        device: str,
        cfg: dict,
        on_game_start=None,
        on_game_finish=None,
    ):
        self.client = client
        self.net = net
        self.device = device
        self.cfg = cfg
        self.on_game_start = on_game_start
        self.on_game_finish = on_game_finish
        self._active_games: dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()

    def run(self) -> None:
        """Main event loop. Blocks until stop() is called."""
        log.info("Starting Lichess event loop")

        # Re-attach any ongoing games from a previous session
        self._reattach_ongoing()

        for event in self.client.stream("/api/stream/event"):
            if self._stop_event.is_set():
                break
            etype = event.get("type", "")

            try:
                if etype == "challenge":
                    self._handle_challenge(event.get("challenge", {}))
                elif etype == "gameStart":
                    self._handle_game_start(event.get("game", {}))
                elif etype == "gameFinish":
                    game_id = event.get("game", {}).get("gameId", "")
                    log.info("Game finished: %s", game_id)
            except Exception as e:
                log.exception("Unhandled error processing event %r: %s", etype, e)

    def _reattach_ongoing(self) -> None:
        ongoing = self.client.get_ongoing_games()
        for game in ongoing:
            game_id = game.get("gameId", "")
            if game_id and game_id not in self._active_games:
                log.info("Reattaching to ongoing game %s", game_id)
                self._start_game_thread(game_id, game.get("color", "white"), game)

    def _handle_challenge(self, challenge: dict) -> None:
        challenge_id = challenge.get("id", "")
        challenger = challenge.get("challenger", {}).get("name", "?")
        max_concurrent = self.cfg.get("max_concurrent_games", 1)

        if len(self._active_games) >= max_concurrent:
            try:
                self.client.decline_challenge(challenge_id, "later")
            except Exception:
                pass
            log.info("Declined challenge from %s (at max concurrent games)", challenger)
            return

        # Decline during self-play-only hours
        try:
            from ouroboros.scheduler import is_lichess_active
            if not is_lichess_active():
                try:
                    self.client.decline_challenge(challenge_id, "later")
                except Exception:
                    pass
                log.info("Declined challenge from %s (self-play mode)", challenger)
                return
        except Exception:
            pass

        accept, reason = _should_accept(challenge, self.cfg)
        if accept:
            log.info("Accepting challenge from %s (%s)", challenger, challenge_id)
            try:
                self.client.accept_challenge(challenge_id)
            except Exception as e:
                log.warning("Failed to accept challenge %s: %s", challenge_id, e)
        else:
            log.info("Declining challenge from %s: %s", challenger, reason)
            try:
                self.client.decline_challenge(challenge_id, reason)
            except Exception as e:
                log.warning("Failed to decline challenge %s: %s", challenge_id, e)

    def _handle_game_start(self, game: dict) -> None:
        game_id = game.get("gameId", "")
        color = game.get("color", "white")
        if game_id in self._active_games:
            return
        self._start_game_thread(game_id, color, game)

    def _start_game_thread(self, game_id: str, color: str, game_info: dict) -> None:
        t = threading.Thread(
            target=self._run_game,
            args=(game_id, color, game_info),
            daemon=True,
        )
        self._active_games[game_id] = t
        t.start()

        if self.on_game_start:
            try:
                self.on_game_start(game_id)
            except Exception as e:
                log.exception("on_game_start error for %s: %s", game_id, e)

    def _run_game(self, game_id: str, color: str, game_info: dict) -> None:
        from ouroboros.lichess.game import GameRunner
        from ouroboros.opponents.adapt import build_context
        from ouroboros.opponents.antibot import AntiBotController, get_determinism
        from ouroboros.opponents.profiles import get_or_create_opponent

        # Extracted outside try so on_game_finish always has them
        opponent = game_info.get("opponent", {})
        opp_username = opponent.get("username", "unknown")
        opp_elo = opponent.get("rating", 1500)
        opp_title = opponent.get("title", "") or ""
        opp_is_bot = opp_title == "BOT"

        result = None
        try:
            opp_id = get_or_create_opponent(opp_username, opp_is_bot, opp_title, opp_elo)

            # Check if this is a revenge match (opponent has beaten us more than we've beaten them)
            is_revenge = _check_revenge(opp_username)

            context = build_context(opp_username, opp_is_bot, opp_elo, self.cfg)

            # Push matchup details to the web viewer
            try:
                from ouroboros.web_viewer import update_game_details
                update_game_details(game_id, opp_username, color, is_revenge)
            except Exception:
                pass

            antibot = None
            if opp_is_bot:
                det = get_determinism(opp_id)
                antibot = AntiBotController(opp_id, det, self.cfg)

            runner = GameRunner(
                client=self.client,
                net=self.net,
                device=self.device,
                cfg=self.cfg,
                game_id=game_id,
                our_color=color,
                context=context,
                antibot=antibot,
            )
            result = runner.run()
            log.info("Game %s finished: %s", game_id, result)

        except Exception as e:
            log.exception("Error in game %s: %s", game_id, e)
        finally:
            self._active_games.pop(game_id, None)
            # Always notify finish so matchmaker/throttle are unblocked even on crash
            if self.on_game_finish:
                try:
                    self.on_game_finish(game_id, result, opp_username, opp_elo, opp_is_bot, color)
                except Exception as e:
                    log.exception("on_game_finish error for %s: %s", game_id, e)
            _push_record_to_viewer()

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def active_game_count(self) -> int:
        return len(self._active_games)

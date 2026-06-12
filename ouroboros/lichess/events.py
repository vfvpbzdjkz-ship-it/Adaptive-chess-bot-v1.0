"""Account event loop: challenges, game starts, reconnects."""
import logging
import threading
import time
from typing import Optional

from ouroboros.lichess.client import LichessClient
from ouroboros.engine.network import OuroborosNet

log = logging.getLogger(__name__)


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

            if etype == "challenge":
                self._handle_challenge(event.get("challenge", {}))
            elif etype == "gameStart":
                self._handle_game_start(event.get("game", {}))
            elif etype == "gameFinish":
                game_id = event.get("game", {}).get("gameId", "")
                log.info("Game finished: %s", game_id)

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
            self.client.decline_challenge(challenge_id, "later")
            log.info("Declined challenge from %s (at max concurrent games)", challenger)
            return

        accept, reason = _should_accept(challenge, self.cfg)
        if accept:
            log.info("Accepting challenge from %s (%s)", challenger, challenge_id)
            self.client.accept_challenge(challenge_id)
        else:
            log.info("Declining challenge from %s: %s", challenger, reason)
            self.client.decline_challenge(challenge_id, reason)

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
            self.on_game_start(game_id)

    def _run_game(self, game_id: str, color: str, game_info: dict) -> None:
        from ouroboros.lichess.game import GameRunner
        from ouroboros.opponents.adapt import build_context
        from ouroboros.opponents.antibot import AntiBotController, get_determinism
        from ouroboros.opponents.profiles import get_or_create_opponent

        try:
            # Build opponent context
            opponent = game_info.get("opponent", {})
            opp_username = opponent.get("username", "unknown")
            opp_elo = opponent.get("rating", 1500)
            opp_title = opponent.get("title", "") or ""
            opp_is_bot = opp_title == "BOT"

            opp_id = get_or_create_opponent(opp_username, opp_is_bot, opp_title, opp_elo)

            context = build_context(opp_username, opp_is_bot, opp_elo, self.cfg)

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

            if self.on_game_finish:
                self.on_game_finish(game_id, result, opp_username, opp_elo, opp_is_bot)

        except Exception as e:
            log.exception("Error in game %s: %s", game_id, e)
        finally:
            self._active_games.pop(game_id, None)

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def active_game_count(self) -> int:
        return len(self._active_games)

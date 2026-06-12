"""Raw HTTP + ndjson streaming client for Lichess API."""
import json
import logging
import time
from typing import Generator, Optional

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://lichess.org"
STREAM_TIMEOUT = 60  # seconds between keepalives before reconnect


class LichessClient:
    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "User-Agent": "Ouroboros-Bot/1.0",
        })

    def get(self, path: str, **kwargs) -> dict:
        url = BASE_URL + path
        resp = self._request("GET", url, **kwargs)
        return resp.json()

    def post(self, path: str, **kwargs) -> Optional[dict]:
        url = BASE_URL + path
        resp = self._request("POST", url, **kwargs)
        if resp.content:
            try:
                return resp.json()
            except Exception:
                return None
        return None

    def _request(self, method: str, url: str, retry: int = 3, **kwargs) -> requests.Response:
        backoff = 1.0
        last_exc = None
        for attempt in range(retry):
            try:
                resp = self.session.request(method, url, timeout=15, **kwargs)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", "60"))
                    log.warning("Rate limited; waiting %ds", retry_after)
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                last_exc = e
                if attempt < retry - 1:
                    log.warning("Request failed (%s), retrying in %.1fs", e, backoff)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
        raise last_exc

    def stream(self, path: str, **kwargs) -> Generator[dict, None, None]:
        """Yield parsed ndjson objects from a streaming endpoint."""
        url = BASE_URL + path
        backoff = 1.0
        while True:
            try:
                with self.session.get(url, stream=True, timeout=STREAM_TIMEOUT, **kwargs) as resp:
                    resp.raise_for_status()
                    backoff = 1.0
                    for line in resp.iter_lines():
                        if line:
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                log.debug("Non-JSON line: %r", line)
            except requests.RequestException as e:
                log.warning("Stream error (%s); reconnecting in %.1fs", e, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def get_account(self) -> dict:
        return self.get("/api/account")

    def upgrade_to_bot(self) -> bool:
        try:
            self.post("/api/bot/account/upgrade")
            return True
        except requests.HTTPError as e:
            log.error("BOT upgrade failed: %s", e)
            return False

    def accept_challenge(self, challenge_id: str) -> None:
        self.post(f"/api/challenge/{challenge_id}/accept")

    def decline_challenge(self, challenge_id: str, reason: str = "generic") -> None:
        self.post(f"/api/challenge/{challenge_id}/decline", json={"reason": reason})

    def send_move(self, game_id: str, uci: str) -> bool:
        try:
            self.post(f"/api/bot/game/{game_id}/move/{uci}")
            return True
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code >= 500:
                log.warning("5xx on move %s game %s; retrying once", uci, game_id)
                try:
                    self.post(f"/api/bot/game/{game_id}/move/{uci}")
                    return True
                except Exception:
                    pass
            log.error("Failed to send move %s: %s", uci, e)
            return False

    def chat(self, game_id: str, room: str, text: str) -> None:
        try:
            self.post(f"/api/bot/game/{game_id}/chat", data={"room": room, "text": text})
        except Exception as e:
            log.debug("Chat failed: %s", e)

    def abort_game(self, game_id: str) -> None:
        try:
            self.post(f"/api/bot/game/{game_id}/abort")
        except Exception:
            pass

    def resign_game(self, game_id: str) -> None:
        try:
            self.post(f"/api/bot/game/{game_id}/resign")
        except Exception:
            pass

    def get_ongoing_games(self) -> list[dict]:
        try:
            data = self.get("/api/account/playing")
            return data.get("nowPlaying", [])
        except Exception:
            return []

    def get_online_bots(self) -> Generator[dict, None, None]:
        yield from self.stream("/api/bot/online")

    def challenge_player(self, username: str, time_limit: int, increment: int, rated: bool = False) -> Optional[dict]:
        try:
            return self.post("/api/challenge/" + username, json={
                "rated": rated,
                "clock.limit": time_limit * 60,
                "clock.increment": increment,
                "color": "random",
            })
        except Exception as e:
            log.debug("Challenge to %s failed: %s", username, e)
            return None

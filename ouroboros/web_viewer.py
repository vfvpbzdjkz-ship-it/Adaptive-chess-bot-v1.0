"""Minimal HTTP spectator page — no external dependencies.

Serves a single HTML page that embeds the current Lichess game via
the official Lichess embed iframe. Polls /api/state every 4 seconds
to pick up new game IDs without a page reload.

Listens on $PORT (Railway sets this automatically) or 8080.
"""
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

log = logging.getLogger(__name__)

_lock = threading.Lock()
_state: dict = {"game_id": None, "bot_username": "OUROBOROS"}

_HTML = b"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>OUROBOROS &mdash; Live</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{background:#1a1a2e;color:#e0e0e0;font-family:'Segoe UI',sans-serif;
         min-height:100vh;display:flex;flex-direction:column;align-items:center;
         padding:28px 16px}
    h1{color:#c89b3c;font-size:2rem;letter-spacing:3px;margin-bottom:4px}
    #sub{color:#666;font-size:.85rem;margin-bottom:28px}
    #board{width:600px;max-width:100%}
    iframe{width:100%;border:none;border-radius:10px;display:block}
    #note{text-align:center;margin-top:10px;font-size:.82rem}
    #idle{text-align:center;padding:60px 20px}
    .pulse{animation:p 2s ease-in-out infinite}
    @keyframes p{0%,100%{opacity:1}50%{opacity:.3}}
    a{color:#c89b3c;text-decoration:none}
    a:hover{text-decoration:underline}
  </style>
</head>
<body>
  <h1>&#9820; OUROBOROS</h1>
  <p id="sub">self-learning chess bot &mdash; live spectator</p>
  <div id="board">
    <div id="idle">
      <p class="pulse" style="margin-bottom:16px">Waiting for a game&hellip;</p>
      <p><a id="tv" href="#" target="_blank">Watch on Lichess TV</a></p>
    </div>
  </div>
  <script>
    let cur = null;
    function tick() {
      fetch('/api/state').then(r => r.json()).then(d => {
        const u = d.bot_username || 'OUROBOROS_MODEL_1';
        document.getElementById('tv') && (document.getElementById('tv').href =
          'https://lichess.org/@/' + u + '/tv');
        const b = document.getElementById('board');
        if (d.game_id) {
          if (d.game_id !== cur) {
            cur = d.game_id;
            b.innerHTML =
              '<iframe src="https://lichess.org/embed/game/' + cur +
              '?theme=brown&bg=dark" height="397" allowtransparency="true"></iframe>' +
              '<p id="note"><a href="https://lichess.org/' + cur +
              '" target="_blank">Open full game on Lichess &#x2197;</a></p>';
          }
        } else if (!d.game_id && cur !== null) {
          cur = null;
          b.innerHTML =
            '<div id="idle"><p class="pulse" style="margin-bottom:16px">Waiting for a game&hellip;</p>' +
            '<p><a href="https://lichess.org/@/' + u + '/tv" target="_blank">Watch on Lichess TV</a></p></div>';
        }
      }).catch(() => {});
    }
    tick();
    setInterval(tick, 4000);
  </script>
</body>
</html>"""


def update_game(game_id=None) -> None:
    with _lock:
        _state["game_id"] = game_id


def set_username(username: str) -> None:
    with _lock:
        _state["bot_username"] = username


def _snapshot() -> bytes:
    with _lock:
        return json.dumps(_state).encode()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # suppress per-request logs

    def do_GET(self):
        if self.path.startswith("/api/state"):
            body = _snapshot()
            self._send(200, "application/json", body)
        elif self.path in ("/health", "/healthz"):
            self._send(200, "text/plain", b"ok")
        else:
            self._send(200, "text/html; charset=utf-8", _HTML)

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class WebViewer:
    def __init__(self) -> None:
        self.port = int(os.environ.get("PORT", 8080))
        self._server: HTTPServer | None = None

    def start(self) -> None:
        self._server = HTTPServer(("0.0.0.0", self.port), _Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        log.info("Spectator page at http://0.0.0.0:%d", self.port)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()

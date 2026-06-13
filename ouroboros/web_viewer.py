"""Minimal HTTP spectator page -- no external dependencies.

Serves a single HTML page with:
- Live chess board rendered from FEN (polls our own API, always current)
- Player usernames + colors shown during active game
- Revenge vs random match label
- Countdown to the next bot challenge
- Running win/loss/draw record

Listens on $PORT (Railway injects this) or 8080.
"""
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

log = logging.getLogger(__name__)

_STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

_lock = threading.Lock()
_state: dict = {
    "game_id": None,
    "last_game_id": None,
    "bot_username": "OUROBOROS",
    "next_challenge_at": None,
    # Per-game details (cleared when game ends)
    "opponent_username": None,
    "our_color": None,          # "white" or "black"
    "is_revenge": False,
    # Board position (FEN) -- updated after every move
    "current_fen": _STARTING_FEN,
    # Cumulative record
    "total_wins": 0,
    "total_losses": 0,
    "total_draws": 0,
}

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
         padding:28px 16px;gap:0}
    h1{color:#c89b3c;font-size:2rem;letter-spacing:3px;margin-bottom:4px}
    #sub{color:#666;font-size:.85rem;margin-bottom:18px}
    #wrap{width:820px;max-width:96%}
    #badge-row{min-height:24px;margin-bottom:6px;display:flex;align-items:center;gap:8px}
    .badge{display:inline-block;padding:3px 11px;border-radius:12px;
           font-size:.73rem;font-weight:700;letter-spacing:1px}
    .live{background:#1b4a1b;color:#6fcf6f}
    .ended{background:#383838;color:#999}
    .revenge{background:#4a1b1b;color:#cf6f6f;font-size:.72rem}
    #matchup{font-size:.82rem;color:#bbb;margin-bottom:8px;
             display:flex;gap:6px;align-items:center;min-height:18px}
    .side{display:inline-flex;flex-direction:column;align-items:center;gap:2px}
    .piece{font-size:1.1rem}
    .uname{max-width:160px;overflow:hidden;text-overflow:ellipsis;
           white-space:nowrap;font-weight:600;color:#e0e0e0}
    .vs{color:#555;font-size:.75rem;padding:0 4px}
    /* Chess board */
    .chessboard{display:grid;grid-template-columns:repeat(8,1fr);
                width:min(640px,100%);margin:0 auto;
                border-radius:6px;overflow:hidden;
                box-shadow:0 4px 24px rgba(0,0,0,.6)}
    .sq{aspect-ratio:1;display:flex;align-items:center;justify-content:center}
    .sql{background:#f0d9b5}.sqd{background:#b58863}
    .pw,.pb{font-size:min(64px,10.5vw);line-height:1;user-select:none}
    .pw{color:#fff;text-shadow:0 0 4px #000,0 0 4px #000}
    .pb{color:#111}
    #idle{text-align:center;padding:60px 20px}
    .pulse{animation:p 2s ease-in-out infinite}
    @keyframes p{0%,100%{opacity:1}50%{opacity:.3}}
    #links{margin-top:9px;display:flex;gap:14px;font-size:.82rem;
           align-items:center;flex-wrap:wrap;min-height:22px}
    #footer{margin-top:22px;text-align:center;font-size:.83rem;
            color:#888;display:flex;flex-direction:column;gap:5px;
            width:820px;max-width:96%}
    #record{color:#bbb;font-size:.85rem}
    .rw{color:#6fcf6f}.rl{color:#cf6f6f}.rd{color:#aaa}
    #countdown{color:#c89b3c;font-weight:600;font-size:.95rem}
    a{color:#c89b3c;text-decoration:none}
    a:hover{text-decoration:underline}
  </style>
</head>
<body>
  <h1>&#9820; OUROBOROS</h1>
  <p id="sub">self-learning chess bot &mdash; live spectator</p>
  <div id="wrap">
    <div id="badge-row"></div>
    <div id="matchup"></div>
    <div id="board">
      <div id="idle"><p class="pulse">Waiting for a game&hellip;</p></div>
    </div>
    <div id="links"></div>
  </div>
  <div id="footer">
    <div id="record"></div>
    <div id="countdown"></div>
    <div><a id="tv" href="#" target="_blank">Watch on Lichess TV</a></div>
  </div>
  <script>
    /* Unicode chess pieces (white / black) */
    var WP = {
      'K':'\\u2654','Q':'\\u2655','R':'\\u2656','B':'\\u2657','N':'\\u2658','P':'\\u2659'
    };
    var BP = {
      'k':'\\u265a','q':'\\u265b','r':'\\u265c','b':'\\u265d','n':'\\u265e','p':'\\u265f'
    };

    var shownFen   = null;
    var shownColor = null;
    var nextChalAt = null;
    var lastState  = null;

    function renderBoard(fen, color) {
      var pos = (fen || 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR').split(' ')[0];
      var rows = pos.split('/');
      var flipped = (color === 'black');

      /* Expand each FEN rank to exactly 8 squares */
      var grid = rows.map(function(row) {
        var sq = [];
        for (var i = 0; i < row.length; i++) {
          var c = row[i];
          if (c >= '1' && c <= '8') { for (var j = 0; j < +c; j++) sq.push(''); }
          else sq.push(c);
        }
        return sq;
      });

      if (flipped) {
        grid.reverse();
        grid = grid.map(function(r) { return r.slice().reverse(); });
      }

      var html = '<div class="chessboard">';
      for (var r = 0; r < 8; r++) {
        for (var c = 0; c < 8; c++) {
          var rank = flipped ? r + 1 : 8 - r;
          var file = flipped ? 7 - c : c;
          var light = (rank + file) % 2 === 0;
          var p = grid[r][c];
          html += '<div class="sq ' + (light ? 'sql' : 'sqd') + '">';
          if (p) {
            var isW = (p === p.toUpperCase());
            html += '<span class="' + (isW ? 'pw' : 'pb') + '">' +
                    (isW ? (WP[p]||'') : (BP[p.toLowerCase()]||'')) + '</span>';
          }
          html += '</div>';
        }
      }
      return html + '</div>';
    }

    function showBoard(fen, color) {
      if (fen === shownFen && color === shownColor) return;
      shownFen = fen; shownColor = color;
      document.getElementById('board').innerHTML = renderBoard(fen, color);
    }

    function showIdle() {
      shownFen = null; shownColor = null;
      document.getElementById('board').innerHTML =
        '<div id="idle"><p class="pulse">Waiting for a game&hellip;</p></div>';
    }

    function escHtml(s) {
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')
                      .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }

    /* Countdown -- runs every second */
    setInterval(function() {
      var el = document.getElementById('countdown');
      if (!el) return;
      if (lastState && lastState.game_id) { el.textContent = ''; return; }
      if (!nextChalAt) { el.textContent = ''; return; }
      var rem = Math.max(0, Math.round(nextChalAt - Date.now() / 1000));
      if (rem === 0) { el.textContent = 'Challenge sent!'; return; }
      var m = Math.floor(rem / 60), s = rem % 60;
      el.textContent = 'Next challenge in: ' + m + ':' + (s < 10 ? '0' : '') + s;
    }, 1000);

    function render(d) {
      var liveId = d.game_id      || null;
      var lastId = d.last_game_id || null;
      var u      = d.bot_username || 'OUROBOROS';
      var opp    = d.opponent_username || null;
      var ourCol = d.our_color || null;
      var fen    = d.current_fen || null;

      var tv = document.getElementById('tv');
      if (tv) tv.href = 'https://lichess.org/@/' + u + '/tv';

      if (d.next_challenge_at) nextChalAt = d.next_challenge_at;

      /* Board: always show current FEN from our API */
      if (liveId)      showBoard(fen, ourCol);
      else if (lastId) showBoard(fen, null);
      else             showIdle();

      /* Badge row */
      var br = document.getElementById('badge-row');
      var bHtml = '';
      if (liveId) {
        bHtml += '<span class="badge live">&#9679;&nbsp;LIVE</span>';
        if (d.is_revenge) bHtml += '<span class="badge revenge">REVENGE</span>';
      } else if (lastId) {
        bHtml += '<span class="badge ended">ENDED</span>';
      }
      br.innerHTML = bHtml;

      /* Matchup line */
      var mu = document.getElementById('matchup');
      if (liveId && opp && ourCol) {
        var white = ourCol === 'white' ? u : opp;
        var black = ourCol === 'black' ? u : opp;
        mu.innerHTML =
          '<span class="side"><span class="piece">&#9817;</span>' +
          '<span class="uname">' + escHtml(white) + '</span></span>' +
          '<span class="vs">vs</span>' +
          '<span class="side"><span class="piece">&#9823;</span>' +
          '<span class="uname">' + escHtml(black) + '</span></span>';
      } else {
        mu.innerHTML = '';
      }

      /* Links */
      var viewId = liveId || lastId;
      document.getElementById('links').innerHTML = viewId
        ? '<a href="https://lichess.org/' + viewId + '" target="_blank">Open on Lichess &#x2197;</a>'
        : '';

      /* Record */
      var rec = document.getElementById('record');
      var w = d.total_wins || 0, dr = d.total_draws || 0, l = d.total_losses || 0;
      if (w + dr + l > 0) {
        rec.innerHTML = 'Record: <span class="rw">' + w + 'W</span>' +
          ' &middot; <span class="rd">' + dr + 'D</span>' +
          ' &middot; <span class="rl">' + l + 'L</span>';
      } else {
        rec.textContent = '';
      }
    }

    function tick() {
      fetch('/api/state').then(function(r){ return r.json(); }).then(function(d) {
        lastState = d;
        render(d);
      }).catch(function(){});
    }

    tick();
    setInterval(tick, 3000);
  </script>
</body>
</html>"""


def update_game(game_id=None) -> None:
    """Called when a game starts (non-None) or ends (None)."""
    with _lock:
        if game_id is not None:
            _state["game_id"] = game_id
        else:
            if _state["game_id"] is not None:
                _state["last_game_id"] = _state["game_id"]
            _state["game_id"] = None
            _state["opponent_username"] = None
            _state["our_color"] = None
            _state["is_revenge"] = False


def update_game_details(
    game_id: str,
    opponent_username: str,
    our_color: str,
    is_revenge: bool,
) -> None:
    """Enrich the current game with matchup details."""
    with _lock:
        _state["game_id"] = game_id
        _state["opponent_username"] = opponent_username
        _state["our_color"] = our_color
        _state["is_revenge"] = is_revenge


def update_fen(fen: str) -> None:
    """Update the current board position after each move."""
    with _lock:
        _state["current_fen"] = fen


def set_username(username: str) -> None:
    with _lock:
        _state["bot_username"] = username


def update_next_challenge(ts: float) -> None:
    with _lock:
        _state["next_challenge_at"] = ts


def update_record(wins: int, losses: int, draws: int) -> None:
    with _lock:
        _state["total_wins"] = wins
        _state["total_losses"] = losses
        _state["total_draws"] = draws


def _snapshot() -> bytes:
    with _lock:
        return json.dumps(_state).encode()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

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
        self._server = None

    def start(self) -> None:
        self._server = HTTPServer(("0.0.0.0", self.port), _Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        log.info("Spectator page at http://0.0.0.0:%d", self.port)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()

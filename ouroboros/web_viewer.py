"""Minimal HTTP spectator page -- no external dependencies.

Serves a single HTML page with:
- Live chess board rendered from FEN (polls our own API, always current)
- Player usernames + colors shown during active game
- Revenge vs random match label
- Current play mode (Lichess / Selfplay) with countdown to next switch
- Countdown to the next bot challenge
- Running win/loss/draw record
- Training & self-play dashboard: steps, loss sparkline, buffer fill, ELO chart

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
_force_game_callback = None
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
    # Mode scheduling
    "play_mode": "lichess",     # "lichess" or "selfplay"
    "mode_switch_at": None,     # Unix timestamp of next mode switch
    # Training & self-play stats
    "train_steps": 0,
    "last_loss": None,
    "buffer_fill": 0,
    "buffer_cap": 1_000_000,
    "selfplay_games": 0,
    "ladder_elo": 1500.0,
    "loss_history": [],         # last 50 loss values for sparkline
    "elo_history": [],          # ELO after each ladder match
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
         padding:28px 16px 40px;gap:0}
    h1{color:#c89b3c;font-size:2rem;letter-spacing:3px;margin-bottom:4px}
    #sub{color:#666;font-size:.85rem;margin-bottom:10px}
    #mode-row{font-size:.82rem;margin-bottom:14px;letter-spacing:.5px;
              display:flex;gap:8px;align-items:center}
    .ml{color:#6fcf6f;font-weight:700}
    .ms{color:#c89b3c;font-weight:700}
    .msw{color:#777;font-size:.78rem}
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
    .sq-from{background:#f6f669 !important}
    .sq-to{background:#cdd26a !important}
    .piece-img{width:88%;height:88%;object-fit:contain;display:block;
               pointer-events:none;user-select:none;-webkit-user-select:none}
    @keyframes piecein{from{opacity:.2;transform:scale(.7)}to{opacity:1;transform:scale(1)}}
    .piece-new{animation:piecein .18s ease-out}
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
    /* Force-game button */
    #force-btn{background:#1e1e3a;color:#c89b3c;border:1px solid #c89b3c;
               border-radius:6px;padding:6px 14px;font-size:.75rem;font-weight:700;
               letter-spacing:1px;cursor:pointer;transition:background .15s,color .15s}
    #force-btn:hover:not(:disabled){background:#c89b3c;color:#1a1a2e}
    #force-btn:disabled{opacity:.45;cursor:default}
    /* Training panel */
    #train-panel{width:820px;max-width:96%;margin-top:22px;
                 border:1px solid #2a2a4a;border-radius:10px;
                 padding:18px 20px;background:#141428}
    .tp-hdr{color:#c89b3c;font-size:.72rem;font-weight:700;
            letter-spacing:2.5px;margin-bottom:14px}
    .tp-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;
             margin-bottom:14px}
    .tp-box{background:#1e1e3a;border-radius:6px;padding:10px 12px}
    .tp-lbl{color:#555;font-size:.67rem;letter-spacing:.5px;
            text-transform:uppercase;margin-bottom:5px}
    .tp-val{color:#e0e0e0;font-size:1.1rem;font-weight:700;letter-spacing:.5px}
    .tp-spk{margin-top:6px;line-height:0}
    .tp-divider{border:none;border-top:1px solid #2a2a4a;margin:12px 0}
    /* Buffer bar */
    .buf-lbl{color:#555;font-size:.67rem;letter-spacing:.5px;
             text-transform:uppercase;margin-bottom:6px}
    .buf-track{background:#1e1e3a;border-radius:4px;height:8px;overflow:hidden}
    .buf-fill{background:linear-gradient(90deg,#8a6a20,#c89b3c);
              height:100%;border-radius:4px;transition:width .6s ease}
    .buf-info{color:#888;font-size:.73rem;margin-top:5px}
    /* ELO row */
    #tp-elo{display:flex;align-items:flex-start;gap:22px;margin-top:14px}
    .elo-block{min-width:80px}
    .elo-lbl{color:#555;font-size:.67rem;letter-spacing:.5px;
             text-transform:uppercase;margin-bottom:5px}
    .elo-num{font-size:1.6rem;font-weight:700;color:#c89b3c;letter-spacing:1px}
    .elo-delta{font-size:.78rem;margin-top:3px;font-weight:600}
    .elo-up{color:#6fcf6f}.elo-dn{color:#cf6f6f}.elo-flat{color:#555}
    .spk-lbl{color:#555;font-size:.67rem;letter-spacing:.5px;
             text-transform:uppercase;margin-bottom:6px}
    .spk-empty{color:#555;font-size:.73rem;font-style:italic;padding-top:8px}
  </style>
</head>
<body>
  <h1>&#9820; OUROBOROS</h1>
  <p id="sub">self-learning chess bot &mdash; live spectator</p>
  <div id="mode-row"></div>
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
  <div id="train-panel">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
      <div class="tp-hdr" style="margin-bottom:0">TRAINING &amp; SELF-PLAY</div>
      <button id="force-btn" onclick="forceGame()">&#9654; Play One Game</button>
    </div>
    <div class="tp-grid" id="tp-stats"></div>
    <hr class="tp-divider">
    <div class="buf-lbl">REPLAY BUFFER</div>
    <div class="buf-track"><div class="buf-fill" id="tp-buf-fill" style="width:0%"></div></div>
    <div class="buf-info" id="tp-buf-info"></div>
    <div id="tp-elo">
      <div class="elo-block">
        <div class="elo-lbl">INTERNAL ELO</div>
        <div class="elo-num" id="tp-elo-num">1500</div>
        <div class="elo-delta" id="tp-elo-delta"></div>
      </div>
      <div style="flex:1">
        <div class="spk-lbl">ELO HISTORY</div>
        <div id="tp-elo-chart"><div class="spk-empty">Populates after first ladder match</div></div>
      </div>
    </div>
  </div>
  <script>
    /* Lichess CBurnett piece images served from CDN */
    var CDN = 'https://lichess1.org/assets/piece/cburnett/';
    var PIECES = {
      'K':'wK','Q':'wQ','R':'wR','B':'wB','N':'wN','P':'wP',
      'k':'bK','q':'bQ','r':'bR','b':'bB','n':'bN','p':'bP'
    };

    var shownFen   = null;
    var shownColor = null;
    var nextChalAt = null;
    var lastState  = null;
    var _forcePending = false;

    /* Per-second tick: challenge countdown + mode-row */
    setInterval(function() {
      var now = Date.now() / 1000;

      /* Challenge countdown (only in Lichess mode, not during a game) */
      var el = document.getElementById('countdown');
      if (el) {
        var inLichess = !lastState || (lastState.play_mode || 'lichess') === 'lichess';
        if (!inLichess || (lastState && lastState.game_id)) {
          el.textContent = '';
        } else if (!nextChalAt) {
          el.textContent = '';
        } else {
          var rem = Math.max(0, Math.round(nextChalAt - now));
          if (rem === 0) el.textContent = 'Challenge sent!';
          else {
            var m = Math.floor(rem / 60), s = rem % 60;
            el.textContent = 'Next challenge in: ' + m + ':' + (s < 10 ? '0' : '') + s;
          }
        }
      }

      /* Mode row */
      var mEl = document.getElementById('mode-row');
      if (mEl && lastState) {
        var mode = lastState.play_mode || 'lichess';
        var isL  = (mode === 'lichess');
        var badge = isL ? '<span class="ml">&#9679; LICHESS</span>'
                        : '<span class="ms">&#9651; SELFPLAY</span>';
        var sw = lastState.mode_switch_at || 0;
        var swStr = '';
        if (sw > 0) {
          var rem2 = Math.max(0, Math.round(sw - now));
          var m2 = Math.floor(rem2 / 60), s2 = rem2 % 60;
          swStr = '<span class="msw">&middot; switch in ' +
                  m2 + ':' + (s2 < 10 ? '0' : '') + s2 + '</span>';
        }
        mEl.innerHTML = badge + (swStr ? ' ' + swStr : '');
      }
    }, 1000);

    /* Parse FEN position into 8x8 grid (row 0 = rank 8, row 7 = rank 1) */
    function fenToGrid(fen) {
      var pos = (fen || '').split(' ')[0];
      var rows = pos.split('/');
      var grid = [];
      for (var r = 0; r < rows.length; r++) {
        var row = [];
        for (var i = 0; i < rows[r].length; i++) {
          var ch = rows[r][i];
          if (ch >= '1' && ch <= '8') { for (var j = 0; j < +ch; j++) row.push(''); }
          else row.push(ch);
        }
        grid.push(row);
      }
      return grid;
    }

    /* Diff two FENs; returns {from:[], to:[]} in FEN grid coords */
    function findChangedSqs(fen1, fen2) {
      if (!fen1 || !fen2) return {from:[], to:[]};
      var g1 = fenToGrid(fen1), g2 = fenToGrid(fen2);
      var from = [], to = [];
      for (var r = 0; r < 8; r++) {
        for (var c = 0; c < 8; c++) {
          var p1 = (g1[r] && g1[r][c]) || '';
          var p2 = (g2[r] && g2[r][c]) || '';
          if (p1 !== '' && p2 === '') from.push(r + ',' + c);
          else if (p1 !== p2 && p2 !== '') to.push(r + ',' + c);
        }
      }
      return {from: from, to: to};
    }

    function renderBoard(fen, color, fromSqs, toSqs) {
      var pos = (fen || 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR').split(' ')[0];
      var rows = pos.split('/');
      var flipped = (color === 'black');

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

      var fromSet = {}, toSet = {};
      (fromSqs || []).forEach(function(k){ fromSet[k] = 1; });
      (toSqs   || []).forEach(function(k){ toSet[k]   = 1; });

      var html = '<div class="chessboard">';
      for (var r = 0; r < 8; r++) {
        for (var c = 0; c < 8; c++) {
          var rank = flipped ? r + 1 : 8 - r;
          var file = flipped ? 7 - c : c;
          var light = (rank + file) % 2 === 0;
          /* Map visual (r,c) back to FEN grid coords for highlight lookup */
          var fenRow = flipped ? 7 - r : r;
          var fenCol = flipped ? 7 - c : c;
          var sqKey  = fenRow + ',' + fenCol;
          var hlCls  = fromSet[sqKey] ? ' sq-from' : (toSet[sqKey] ? ' sq-to' : '');
          var isNew  = !!toSet[sqKey];
          var p = grid[r][c];
          html += '<div class="sq ' + (light ? 'sql' : 'sqd') + hlCls + '">';
          if (p) {
            var pn = PIECES[p] || '';
            if (pn) {
              html += '<img class="piece-img' + (isNew ? ' piece-new' : '') +
                      '" src="' + CDN + pn + '.svg" draggable="false" alt="' + p + '">';
            }
          }
          html += '</div>';
        }
      }
      return html + '</div>';
    }

    function showBoard(fen, color) {
      if (fen === shownFen && color === shownColor) return;
      var chg = findChangedSqs(shownFen, fen);
      shownFen = fen; shownColor = color;
      document.getElementById('board').innerHTML = renderBoard(fen, color, chg.from, chg.to);
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

    function fmtNum(n) {
      return (n || 0).toLocaleString();
    }

    /* SVG sparkline -- returns empty string if fewer than 2 points */
    function sparkline(vals, w, h, col) {
      if (!vals || vals.length < 2) return '';
      var mn = Math.min.apply(null, vals);
      var mx = Math.max.apply(null, vals);
      var rng = mx - mn || 1;
      var pts = vals.map(function(v, i) {
        var x = (i / (vals.length - 1) * w).toFixed(1);
        var y = ((1 - (v - mn) / rng) * h).toFixed(1);
        return x + ',' + y;
      }).join(' ');
      var lx = w, ly = ((1-(vals[vals.length-1]-mn)/rng)*h).toFixed(1);
      return '<svg width="'+w+'" height="'+h+'" viewBox="0 0 '+w+' '+h+'" style="display:block;overflow:visible">' +
        '<polyline points="'+pts+'" fill="none" stroke="'+col+'" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>' +
        '<circle cx="'+lx+'" cy="'+ly+'" r="2.5" fill="'+col+'"/>' +
        '</svg>';
    }

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

      /* Board */
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

      /* Force-game button -- only update when no pending challenge in flight */
      var btn = document.getElementById('force-btn');
      if (btn && !_forcePending) {
        if (liveId) {
          btn.disabled = true;
          btn.innerHTML = '&#9632; In Game';
        } else {
          btn.disabled = false;
          btn.innerHTML = '&#9654; Play One Game';
        }
      }

      /* ---- Training panel ---- */
      var steps   = d.train_steps    || 0;
      var sp      = d.selfplay_games || 0;
      var loss    = d.last_loss;
      var lossH   = d.loss_history   || [];
      var bf      = d.buffer_fill    || 0;
      var bc      = d.buffer_cap     || 1000000;
      var elo     = d.ladder_elo     || 1500;
      var eloH    = d.elo_history    || [];

      /* Stat boxes */
      var lossSpk = lossH.length >= 3
        ? '<div class="tp-spk">' + sparkline(lossH, 110, 24, '#888') + '</div>'
        : '';
      var lossStr = (loss !== null && loss !== undefined) ? loss.toFixed(4) : '&mdash;';
      var ts = document.getElementById('tp-stats');
      if (ts) ts.innerHTML =
        '<div class="tp-box"><div class="tp-lbl">Self-Play Games</div>' +
          '<div class="tp-val">' + fmtNum(sp) + '</div></div>' +
        '<div class="tp-box"><div class="tp-lbl">Training Steps</div>' +
          '<div class="tp-val">' + fmtNum(steps) + '</div></div>' +
        '<div class="tp-box"><div class="tp-lbl">Loss</div>' +
          '<div class="tp-val">' + lossStr + '</div>' + lossSpk + '</div>';

      /* Buffer bar */
      var pct = Math.min(100, bc > 0 ? Math.round(bf / bc * 100) : 0);
      var bfEl = document.getElementById('tp-buf-fill');
      if (bfEl) bfEl.style.width = pct + '%';
      var biEl = document.getElementById('tp-buf-info');
      if (biEl) biEl.textContent = pct + '% -- ' + fmtNum(bf) + ' / ' + fmtNum(bc) + ' positions';

      /* ELO number + delta */
      var eloEl = document.getElementById('tp-elo-num');
      if (eloEl) eloEl.textContent = Math.round(elo);
      var deltaEl = document.getElementById('tp-elo-delta');
      if (deltaEl && eloH.length >= 2) {
        var delta = elo - eloH[eloH.length - 2];
        var dClass = delta > 0.5 ? 'elo-up' : (delta < -0.5 ? 'elo-dn' : 'elo-flat');
        var dStr = delta > 0 ? '+' + delta.toFixed(0) : delta.toFixed(0);
        deltaEl.innerHTML = '<span class="' + dClass + '">' + dStr + ' from last match</span>';
      } else if (deltaEl) {
        deltaEl.textContent = '';
      }

      /* ELO chart */
      var chartEl = document.getElementById('tp-elo-chart');
      if (chartEl) {
        if (eloH.length >= 2) {
          chartEl.innerHTML = sparkline(eloH, 400, 60, '#c89b3c');
        } else {
          chartEl.innerHTML = '<div class="spk-empty">Populates after first ladder match</div>';
        }
      }
    }

    function forceGame() {
      if (_forcePending) return;
      _forcePending = true;
      var btn = document.getElementById('force-btn');
      btn.disabled = true;
      btn.textContent = 'Sending\\u2026';
      fetch('/api/force-game', {method:'POST'})
        .then(function(r){ return r.json(); })
        .then(function(d) {
          btn.textContent = d.ok ? '\\u2713 Challenge sent!' : '\\u2717 Failed';
          setTimeout(function(){ _forcePending = false; }, 3000);
        }).catch(function(){
          btn.textContent = 'Error';
          setTimeout(function(){ _forcePending = false; }, 3000);
        });
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


def update_play_mode(mode: str, switch_at: float) -> None:
    """Called by PlayScheduler on every mode transition."""
    with _lock:
        _state["play_mode"] = mode
        _state["mode_switch_at"] = switch_at


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


def update_training_stats(
    steps: int,
    loss: float,
    buffer_fill: int,
    buffer_cap: int,
    selfplay_games: int,
) -> None:
    """Called after each training step to push stats to the web viewer."""
    with _lock:
        _state["train_steps"] = steps
        _state["last_loss"] = round(loss, 6)
        _state["buffer_fill"] = buffer_fill
        _state["buffer_cap"] = buffer_cap
        _state["selfplay_games"] = selfplay_games
        hist = _state["loss_history"]
        hist.append(round(loss, 6))
        if len(hist) > 50:
            del hist[:-50]


def update_elo(elo: float) -> None:
    """Called after each ladder match to record the new internal ELO."""
    with _lock:
        _state["ladder_elo"] = round(elo, 1)
        hist = _state["elo_history"]
        hist.append(round(elo, 1))
        if len(hist) > 40:
            del hist[:-40]


def set_force_game_callback(fn) -> None:
    """Register a callable invoked when the 'Play One Game' button is pressed."""
    global _force_game_callback
    _force_game_callback = fn


def load_elo_history() -> None:
    """Seed elo_history from the DB ladder table on startup."""
    try:
        from ouroboros.persistence import get_db
        with get_db() as conn:
            rows = conn.execute(
                "SELECT elo FROM ladder ORDER BY id ASC"
            ).fetchall()
        elos = [round(float(r["elo"]), 1) for r in rows]
        with _lock:
            _state["elo_history"] = elos[-40:]
            if elos:
                _state["ladder_elo"] = elos[-1]
    except Exception as e:
        log.debug("load_elo_history: %s", e)


def _snapshot() -> bytes:
    with _lock:
        return json.dumps(_state).encode()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            self.rfile.read(content_length)
        if self.path == "/api/force-game":
            cb = _force_game_callback
            if cb is not None:
                try:
                    cb()
                    self._send(200, "application/json", b'{"ok":true}')
                except Exception as e:
                    log.warning("force-game callback failed: %s", e)
                    self._send(500, "application/json", b'{"ok":false}')
            else:
                self._send(503, "application/json", b'{"ok":false,"error":"not_configured"}')
        else:
            self._send(404, "text/plain", b"not found")

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

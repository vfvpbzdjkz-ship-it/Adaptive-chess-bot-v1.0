"""SQLite opponent profile management + embedded micro-book."""
import datetime
import logging
import io
from typing import Optional

import chess
import chess.pgn

from ouroboros.persistence import get_db

log = logging.getLogger(__name__)

# ── Embedded micro-book (~40 entries) ─────────────────────────────────────────
# Used ONLY as the zero-data prior in opening steering.
# Keys: FEN position_key string (Zobrist hash as int → str).
# Value: {uci_move: weight}
# We use polyglot board hash via board._transposition_key().

MICRO_BOOK: dict[str, dict[str, float]] = {
    # Starting position
    "opening_start_white": {
        "e2e4": 0.35, "d2d4": 0.35, "c2c4": 0.15, "g1f3": 0.15
    },
    # After 1.e4
    "after_e4_black": {
        "e7e5": 0.30, "c7c5": 0.25, "e7e6": 0.15, "c7c6": 0.15, "d7d5": 0.10, "g8f6": 0.05
    },
    # After 1.d4
    "after_d4_black": {
        "d7d5": 0.30, "g8f6": 0.35, "e7e6": 0.15, "c7c5": 0.15, "f7f5": 0.05
    },
    # After 1.c4
    "after_c4_black": {
        "e7e5": 0.30, "c7c5": 0.25, "g8f6": 0.30, "e7e6": 0.15
    },
    # After 1.Nf3
    "after_nf3_black": {
        "d7d5": 0.30, "g8f6": 0.35, "c7c5": 0.20, "e7e6": 0.15
    },
}


def _pos_key(board: chess.Board) -> str:
    return str(board._transposition_key())


def get_micro_book_priors(board: chess.Board) -> Optional[dict[str, float]]:
    """Return micro-book priors for current position, or None."""
    ply = board.ply()
    if ply == 0 and board.turn == chess.WHITE:
        return MICRO_BOOK.get("opening_start_white")
    # We don't store all positions; return None and fall through to general logic
    return None


def get_or_create_opponent(username: str, is_bot: bool, title: str, last_elo: int) -> int:
    """Return opponent DB id, creating row if needed."""
    ts = datetime.datetime.utcnow().isoformat()
    with get_db() as conn:
        row = conn.execute("SELECT id FROM opponents WHERE username=?", (username,)).fetchone()
        if row:
            conn.execute(
                "UPDATE opponents SET is_bot=?, title=?, last_elo=?, last_seen=? WHERE username=?",
                (int(is_bot), title, last_elo, ts, username),
            )
            return row["id"]
        else:
            conn.execute(
                "INSERT INTO opponents(username, is_bot, title, last_elo, last_seen) VALUES(?,?,?,?,?)",
                (username, int(is_bot), title, last_elo, ts),
            )
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def get_opponent(username: str) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM opponents WHERE username=?", (username,)).fetchone()
        return dict(row) if row else None


def update_opponent_after_game(
    username: str,
    is_bot: bool,
    last_elo: int,
    result: str,  # "win" | "loss" | "draw"
    pgn: str,
    our_color: str,
) -> None:
    ts = datetime.datetime.utcnow().isoformat()
    with get_db() as conn:
        row = conn.execute("SELECT * FROM opponents WHERE username=?", (username,)).fetchone()
        if not row:
            conn.execute(
                "INSERT OR IGNORE INTO opponents(username, is_bot, last_elo, last_seen) VALUES(?,?,?,?)",
                (username, int(is_bot), last_elo, ts),
            )
            row = conn.execute("SELECT * FROM opponents WHERE username=?", (username,)).fetchone()
        opp_id = row["id"]

        wins = row["wins_vs_us"] + (1 if result == "loss" else 0)
        losses = row["losses_vs_us"] + (1 if result == "win" else 0)
        draws = row["draws_vs_us"] + (1 if result == "draw" else 0)
        games = row["games"] + 1

        conn.execute(
            "UPDATE opponents SET games=?, wins_vs_us=?, losses_vs_us=?, draws_vs_us=?, last_elo=?, last_seen=? WHERE id=?",
            (games, wins, losses, draws, last_elo, ts, opp_id),
        )

    # Update opening moves from PGN
    _update_opening_moves(pgn, our_color, username, result)


def _update_opening_moves(pgn: str, our_color: str, username: str, result: str) -> None:
    """Record opponent's opening moves and our result after each of our moves."""
    our_chess_color = chess.WHITE if our_color == "white" else chess.BLACK
    if result == "win":
        our_score = 1.0
    elif result == "loss":
        our_score = 0.0
    else:
        our_score = 0.5

    try:
        game = chess.pgn.read_game(io.StringIO(pgn))
    except Exception:
        return

    board = game.board()
    node = game
    ply = 0
    ts = datetime.datetime.utcnow().isoformat()

    with get_db() as conn:
        opp_row = conn.execute("SELECT id FROM opponents WHERE username=?", (username,)).fetchone()
        if not opp_row:
            return
        opp_id = opp_row["id"]

        while node.variations and ply < 16:
            node = node.variations[0]
            move = node.move
            pkey = _pos_key(board)
            move_uci = move.uci()

            if board.turn != our_chess_color:
                # Opponent's move: track their opening habits
                existing = conn.execute(
                    "SELECT times_played FROM opening_moves WHERE opponent_id=? AND position_key=? AND move_uci=?",
                    (opp_id, pkey, move_uci),
                ).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE opening_moves SET times_played=times_played+1, last_played=? WHERE opponent_id=? AND position_key=? AND move_uci=?",
                        (ts, opp_id, pkey, move_uci),
                    )
                else:
                    conn.execute(
                        "INSERT INTO opening_moves(opponent_id, position_key, move_uci, times_played, our_score_after, last_played) VALUES(?,?,?,1,0.5,?)",
                        (opp_id, pkey, move_uci, ts),
                    )
            else:
                # Our move: update our_score_after via EMA
                alpha = 0.25
                existing = conn.execute(
                    "SELECT our_score_after FROM opening_moves WHERE opponent_id=? AND position_key=? AND move_uci=?",
                    (opp_id, pkey, move_uci),
                ).fetchone()
                if existing:
                    old_score = existing["our_score_after"]
                    new_score = (1 - alpha) * old_score + alpha * our_score
                    conn.execute(
                        "UPDATE opening_moves SET our_score_after=?, times_played=times_played+1, last_played=? WHERE opponent_id=? AND position_key=? AND move_uci=?",
                        (new_score, ts, opp_id, pkey, move_uci),
                    )
                else:
                    conn.execute(
                        "INSERT INTO opening_moves(opponent_id, position_key, move_uci, times_played, our_score_after, last_played) VALUES(?,?,?,1,?,?)",
                        (opp_id, pkey, move_uci, our_score, ts),
                    )

            # Update band opening moves
            opp_row2 = conn.execute("SELECT last_elo FROM opponents WHERE id=?", (opp_id,)).fetchone()
            if opp_row2 and not conn.execute("SELECT is_bot FROM opponents WHERE id=?", (opp_id,)).fetchone()["is_bot"]:
                elo = opp_row2["last_elo"]
                band = (elo // 100) * 100
                existing_band = conn.execute(
                    "SELECT times_played, our_score_after FROM band_opening_moves WHERE band=? AND position_key=? AND move_uci=?",
                    (band, pkey, move_uci),
                ).fetchone()
                if existing_band:
                    old_score = existing_band["our_score_after"]
                    new_score = (1 - 0.25) * old_score + 0.25 * our_score
                    conn.execute(
                        "UPDATE band_opening_moves SET times_played=times_played+1, our_score_after=?, last_played=? WHERE band=? AND position_key=? AND move_uci=?",
                        (new_score, ts, band, pkey, move_uci),
                    )
                else:
                    conn.execute(
                        "INSERT INTO band_opening_moves(band, position_key, move_uci, times_played, our_score_after, last_played) VALUES(?,?,?,1,?,?)",
                        (band, pkey, move_uci, our_score, ts),
                    )

            board.push(move)
            ply += 1


def get_opening_stats(opponent_id: int, position_key: str) -> list[dict]:
    """Get opponent's opening moves from a position."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT move_uci, times_played, our_score_after FROM opening_moves WHERE opponent_id=? AND position_key=? ORDER BY times_played DESC",
            (opponent_id, position_key),
        ).fetchall()
        return [dict(r) for r in rows]


def get_band_opening_stats(band: int, position_key: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT move_uci, times_played, our_score_after FROM band_opening_moves WHERE band=? AND position_key=? ORDER BY times_played DESC",
            (band, position_key),
        ).fetchall()
        return [dict(r) for r in rows]

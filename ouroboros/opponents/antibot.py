"""Anti-bot exploitation: determinism tracking, exploit lines, branch-after-loss."""
import datetime
import logging
from typing import Optional

import chess
import numpy as np

from ouroboros.opponents.profiles import _pos_key, get_opponent
from ouroboros.persistence import get_db

log = logging.getLogger(__name__)

DETERMINISM_EMA_ALPHA = 0.3
DETERMINISM_HIGH = 0.7
FORCED_EXPLORATION_TAU = 0.7
VALUE_DROP_THRESHOLD = -0.3


def update_determinism(opponent_id: int, position_key: str, played_move_uci: str) -> None:
    """Update determinism_score EMA: did this bot repeat a previously-seen move?"""
    with get_db() as conn:
        row = conn.execute(
            "SELECT move_uci, times_played FROM opening_moves WHERE opponent_id=? AND position_key=?",
            (opponent_id, position_key),
        ).fetchone()
        opp = conn.execute("SELECT determinism_score FROM opponents WHERE id=?", (opponent_id,)).fetchone()
        if opp is None:
            return
        old_score = opp["determinism_score"]

        if row:
            # Position was seen before
            repeated = int(row["move_uci"] == played_move_uci)
            new_score = (1 - DETERMINISM_EMA_ALPHA) * old_score + DETERMINISM_EMA_ALPHA * repeated
        else:
            new_score = old_score  # first time — no information

        conn.execute(
            "UPDATE opponents SET determinism_score=? WHERE id=?",
            (new_score, opponent_id),
        )


def get_determinism(opponent_id: int) -> float:
    with get_db() as conn:
        row = conn.execute("SELECT determinism_score FROM opponents WHERE id=?", (opponent_id,)).fetchone()
        return row["determinism_score"] if row else 0.0


def get_exploit_line(opponent_id: int) -> Optional[list[str]]:
    """Return the best still-valid winning exploit line as list of UCI strings."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT line_uci FROM exploit_lines WHERE opponent_id=? AND result>=1.0 AND still_valid=1 ORDER BY times_used DESC LIMIT 1",
            (opponent_id,),
        ).fetchone()
    if row:
        return row["line_uci"].split()
    return None


def record_exploit_line(opponent_id: int, line_ucis: list[str], result: float) -> None:
    line_str = " ".join(line_ucis)
    ts = datetime.datetime.utcnow().isoformat()
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM exploit_lines WHERE opponent_id=? AND line_uci=?",
            (opponent_id, line_str),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE exploit_lines SET times_used=times_used+1, last_used=?, result=? WHERE id=?",
                (ts, result, existing["id"]),
            )
        else:
            conn.execute(
                "INSERT INTO exploit_lines(opponent_id, line_uci, result, length, last_used, times_used, still_valid) VALUES(?,?,?,?,?,1,1)",
                (opponent_id, line_str, result, len(line_ucis), ts),
            )


def invalidate_exploit_line_at(opponent_id: int, line_ucis: list[str], diverge_ply: int) -> None:
    line_str = " ".join(line_ucis)
    with get_db() as conn:
        conn.execute(
            "UPDATE exploit_lines SET still_valid=0, diverge_ply=? WHERE opponent_id=? AND line_uci=?",
            (diverge_ply, opponent_id, line_str),
        )


class AntiBotController:
    """Per-game state machine for anti-bot exploitation."""

    def __init__(self, opponent_id: int, determinism: float, cfg: dict):
        self.opponent_id = opponent_id
        self.determinism = determinism
        self.cfg = cfg
        self.is_high_determinism = determinism >= DETERMINISM_HIGH
        self._exploit_line: Optional[list[str]] = None
        self._exploit_ply = 0
        self._game_moves: list[str] = []        # our moves as UCI
        self._opp_moves: list[str] = []         # opp moves as UCI
        self._value_history: list[float] = []   # MCTS root values at our turns
        self._lost_game_moves: Optional[list[str]] = None  # previous loss moves
        self._branch_ply: Optional[int] = None
        self._excluded_moves: set = set()
        self._forced_exploration_ply: Optional[int] = None

        if self.is_high_determinism:
            self._exploit_line = get_exploit_line(opponent_id)
            log.info("Anti-bot: high determinism (%.2f) vs %d; exploit line: %s",
                     determinism, opponent_id, self._exploit_line)

    def load_previous_loss(self, our_moves: list[str], value_trace: list[float]) -> None:
        """Load our previous loss game to find branching point."""
        if not our_moves or not value_trace:
            return
        self._lost_game_moves = our_moves
        # Find ply where value first dropped below threshold
        for ply, v in enumerate(value_trace):
            if v < VALUE_DROP_THRESHOLD:
                self._branch_ply = ply
                if ply < len(our_moves):
                    bad_move_uci = our_moves[ply]
                    self._excluded_moves = {chess.Move.from_uci(bad_move_uci)}
                    self._forced_exploration_ply = ply
                log.info("Anti-bot: will branch at ply %d (exclude %s)", ply, bad_move_uci if ply < len(our_moves) else "?")
                break

    def get_move_override(
        self,
        board: chess.Board,
        current_ply: int,
    ) -> tuple[Optional[chess.Move], bool, Optional[float]]:
        """Return (move_override, add_noise, forced_tau) for current position.

        move_override: if set, play this move book-speed
        add_noise: if True, add Dirichlet noise to root
        forced_tau: if set, use this temperature at root
        """
        if not self.is_high_determinism:
            return None, False, None

        # Follow exploit line
        if self._exploit_line and self._exploit_ply < len(self._exploit_line):
            uci = self._exploit_line[self._exploit_ply]
            try:
                move = chess.Move.from_uci(uci)
                if move in board.legal_moves:
                    self._exploit_ply += 1
                    return move, False, None
                else:
                    # Bot deviated from expected line
                    log.info("Anti-bot: opponent deviated at ply %d; invalidating exploit line", current_ply)
                    if self._exploit_line:
                        invalidate_exploit_line_at(self.opponent_id, self._exploit_line, current_ply)
                    self._exploit_line = None
            except ValueError:
                self._exploit_line = None

        # Branch-after-loss: at the branching ply, exclude the bad move and explore
        if (self._forced_exploration_ply is not None and
                current_ply == self._forced_exploration_ply and
                self._excluded_moves):
            return None, True, FORCED_EXPLORATION_TAU

        return None, False, None

    def record_our_move(self, move: chess.Move, root_value: float) -> None:
        self._game_moves.append(move.uci())
        self._value_history.append(root_value)

    def record_opponent_move(self, board_before: chess.Board, move: chess.Move) -> None:
        pkey = _pos_key(board_before)
        self._opp_moves.append(move.uci())
        update_determinism(self.opponent_id, pkey, move.uci())

        # Check if exploit line still valid
        if (self._exploit_line and
                len(self._opp_moves) <= len(self._exploit_line) and
                self._exploit_ply > 0):
            expected_idx = self._exploit_ply - 1
            if expected_idx < len(self._exploit_line):
                # Opp moves at even plies if we go first
                pass  # validation handled in get_move_override

    def on_game_end(self, result: str, board: chess.Board) -> None:
        """Called after game ends. Records exploit lines on wins."""
        if result == "win" and len(self._game_moves) > 0:
            record_exploit_line(self.opponent_id, self._game_moves, 1.0)
            log.info("Anti-bot: recorded winning exploit line (%d moves)", len(self._game_moves))

    def get_excluded_moves(self, current_ply: int) -> set:
        if current_ply == self._forced_exploration_ply:
            return self._excluded_moves
        return set()

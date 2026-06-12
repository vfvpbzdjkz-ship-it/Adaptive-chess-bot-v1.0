"""Turn an opponent profile into live play adjustments."""
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import chess
import numpy as np

from ouroboros.opponents.profiles import (
    get_opponent, get_opening_stats, get_band_opening_stats, _pos_key
)

log = logging.getLogger(__name__)

ADAPT_LAMBDA_MAX = 0.6
ADAPT_LAMBDA_ROOT_MAX = 0.3


@dataclass
class OpponentContext:
    username: str
    is_bot: bool
    opponent_id: Optional[int]
    last_elo: int
    games_vs_us: int
    confidence: float                   # λ ∈ [0, ADAPT_LAMBDA_MAX]
    opening_plies: int = 16
    book_speed_threshold: float = 0.5   # confidence for fast book play
    excluded_moves: set = field(default_factory=set)  # anti-bot exclusions
    forced_ply_tau: Optional[tuple[int, float]] = None  # (ply, tau) for exploration


def build_context(
    username: str,
    is_bot: bool,
    opponent_elo: int,
    cfg: dict,
) -> OpponentContext:
    opp = get_opponent(username)
    if opp is None:
        opp_id = None
        games = 0
    else:
        opp_id = opp["id"]
        games = opp.get("games", 0)

    # Confidence: for bots, always use individual profile; for humans, blend
    if is_bot:
        # w = games / (games + 5), but cap at lambda_max
        w = games / (games + 5) if games > 0 else 0.0
        confidence = min(w, ADAPT_LAMBDA_MAX)
    else:
        if games >= 5:
            w = games / (games + 5)
            confidence = min(w, ADAPT_LAMBDA_MAX)
        else:
            confidence = 0.0

    return OpponentContext(
        username=username,
        is_bot=is_bot,
        opponent_id=opp_id,
        last_elo=opponent_elo,
        games_vs_us=games,
        confidence=confidence,
        opening_plies=cfg.get("opening_plies", 16),
        book_speed_threshold=cfg.get("book_speed_confidence", 0.5),
    )


def get_opening_bias(
    context: OpponentContext,
    board: chess.Board,
    legal_moves: list[chess.Move],
    our_color: chess.Color,
) -> Optional[np.ndarray]:
    """Return a bias array (4672,) or None for no bias.

    Computes adjusted prior: P' ∝ P_net^(1-λ) * exp(λ * score_vs_profile)
    Returns log-space additive term to be added to log-policy.
    """
    if context.confidence <= 0.0:
        return None
    if board.ply() >= context.opening_plies * 2:
        return None

    pkey = _pos_key(board)
    lam = context.confidence

    stats: dict[str, float] = {}

    if context.opponent_id is not None:
        rows = get_opening_stats(context.opponent_id, pkey)
        for r in rows:
            stats[r["move_uci"]] = r["our_score_after"]

    # Blend with band stats for humans
    if not context.is_bot and context.games_vs_us < 5:
        band = (context.last_elo // 100) * 100
        band_rows = get_band_opening_stats(band, pkey)
        for r in band_rows:
            if r["move_uci"] not in stats:
                stats[r["move_uci"]] = r["our_score_after"]

    if not stats:
        return None

    from ouroboros.engine.encoding import move_to_index
    bias = np.zeros(4672, dtype=np.float32)
    for move in legal_moves:
        try:
            idx = move_to_index(board, move)
            score = stats.get(move.uci(), 0.5)
            bias[idx] = lam * score
        except ValueError:
            pass

    if bias.max() == bias.min():
        return None
    return bias


def get_root_bias(
    context: OpponentContext,
    board: chess.Board,
    legal_moves: list[chess.Move],
) -> Optional[np.ndarray]:
    """Lighter version of opening bias for MCTS root, capped at lambda_root_max."""
    if context.confidence <= 0.0:
        return None
    if board.ply() >= context.opening_plies * 2:
        return None

    pkey = _pos_key(board)
    lam = min(context.confidence, ADAPT_LAMBDA_ROOT_MAX)
    stats: dict[str, float] = {}

    if context.opponent_id is not None:
        rows = get_opening_stats(context.opponent_id, pkey)
        for r in rows:
            stats[r["move_uci"]] = r["our_score_after"]

    if not stats:
        return None

    from ouroboros.engine.encoding import move_to_index
    bias = np.zeros(4672, dtype=np.float32)
    for move in legal_moves:
        try:
            idx = move_to_index(board, move)
            score = stats.get(move.uci(), 0.5)
            bias[idx] = lam * (score - 0.5)  # centre around 0
        except ValueError:
            pass

    return bias if bias.any() else None


def is_book_speed(context: OpponentContext, board: chess.Board) -> bool:
    """Return True if we should move book-speed (high confidence, known line)."""
    if context.confidence < context.book_speed_threshold:
        return False
    if board.ply() >= context.opening_plies * 2:
        return False
    pkey = _pos_key(board)
    if context.opponent_id is None:
        return False
    rows = get_opening_stats(context.opponent_id, pkey)
    if not rows:
        return False
    best = max(rows, key=lambda r: r["times_played"])
    return best["times_played"] >= 3 and best["our_score_after"] >= 0.6

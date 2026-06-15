"""Convert finished Lichess games into weighted training data."""
import datetime
import logging
from typing import Optional

import chess
import chess.pgn
import io
import numpy as np

from ouroboros.engine.encoding import board_to_tensor, move_to_index
from ouroboros.learning.buffer import ReplayBuffer, SOURCE_LICHESS_LIVE, SOURCE_OPP_IMITATION
from ouroboros.persistence import get_db

log = logging.getLogger(__name__)

LIVE_WEIGHT = 4.0        # weight for our own game positions
LIVE_LOSS_WEIGHT = 2.0   # lower weight for our positions in lost games (policy target is a losing move)
IMITATION_WEIGHT = 3.0   # weight for imitating the winner's moves


def process_finished_game(
    buffer: ReplayBuffer,
    game_id: str,
    pgn: str,
    our_color: str,    # "white" or "black"
    result: str,       # "win" | "loss" | "draw"
    opponent_username: str,
    opponent_elo: int,
    opponent_is_bot: bool,
    clocks: Optional[str] = None,
) -> None:
    """Add game positions to buffer and update DB."""
    try:
        game = chess.pgn.read_game(io.StringIO(pgn))
    except Exception as e:
        log.warning("Failed to parse PGN for game %s: %s", game_id, e)
        return

    our_chess_color = chess.WHITE if our_color == "white" else chess.BLACK

    # Determine numeric result for our side
    if result == "win":
        our_result = 1.0
    elif result == "loss":
        our_result = -1.0
    else:
        our_result = 0.0

    board = game.board()
    positions = []  # (board_before, move)
    node = game
    while node.variations:
        node = node.variations[0]
        move = node.move
        positions.append((board.copy(), move))
        board.push(move)

    # Add all positions with our-color perspective z.
    # Use lower weight for our own positions in losses: the policy target (the move
    # we played) led to a loss, so reinforcing it strongly would be counterproductive.
    # The value signal (z=-1) is still valuable; we just down-weight the policy loss.
    own_weight = LIVE_LOSS_WEIGHT if result == "loss" else LIVE_WEIGHT
    for pos_board, move in positions:
        state = board_to_tensor(pos_board).numpy()
        # Policy: one-hot on the played move
        policy = np.zeros(4672, dtype=np.float32)
        try:
            idx = move_to_index(pos_board, move)
            policy[idx] = 1.0
        except ValueError:
            continue

        z = our_result if pos_board.turn == our_chess_color else -our_result
        w = own_weight if pos_board.turn == our_chess_color else LIVE_WEIGHT
        buffer.add(state, policy, z, weight=w, source=SOURCE_LICHESS_LIVE)

    # Winner imitation: if we lost, add opponent's moves
    if result == "loss":
        board = game.board()
        node = game
        while node.variations:
            node = node.variations[0]
            move = node.move
            if board.turn != our_chess_color:
                # This is opponent's move — imitate it
                state = board_to_tensor(board).numpy()
                policy = np.zeros(4672, dtype=np.float32)
                try:
                    idx = move_to_index(board, move)
                    policy[idx] = 1.0
                    z = 1.0  # z=+1 from opponent's side
                    buffer.add(state, policy, z, weight=IMITATION_WEIGHT, source=SOURCE_OPP_IMITATION)
                except ValueError:
                    pass
            board.push(move)

    # Store game in DB
    _store_game(
        game_id=game_id,
        pgn=pgn,
        our_color=our_color,
        result=result,
        opponent_username=opponent_username,
        opponent_elo=opponent_elo,
        opponent_is_bot=opponent_is_bot,
        clocks=clocks or "",
    )

    # Update opponent profile
    from ouroboros.opponents.profiles import update_opponent_after_game
    update_opponent_after_game(
        username=opponent_username,
        is_bot=opponent_is_bot,
        last_elo=opponent_elo,
        result=result,
        pgn=pgn,
        our_color=our_color,
    )

    log.info(
        "Processed game %s vs %s (%s): %d positions added",
        game_id, opponent_username, result, len(positions),
    )


def _store_game(
    game_id: str, pgn: str, our_color: str, result: str,
    opponent_username: str, opponent_elo: int, opponent_is_bot: bool, clocks: str,
) -> None:
    ts = datetime.datetime.utcnow().isoformat()
    with get_db() as conn:
        opp = conn.execute(
            "SELECT id FROM opponents WHERE username=?", (opponent_username,)
        ).fetchone()
        opp_id = opp["id"] if opp else None

        conn.execute("""
            INSERT OR IGNORE INTO games
            (lichess_id, opponent_username, opponent_id, our_color, result, pgn, clocks, timestamp, opponent_elo, opponent_is_bot)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (game_id, opponent_username, opp_id, our_color, result, pgn, clocks, ts, opponent_elo, int(opponent_is_bot)))

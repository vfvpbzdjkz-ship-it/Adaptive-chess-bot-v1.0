"""Round-trip encoding test: move_to_index → index_to_move over 1000+ random positions."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chess
import pytest

from ouroboros.engine.encoding import move_to_index, index_to_move, board_to_tensor, POLICY_SIZE


def _random_positions(n: int = 1000):
    positions = []
    board = chess.Board()
    positions.append(board.copy())
    for _ in range(n * 10):
        if board.is_game_over(claim_draw=True) or board.ply() > 200:
            board = chess.Board()
        legal = list(board.legal_moves)
        if not legal:
            board = chess.Board()
            continue
        board.push(random.choice(legal))
        if len(positions) < n:
            positions.append(board.copy())
    return positions[:n]


def test_round_trip():
    """Every legal move in 1000 random positions must survive a round-trip through the index."""
    random.seed(42)
    positions = _random_positions(1000)
    errors = 0
    total = 0
    for board in positions:
        for move in board.legal_moves:
            total += 1
            try:
                idx = move_to_index(board, move)
            except ValueError as e:
                print(f"ENCODE FAIL: {move} on {board.fen()} — {e}")
                errors += 1
                continue

            assert 0 <= idx < POLICY_SIZE, f"Index {idx} out of range for move {move}"

            try:
                recovered = index_to_move(board, idx)
            except ValueError as e:
                print(f"DECODE FAIL: idx={idx} move={move} on {board.fen()} — {e}")
                errors += 1
                continue

            # Compare by from/to/promotion
            assert recovered.from_square == move.from_square, \
                f"from_sq mismatch: {recovered} vs {move} (idx={idx}) board={board.fen()}"
            assert recovered.to_square == move.to_square, \
                f"to_sq mismatch: {recovered} vs {move} (idx={idx}) board={board.fen()}"
            assert recovered.promotion == move.promotion, \
                f"promo mismatch: {recovered} vs {move} (idx={idx}) board={board.fen()}"

    assert errors == 0, f"{errors}/{total} round-trip failures"
    print(f"✓ {total} move round-trips passed across {len(positions)} positions")


def test_board_tensor_shape():
    board = chess.Board()
    t = board_to_tensor(board)
    assert t.shape == (19, 8, 8), f"Expected (19,8,8), got {t.shape}"


def test_board_tensor_black():
    """Tensor from Black's perspective should have correct own-piece planes."""
    board = chess.Board()
    board.push_uci("e2e4")  # now Black to move
    t = board_to_tensor(board)
    assert t.shape == (19, 8, 8)
    # Black's pawns should be in plane 0 (own pieces)
    # From black's flipped perspective, Black pawns are on rank 1 (row index 1)
    assert t[0].sum() == 8, "Black should have 8 pawns in own-pawn plane"


def test_index_range():
    """All legal moves in starting position map to valid indices."""
    board = chess.Board()
    for move in board.legal_moves:
        idx = move_to_index(board, move)
        assert 0 <= idx < POLICY_SIZE


if __name__ == "__main__":
    test_round_trip()
    test_board_tensor_shape()
    test_board_tensor_black()
    test_index_range()
    print("All encoding tests passed.")

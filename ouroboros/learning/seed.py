"""One-time heuristic bootstrap. Runs once during wizard setup.

THE HEURISTIC IN THIS FILE IS NEVER IMPORTED OR USED AT PLAY TIME.
It exists solely to give the network a warm start before real learning begins.
"""
import logging
import random
import time
from pathlib import Path

import chess
import numpy as np
import torch

from ouroboros.engine.encoding import board_to_tensor, move_to_index, legal_move_mask
from ouroboros.engine.network import OuroborosNet, build_net, save_checkpoint, best_path, latest_path, ckpt_path
from ouroboros.learning.buffer import ReplayBuffer, SOURCE_SELFPLAY
from ouroboros.persistence import meta_set

log = logging.getLogger(__name__)

# ── Tiny embedded heuristic ────────────────────────────────────────────────────
# Lives ONLY here. Never imported elsewhere. Designed to be overwritten.

_PIECE_VALUE = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

_CENTER_SQUARES = frozenset([
    chess.D4, chess.D5, chess.E4, chess.E5,
    chess.C3, chess.C4, chess.C5, chess.C6,
    chess.D3, chess.D6, chess.E3, chess.E6,
    chess.F3, chess.F4, chess.F5, chess.F6,
])


def _heuristic_score(board: chess.Board, move: chess.Move) -> float:
    """Score a move using pure material + simple positional heuristics."""
    score = random.gauss(0, 0.1)  # noise

    # Capture bonus
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured:
            score += _PIECE_VALUE.get(captured.piece_type, 0) * 0.5

    # Check bonus
    board.push(move)
    if board.is_check():
        score += 0.3
    board.pop()

    # Centrality bonus for destination
    if move.to_square in _CENTER_SQUARES:
        score += 0.15

    # Promotion bonus
    if move.promotion and move.promotion == chess.QUEEN:
        score += 5.0

    return score


def _generate_seed_game(max_plies: int = 120) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Play one seed game using heuristic moves. Returns (state, policy, z) list."""
    board = chess.Board()
    history = []

    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        legal = list(board.legal_moves)
        if not legal:
            break

        scores = [_heuristic_score(board, m) for m in legal]
        # Softmax sampling
        scores_arr = np.array(scores)
        scores_arr -= scores_arr.max()
        probs = np.exp(scores_arr)
        probs /= probs.sum()

        move = np.random.choice(legal, p=probs)
        state = board_to_tensor(board).numpy()

        # Build soft one-hot policy
        policy = np.zeros(4672, dtype=np.float32)
        try:
            idx = move_to_index(board, move)
            # Label smoothing: 0.9 on chosen, spread rest
            policy += 0.1 / 4672
            policy[idx] += 0.9 - 0.1 / 4672
        except ValueError:
            policy[0] = 1.0

        history.append((state, policy, board.turn))
        board.push(move)

    # Result
    if board.is_checkmate():
        white_result = -1.0 if board.turn == chess.WHITE else 1.0
    else:
        white_result = 0.0

    samples = []
    for state, policy, turn_color in history:
        z = white_result if turn_color == chess.WHITE else -white_result
        samples.append((state, policy, z))
    return samples


def run_seed(cfg: dict, n_games: int = 30_000, train_steps: int = 3_000) -> None:
    """Generate seed games, train, save initial checkpoint."""
    from pathlib import Path
    Path("data/models").mkdir(parents=True, exist_ok=True)

    device = cfg.get("device", "cpu")
    net = build_net(cfg, device)
    net.train()

    buffer = ReplayBuffer(capacity=min(cfg.get("buffer_capacity", 1_000_000), 200_000))

    print(f"Generating {n_games} seed games (this takes ~10-30 min on CPU)...")
    t0 = time.time()
    for i in range(n_games):
        samples = _generate_seed_game()
        for state, policy, z in samples:
            buffer.add(state, policy, z, weight=1.0, source=SOURCE_SELFPLAY)
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n_games - i - 1) / max(rate, 0.01)
            print(f"  {i+1}/{n_games} games | {rate:.1f} games/s | ~{remaining/60:.1f} min left")

    print(f"Training for {train_steps} steps on seed data...")
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    bs = min(cfg.get("batch_size", 256), 128)

    for step in range(train_steps):
        if buffer.count < bs:
            continue
        states_np, policies_np, values_np, weights_np = buffer.sample(bs)
        states = torch.from_numpy(states_np).to(device)
        policies = torch.from_numpy(policies_np).to(device)
        values = torch.from_numpy(values_np).to(device)

        optimizer.zero_grad()
        logits, vals = net(states)

        import torch.nn.functional as F
        mask = policies > 0
        logits_masked = logits.clone()
        logits_masked[~mask] = -1e9
        log_probs = F.log_softmax(logits_masked, dim=1)
        policy_loss = -(policies * log_probs).sum(dim=1).mean()
        value_loss = ((vals - values) ** 2).mean()
        loss = policy_loss + value_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 500 == 0:
            print(f"  step {step+1}/{train_steps} loss={loss.item():.4f}")

    print("Saving initial checkpoint...")
    ckpt0 = ckpt_path(0)
    save_checkpoint(net, ckpt0, {"step": 0, "seeded": True})
    save_checkpoint(net, latest_path(), {"step": 0, "seeded": True})
    save_checkpoint(net, best_path(), {"step": 0, "seeded": True})
    meta_set("seeded", "true")
    meta_set("train_step", "0")
    buffer.flush()
    print("Seed complete.")

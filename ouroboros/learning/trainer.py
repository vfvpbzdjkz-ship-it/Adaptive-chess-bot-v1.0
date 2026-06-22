"""Optimization loop: trains from replay buffer, manages checkpoints, ladder."""
import logging
import math
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ouroboros.engine.network import (
    OuroborosNet, build_net, save_checkpoint, load_checkpoint,
    best_path, latest_path, ckpt_path, MODELS_DIR,
)
from ouroboros.learning.buffer import ReplayBuffer
from ouroboros.persistence import get_db, meta_get, meta_set

log = logging.getLogger(__name__)


def _cosine_lr(step: int, lr_init: float, lr_final: float, total_steps: int) -> float:
    t = min(step / max(total_steps, 1), 1.0)
    return lr_final + 0.5 * (lr_init - lr_final) * (1 + math.cos(math.pi * t))


def compute_loss(
    net: OuroborosNet,
    states: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    weights: torch.Tensor,
    l2_coef: float = 1e-4,
) -> tuple[torch.Tensor, float, float]:
    logits, values = net(states)

    # Mask logits: only non-zero policy targets
    mask = policy_targets > 0
    logits_masked = logits.clone()
    logits_masked[~mask] = -1e9

    # Cross-entropy policy loss
    log_probs = F.log_softmax(logits_masked, dim=1)
    policy_loss = -(policy_targets * log_probs).sum(dim=1)
    policy_loss = (policy_loss * weights).mean()

    # MSE value loss
    value_loss = ((values - value_targets) ** 2 * weights).mean()

    # L2 regularization
    l2 = sum(p.pow(2).sum() for p in net.parameters())

    total = policy_loss + value_loss + l2_coef * l2
    return total, float(policy_loss.item()), float(value_loss.item())


class Trainer:
    def __init__(self, net: OuroborosNet, buffer: ReplayBuffer, cfg: dict, device: str):
        self.net = net
        self.buffer = buffer
        self.cfg = cfg
        self.device = device
        self.step = int(meta_get("train_step", "0"))
        self.batch_size = cfg.get("batch_size", 256)
        self.l2_coef = cfg.get("l2_weight", 1e-4)
        self.lr_init = cfg.get("lr_initial", 0.02)
        self.lr_final = cfg.get("lr_final", 0.002)
        self.lr_steps = cfg.get("lr_schedule_steps", 500_000)
        self.checkpoint_every = cfg.get("checkpoint_every", 1000)
        self.ladder_every = cfg.get("ladder_every", 5000)
        self.ladder_games = cfg.get("ladder_games", 40)
        self.promo_threshold = cfg.get("promotion_threshold", 0.55)

        self.optimizer = torch.optim.SGD(
            net.parameters(), lr=self.lr_init, momentum=0.9, weight_decay=0.0
        )
        self._last_loss: Optional[float] = None
        self._last_policy_loss: Optional[float] = None
        self._last_value_loss: Optional[float] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _update_lr(self) -> None:
        lr = _cosine_lr(self.step, self.lr_init, self.lr_final, self.lr_steps)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def train_step(self) -> Optional[float]:
        if self.buffer.count < self.batch_size:
            return None

        # Auto-halve batch if OOM
        bs = self.batch_size
        while bs >= 1:
            try:
                states_np, policies_np, values_np, weights_np = self.buffer.sample(bs)
                break
            except Exception:
                bs = bs // 2
                if bs < 1:
                    return None

        states = torch.from_numpy(states_np).to(self.device)
        policies = torch.from_numpy(policies_np).to(self.device)
        values = torch.from_numpy(values_np).to(self.device)
        weights = torch.from_numpy(weights_np).to(self.device)
        weights = weights / (weights.mean() + 1e-8)

        self._update_lr()
        self.net.train()  # activate BatchNorm batch stats + update running stats
        self.optimizer.zero_grad()
        loss, pl, vl = compute_loss(self.net, states, policies, values, weights, self.l2_coef)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        self.net.eval()   # restore eval mode for MCTS inference

        self.step += 1
        meta_set("train_step", str(self.step))
        self._last_loss = float(loss.item())
        self._last_policy_loss = pl
        self._last_value_loss = vl
        return self._last_loss

    def maybe_checkpoint(self) -> None:
        if self.step > 0 and self.step % self.checkpoint_every == 0:
            path = ckpt_path(self.step)
            save_checkpoint(self.net, path, {"step": self.step})
            save_checkpoint(self.net, latest_path(), {"step": self.step})
            log.info("Checkpoint saved at step %d", self.step)
            self._prune_old_checkpoints(keep=5)

        if self.step > 0 and self.step % self.ladder_every == 0:
            self._run_ladder()

    def _prune_old_checkpoints(self, keep: int = 5) -> None:
        """Delete all but the most recent `keep` numbered checkpoint files."""
        try:
            ckpts = sorted(
                MODELS_DIR.glob("ckpt_*.pt"),
                key=lambda p: int(p.stem.split("_")[1]),
            )
            for old in ckpts[:-keep]:
                old.unlink(missing_ok=True)
                log.debug("Pruned old checkpoint %s", old.name)
        except Exception as e:
            log.warning("Checkpoint pruning failed: %s", e)

    def _run_ladder(self) -> None:
        log.info("Running ladder match at step %d", self.step)
        bp = best_path()
        if not bp.exists():
            save_checkpoint(self.net, bp, {"step": self.step, "elo": 1500.0})
            log.info("No best.pt exists; setting current as best.")
            return

        from ouroboros.engine.network import build_net as _build
        best_net = _build(self.cfg, self.device)
        load_checkpoint(best_net, bp, self.device)
        best_net.eval()

        wins = draws = losses = 0
        n = self.ladder_games
        sims = max(16, self.cfg.get("mcts_sims_selfplay", 96) // 4)

        for i in range(n):
            result = _ladder_game(
                net_white=self.net if i % 2 == 0 else best_net,
                net_black=best_net if i % 2 == 0 else self.net,
                device=self.device,
                cfg=self.cfg,
                sims=sims,
            )
            # result from white's perspective
            if i % 2 == 0:
                if result > 0: wins += 1
                elif result < 0: losses += 1
                else: draws += 1
            else:
                if result < 0: wins += 1
                elif result > 0: losses += 1
                else: draws += 1

        score = (wins + 0.5 * draws) / n
        log.info("Ladder: latest vs best = %.1f%% (%dW/%dD/%dL)", score * 100, wins, draws, losses)

        if score >= self.promo_threshold:
            save_checkpoint(self.net, bp, {"step": self.step})
            log.info("Promoted latest -> best.pt")
            self._record_elo(wins, draws, losses, n)
            # Push new best to HF Hub if configured
            from ouroboros.sync import push_promotion
            push_promotion(self.cfg, self.step)
        else:
            log.info("Latest did not beat best (%.0f%% < %.0f%%)", score * 100, self.promo_threshold * 100)

    def _record_elo(self, wins: int, draws: int, losses: int, n: int) -> None:
        with get_db() as conn:
            row = conn.execute(
                "SELECT elo FROM ladder ORDER BY id DESC LIMIT 1"
            ).fetchone()
            prev_elo = row["elo"] if row else 1500.0
            score = (wins + 0.5 * draws) / max(n, 1)
            expected = 1 / (1 + 10 ** (0.0 / 400))  # symmetric match
            k = 32
            new_elo = prev_elo + k * (score - expected) * n
            import datetime
            conn.execute(
                "INSERT INTO ladder(checkpoint, elo, games_played, wins, timestamp) VALUES(?,?,?,?,?)",
                (f"ckpt_{self.step}", new_elo, n, wins, datetime.datetime.utcnow().isoformat()),
            )
            meta_set("ladder_elo", str(new_elo))
            log.info("Internal Elo updated: %.0f -> %.0f", prev_elo, new_elo)
            try:
                from ouroboros.web_viewer import update_elo
                update_elo(new_elo)
            except Exception:
                pass

    def run_loop(self, status_fn=None) -> None:
        """Blocking training loop until stop_event set."""
        while not self._stop_event.is_set():
            loss = self.train_step()
            if loss is not None:
                self.maybe_checkpoint()
                if status_fn:
                    status_fn(
                        steps=self.step, loss=loss,
                        policy_loss=self._last_policy_loss,
                        value_loss=self._last_value_loss,
                    )
            else:
                time.sleep(0.5)

    def start_background(self, status_fn=None) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self.run_loop, args=(status_fn,), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=30)

    @property
    def last_loss(self) -> Optional[float]:
        return self._last_loss

    @property
    def train_step_count(self) -> int:
        return self.step


def _ladder_game(
    net_white: OuroborosNet,
    net_black: OuroborosNet,
    device: str,
    cfg: dict,
    sims: int,
) -> float:
    """Play one game between two nets. Returns result from white's perspective."""
    from ouroboros.engine.mcts import MCTS
    board = chess.Board()
    mcts_w = MCTS(net_white, device, c_puct=cfg.get("c_puct", 1.5), batch_size=8)
    mcts_b = MCTS(net_black, device, c_puct=cfg.get("c_puct", 1.5), batch_size=8)

    max_moves = 300
    for _ in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break
        mcts = mcts_w if board.turn == chess.WHITE else mcts_b
        move, _ = mcts.search(board, sims, add_noise=False)
        board.push(move)

    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0
    return 0.0


import chess

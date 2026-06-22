"""Self-play worker processes that generate training data."""
import logging
import multiprocessing as mp
import queue
import threading
import time
from typing import Optional

import chess
import numpy as np
import torch

from ouroboros.engine.encoding import board_to_tensor, move_to_index
from ouroboros.engine.mcts import MCTS
from ouroboros.engine.network import OuroborosNet, build_net, load_checkpoint, latest_path
from ouroboros.engine.timeman import Timer
from ouroboros.learning.buffer import ReplayBuffer, SOURCE_SELFPLAY

log = logging.getLogger(__name__)

TEMP_PLIES = 30       # tau=1.0 for first N plies, then argmax
RESIGN_THRESHOLD = -0.90
RESIGN_CONSECUTIVE = 5
RESIGN_AUDIT_FRACTION = 0.10  # audit 10% of games without resignation for learning


def _play_game(
    net: OuroborosNet,
    device: str,
    cfg: dict,
    game_count: int,
    add_noise: bool = True,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Play one self-play game. Returns list of (state, policy, z) triples."""
    board = chess.Board()
    mcts = MCTS(
        net=net,
        device=device,
        c_puct=cfg.get("c_puct", 1.5),
        dirichlet_alpha=cfg.get("dirichlet_alpha", 0.3),
        dirichlet_eps=cfg.get("dirichlet_eps", 0.25),
        batch_size=cfg.get("mcts_batch_size", 16),
    )

    resignation_enabled = game_count >= 50_000
    audit_no_resign = np.random.random() < RESIGN_AUDIT_FRACTION

    history: list[tuple[np.ndarray, np.ndarray, int]] = []  # (state, policy, turn_color)
    consecutive_below_threshold = 0
    ply = 0

    while not board.is_game_over(claim_draw=True):
        n_sims = cfg.get("mcts_sims_selfplay", 96)
        tau = 1.0 if ply < TEMP_PLIES else 0.0

        with Timer() as t:
            move, visit_dist = mcts.search(
                board, n_sims,
                add_noise=add_noise,
                forced_tau=tau if tau > 0 else None,
            )

        state_tensor = board_to_tensor(board).numpy()
        history.append((state_tensor, visit_dist, board.turn))

        # Resignation check
        if resignation_enabled and not audit_no_resign:
            root_value = _estimate_root_value(mcts)
            if root_value < RESIGN_THRESHOLD:
                consecutive_below_threshold += 1
                if consecutive_below_threshold >= RESIGN_CONSECUTIVE:
                    break
            else:
                consecutive_below_threshold = 0

        board.push(move)
        ply += 1

    # Determine result
    result = _game_result(board)

    # Build training samples
    samples = []
    for state, policy, turn_color in history:
        z = result if turn_color == chess.WHITE else -result
        samples.append((state, policy, z))

    return samples


def _estimate_root_value(mcts: MCTS) -> float:
    if mcts._root and mcts._root.children:
        best = max(mcts._root.children, key=lambda c: c.n)
        return -best.q  # negate because best child q is from child's perspective
    return 0.0


def _game_result(board: chess.Board) -> float:
    """Result from White's perspective: +1 white wins, -1 black wins, 0 draw."""
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0
    return 0.0  # draw


class SelfPlayWorker:
    """Runs self-play in a background process, feeding samples to buffer via queue.

    The worker process generates games and puts (state, policy, z, weight, source)
    tuples into an mp.Queue. A drain thread in the main process reads from the queue
    and calls buffer.add(), ensuring buffer._count and _write_ptr are updated in the
    main process where the training loop can see them.

    Direct buffer writes from a forked child process are invisible to the parent
    because _write_ptr and _count are copy-on-write Python integers post-fork.
    """

    _QUEUE_MAXSIZE = 2000  # ~47 MB cap; put_nowait drops samples when full

    def __init__(self, cfg: dict, buffer: ReplayBuffer, worker_id: int = 0):
        self.cfg = cfg
        self.buffer = buffer
        self.worker_id = worker_id
        self._stop_event = mp.Event()
        self._process: Optional[mp.Process] = None
        self._game_count = mp.Value("i", 0)
        self._throttle = mp.Value("i", 0)
        self._sample_queue: mp.Queue = mp.Queue(maxsize=self._QUEUE_MAXSIZE)
        self._drain_thread: Optional[threading.Thread] = None
        self._drain_stop = threading.Event()

    def start(self) -> None:
        self._drain_stop.clear()
        self._drain_thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._drain_thread.start()

        self._process = mp.Process(
            target=_worker_loop,
            args=(self.cfg, self._sample_queue, self._stop_event, self._game_count, self._throttle),
            daemon=True,
        )
        self._process.start()
        log.info("SelfPlayWorker %d started (pid=%d)", self.worker_id, self._process.pid)

    def _drain_loop(self) -> None:
        """Read samples from the inter-process queue and add to buffer in main process."""
        while not self._drain_stop.is_set():
            try:
                item = self._sample_queue.get(timeout=1.0)
                state, policy, z, weight, source = item
                self.buffer.add(state, policy, z, weight=weight, source=source)
            except queue.Empty:
                continue
            except Exception as e:
                log.warning("Sample drain error: %s", e)

    def stop(self) -> None:
        self._stop_event.set()
        if self._process:
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()
        # Let drain thread flush any items still in the queue before stopping
        time.sleep(0.5)
        self._drain_stop.set()
        if self._drain_thread:
            self._drain_thread.join(timeout=5)
        self.buffer.flush()

    def throttle(self, enabled: bool) -> None:
        self._throttle.value = 1 if enabled else 0

    @property
    def games_played(self) -> int:
        return self._game_count.value


def _worker_loop(
    cfg: dict,
    sample_queue: mp.Queue,
    stop_event: mp.Event,
    game_count: mp.Value,
    throttle: mp.Value,
) -> None:
    """Worker process: generate self-play games and push samples to main process via queue."""
    import ouroboros.logging_setup as ls
    ls.setup_logging()
    log = logging.getLogger("selfplay.worker")

    device = cfg.get("device", "cpu")
    net = build_net(cfg, device)
    net.eval()

    games_local = 0
    while not stop_event.is_set():
        if throttle.value:
            time.sleep(5)
            continue

        # Reload latest checkpoint periodically
        if games_local % 10 == 0:
            lp = latest_path()
            if lp.exists():
                try:
                    load_checkpoint(net, lp, device)
                    net.eval()
                except Exception as e:
                    log.warning("Failed to load checkpoint: %s", e)

        try:
            samples = _play_game(net, device, cfg, game_count.value)
            for state, policy, z in samples:
                try:
                    sample_queue.put_nowait((state, policy, z, 1.0, SOURCE_SELFPLAY))
                except queue.Full:
                    pass  # training is behind; drop sample gracefully
            with game_count.get_lock():
                game_count.value += 1
            games_local += 1
        except Exception as e:
            log.exception("Error in self-play game: %s", e)
            time.sleep(1)


class SelfPlayManager:
    """Manages multiple SelfPlayWorker processes."""

    def __init__(self, cfg: dict, buffer: ReplayBuffer):
        self.cfg = cfg
        self.buffer = buffer
        n = max(1, cfg.get("n_workers", 1))
        self.workers = [SelfPlayWorker(cfg, buffer, i) for i in range(n)]

    def start(self) -> None:
        for w in self.workers:
            w.start()

    def stop(self) -> None:
        for w in self.workers:
            w.stop()

    def throttle(self, enabled: bool) -> None:
        for w in self.workers:
            w.throttle(enabled)

    @property
    def total_games(self) -> int:
        return sum(w.games_played for w in self.workers)

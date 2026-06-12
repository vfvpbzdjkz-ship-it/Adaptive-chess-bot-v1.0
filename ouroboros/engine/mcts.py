"""MCTS with PUCT selection, batched inference, and tree reuse."""
import logging
import math
from typing import Optional

import chess
import numpy as np
import torch

from ouroboros.engine.encoding import board_to_tensor, legal_move_mask, move_to_index
from ouroboros.engine.network import OuroborosNet

log = logging.getLogger(__name__)

VIRTUAL_LOSS = 3.0


class Node:
    __slots__ = (
        "board_fen", "move", "parent",
        "children", "n", "w", "q", "p",
        "is_terminal", "terminal_value",
        "expanded", "virtual_loss",
    )

    def __init__(self, board_fen: str, move: Optional[chess.Move], parent: Optional["Node"], prior: float):
        self.board_fen = board_fen
        self.move = move
        self.parent = parent
        self.children: list["Node"] = []
        self.n: int = 0
        self.w: float = 0.0
        self.q: float = 0.0
        self.p: float = prior
        self.is_terminal: bool = False
        self.terminal_value: float = 0.0
        self.expanded: bool = False
        self.virtual_loss: int = 0


class MCTS:
    def __init__(
        self,
        net: OuroborosNet,
        device: str,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        batch_size: int = 16,
        use_fp16: bool = False,
    ):
        self.net = net
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and device == "cuda"
        self._root: Optional[Node] = None
        self._sims_done: int = 0

    def _get_value_sign(self, board: chess.Board) -> float:
        """Return terminal game value from the perspective of the player to move."""
        if board.is_checkmate():
            return -1.0  # side to move is mated
        return 0.0  # stalemate / draw

    def _expand(self, node: Node, board: chess.Board, policy_logits: torch.Tensor) -> None:
        """Expand node by creating children with policy priors."""
        mask = legal_move_mask(board)
        if not mask.any():
            node.is_terminal = True
            node.terminal_value = self._get_value_sign(board)
            node.expanded = True
            return

        logits_np = policy_logits.cpu().float().numpy()
        logits_np[~mask] = -1e9
        logits_np -= logits_np.max()
        probs = np.exp(logits_np)
        probs[~mask] = 0.0
        probs /= probs.sum() + 1e-8

        for move in board.legal_moves:
            try:
                idx = move_to_index(board, move)
            except ValueError:
                continue
            child_board = board.copy()
            child_board.push(move)
            child = Node(
                board_fen=child_board.fen(),
                move=move,
                parent=node,
                prior=float(probs[idx]),
            )
            # Check terminal immediately
            if child_board.is_game_over(claim_draw=True):
                child.is_terminal = True
                child.expanded = True
                if child_board.is_checkmate():
                    child.terminal_value = -1.0  # side to move at child is mated
                else:
                    child.terminal_value = 0.0
            node.children.append(child)
        node.expanded = True

    def _puct_score(self, node: Node, parent_n: int) -> float:
        q = node.q
        u = self.c_puct * node.p * math.sqrt(parent_n) / (1 + node.n + node.virtual_loss)
        return q + u

    def _select(self, node: Node) -> list[Node]:
        """Select path from root to an unexpanded/terminal leaf."""
        path = [node]
        while node.expanded and not node.is_terminal and node.children:
            parent_n = node.n + node.virtual_loss
            best = max(node.children, key=lambda c: self._puct_score(c, parent_n))
            best.virtual_loss += VIRTUAL_LOSS
            path.append(best)
            node = best
        return path

    def _backup(self, path: list[Node], value: float) -> None:
        """Back-propagate value. value is from the perspective of the leaf's player."""
        for i, node in enumerate(reversed(path)):
            node.virtual_loss = max(0, node.virtual_loss - VIRTUAL_LOSS)
            node.n += 1
            # Flip sign at each ply: value is from leaf's side; alternates each step
            v = value if i % 2 == 0 else -value
            node.w += v
            node.q = node.w / node.n

    def _add_dirichlet(self, node: Node) -> None:
        if not node.children:
            return
        n = len(node.children)
        noise = np.random.dirichlet([self.dirichlet_alpha] * n)
        eps = self.dirichlet_eps
        for child, eta in zip(node.children, noise):
            child.p = (1 - eps) * child.p + eps * eta

    def _evaluate_batch(
        self, leaves: list[tuple[list[Node], chess.Board]]
    ) -> list[float]:
        """Run network inference on a batch of leaf boards."""
        tensors = [board_to_tensor(b) for _, b in leaves]
        batch = torch.stack(tensors).to(self.device)
        if self.use_fp16:
            batch = batch.half()

        with torch.inference_mode():
            if self.use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits_batch, values = self.net(batch)
            else:
                logits_batch, values = self.net(batch)

        values = values.float().cpu().numpy()
        logits_batch = logits_batch.float().cpu()

        results = []
        for i, (path, board) in enumerate(leaves):
            node = path[-1]
            if node.is_terminal:
                results.append(node.terminal_value)
                continue
            # Expand node
            self._expand(node, board, logits_batch[i])
            results.append(float(values[i]))
        return results

    def search(
        self,
        board: chess.Board,
        n_sims: int,
        add_noise: bool = False,
        root_prior_bias: Optional[np.ndarray] = None,
        excluded_moves: Optional[set] = None,
        forced_tau: Optional[float] = None,
    ) -> tuple[chess.Move, np.ndarray]:
        """Run MCTS and return (best_move, visit_distribution).

        visit_distribution has shape (4672,) with normalized visit counts.
        """
        # Tree reuse: if root matches current board, reuse subtree
        current_fen = board.fen()
        if self._root is None or self._root.board_fen != current_fen:
            self._root = Node(board_fen=current_fen, move=None, parent=None, prior=1.0)
            self._sims_done = 0

        root = self._root

        # First expansion if needed
        if not root.expanded:
            tensors = board_to_tensor(board).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                logits, value = self.net(tensors)
            self._expand(root, board, logits[0])
            # Apply root bias if provided
            if root_prior_bias is not None and root.children:
                for child in root.children:
                    if child.move is not None:
                        try:
                            idx = move_to_index(board, child.move)
                            child.p *= np.exp(root_prior_bias[idx])
                        except (ValueError, IndexError):
                            pass
                # Renormalize
                total = sum(c.p for c in root.children) + 1e-8
                for c in root.children:
                    c.p /= total

        if add_noise:
            self._add_dirichlet(root)

        # Exclude moves (anti-bot branch exploration)
        if excluded_moves:
            for child in root.children:
                if child.move in excluded_moves:
                    child.p = 0.0
            total = sum(c.p for c in root.children) + 1e-8
            for c in root.children:
                c.p /= total

        remaining = n_sims - self._sims_done
        if remaining <= 0:
            remaining = n_sims  # fresh count for reuse

        batch_size = self.batch_size
        done = 0
        while done < remaining:
            batch: list[tuple[list[Node], chess.Board]] = []
            for _ in range(min(batch_size, remaining - done)):
                path = self._select(root)
                leaf = path[-1]
                if leaf.is_terminal:
                    self._backup(path, leaf.terminal_value)
                    done += 1
                    continue
                # Reconstruct board at leaf via FEN (stateless, always correct)
                leaf_board = chess.Board(leaf.board_fen)
                batch.append((path, leaf_board))

            if batch:
                values = self._evaluate_batch(batch)
                for (path_item, _), val in zip(batch, values):
                    self._backup(path_item, val)
                done += len(batch)

        self._sims_done += remaining

        # Build visit distribution
        visit_dist = np.zeros(4672, dtype=np.float32)
        for child in root.children:
            if child.move is not None:
                try:
                    idx = move_to_index(board, child.move)
                    visit_dist[idx] = child.n
                except ValueError:
                    pass

        total_visits = visit_dist.sum()
        if total_visits > 0:
            visit_dist /= total_visits

        # Select move
        if not root.children:
            # Fallback: return first legal move
            move = next(iter(board.legal_moves))
            return move, visit_dist

        if forced_tau is not None and forced_tau > 0:
            temps = np.array([c.n for c in root.children], dtype=np.float64)
            temps = np.power(temps + 1e-8, 1.0 / forced_tau)
            temps /= temps.sum()
            chosen = root.children[np.random.choice(len(root.children), p=temps)]
        else:
            chosen = max(root.children, key=lambda c: c.n)

        # Advance root to chosen child for tree reuse
        chosen.parent = None
        self._root = chosen

        return chosen.move, visit_dist

    def reset(self) -> None:
        self._root = None
        self._sims_done = 0

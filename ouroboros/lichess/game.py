"""Per-game loop: stream state, clock mgmt, send moves, handle draw/resign."""
import logging
import threading
import time
from typing import Optional

import chess

from ouroboros.engine.mcts import MCTS
from ouroboros.engine.network import OuroborosNet
from ouroboros.engine.timeman import TimeManager, Timer
from ouroboros.lichess.client import LichessClient
from ouroboros.opponents.adapt import OpponentContext, get_root_bias, is_book_speed
from ouroboros.opponents.antibot import AntiBotController
from ouroboros import status as st

log = logging.getLogger(__name__)

RESIGN_THRESHOLD = -0.95
RESIGN_CONSECUTIVE = 6
DRAW_THRESHOLD = 0.05
# Abort (< 2 moves) or resign (>= 2 moves) when opponent is silent this long.
OPPONENT_TIMEOUT_SECONDS = 90


class GameRunner:
    def __init__(
        self,
        client: LichessClient,
        net: OuroborosNet,
        device: str,
        cfg: dict,
        game_id: str,
        our_color: str,
        context: Optional[OpponentContext],
        antibot: Optional[AntiBotController],
    ):
        self.client = client
        self.net = net
        self.device = device
        self.cfg = cfg
        self.game_id = game_id
        self.our_color = our_color
        self.our_chess_color = chess.WHITE if our_color == "white" else chess.BLACK
        self.context = context
        self.antibot = antibot

        self.mcts = MCTS(
            net=net,
            device=device,
            c_puct=cfg.get("c_puct", 1.5),
            dirichlet_alpha=cfg.get("dirichlet_alpha", 0.3),
            dirichlet_eps=cfg.get("dirichlet_eps", 0.25),
            batch_size=cfg.get("mcts_batch_size", 16),
        )
        self.timeman = TimeManager(base_sims=cfg.get("mcts_sims_live", 256))
        self._board = chess.Board()
        self._move_list: list[str] = []
        self._resign_count = 0
        self._draw_offered = False
        self._done = threading.Event()
        self._result: Optional[str] = None  # "win" | "loss" | "draw"
        self._our_move_values: list[float] = []
        self._our_move_ucis: list[str] = []
        self._opponent_turn_start: float = 0.0  # wall-clock time opponent's turn began

    def _watchdog_loop(self) -> None:
        """Abort or resign if the opponent goes silent for too long."""
        timeout = self.cfg.get("opponent_timeout_seconds", OPPONENT_TIMEOUT_SECONDS)
        while not self._done.wait(timeout=10):
            t = self._opponent_turn_start
            if t > 0 and time.time() - t > timeout:
                n = len(self._move_list)
                action = "aborting" if n < 2 else "resigning"
                log.info("Opponent silent %.0fs in game %s (%d moves); %s",
                         time.time() - t, self.game_id, n, action)
                if n < 2:
                    self.client.abort_game(self.game_id)
                    self._result = "draw"
                else:
                    self.client.resign_game(self.game_id)
                    self._result = "loss"
                self._done.set()
                return

    def run(self) -> Optional[str]:
        """Stream game, play moves, return result string."""
        if self.cfg.get("chat_enabled", True):
            self.client.chat(self.game_id, "player", "glhf!")

        threading.Thread(target=self._watchdog_loop, daemon=True).start()

        for event in self.client.stream(f"/api/bot/game/stream/{self.game_id}"):
            if self._done.is_set():
                break
            if event.get("type") == "gameFull":
                self._handle_game_full(event)
            elif event.get("type") == "gameState":
                self._handle_game_state(event)
                if self._done.is_set():
                    break
            elif event.get("type") == "chatLine":
                self._handle_chat(event)

        if self.cfg.get("chat_enabled", True):
            self.client.chat(self.game_id, "player", "gg")

        # Update antibot with game-end info
        if self.antibot and self._result:
            self.antibot.on_game_end(self._result, self._board)

        st.update(live_game=None)
        return self._result

    def _handle_game_full(self, event: dict) -> None:
        state = event.get("state", {})
        self._apply_state(state)
        if self._board.turn == self.our_chess_color:
            self._opponent_turn_start = 0.0
            self._think_and_move(state)
        else:
            self._opponent_turn_start = time.time()

    def _handle_game_state(self, event: dict) -> None:
        status = event.get("status", "")
        if status in ("mate", "resign", "stalemate", "draw", "timeout",
                      "outoftime", "cheat", "noStart", "unknownFinish", "aborted"):
            self._done.set()
            winner = event.get("winner", "")
            if winner == self.our_color:
                self._result = "win"
            elif winner == "":
                self._result = "draw"
            else:
                self._result = "loss"
            return

        self._apply_state(event)

        # Check draw offer
        if event.get("bdraw") or event.get("wdraw"):
            self._handle_draw_offer(event)
            return

        if self._board.turn == self.our_chess_color:
            self._opponent_turn_start = 0.0  # opponent just moved; our turn
            self._think_and_move(event)
        else:
            if self._opponent_turn_start == 0.0:
                self._opponent_turn_start = time.time()  # we just moved; start timer

    def _apply_state(self, state: dict) -> None:
        moves_str = state.get("moves", "")
        move_list = moves_str.split() if moves_str.strip() else []

        # Reconstruct board from move list (stateless, crash-proof)
        board = chess.Board()
        for uci in move_list:
            try:
                board.push_uci(uci)
            except ValueError:
                log.error("Illegal move in game %s: %s", self.game_id, uci)
                break

        # Update antibot with new opponent moves
        if self.antibot and len(move_list) > len(self._move_list):
            old_len = len(self._move_list)
            new_moves = move_list[old_len:]
            temp_board = chess.Board()
            for uci in move_list[:old_len]:
                temp_board.push_uci(uci)
            for uci in new_moves:
                if temp_board.turn != self.our_chess_color:
                    try:
                        m = chess.Move.from_uci(uci)
                        self.antibot.record_opponent_move(temp_board, m)
                    except ValueError:
                        pass
                temp_board.push_uci(uci)

        self._board = board
        self._move_list = move_list

        opponent_username = "opponent"
        st.update(live_game=f"{opponent_username} ({len(move_list)} moves)")

    def _think_and_move(self, state: dict) -> None:
        if self.our_chess_color == chess.WHITE:
            remaining_ms = state.get("wtime", 30000)
            increment_ms = state.get("winc", 0)
        else:
            remaining_ms = state.get("btime", 30000)
            increment_ms = state.get("binc", 0)

        board = self._board
        ply = board.ply()
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return

        # Anti-bot: check for move override or forced exploration
        add_noise = False
        forced_tau = None
        move_override = None
        if self.antibot:
            move_override, add_noise, forced_tau = self.antibot.get_move_override(board, ply // 2)
            excluded = self.antibot.get_excluded_moves(ply // 2)
        else:
            excluded = set()

        if move_override and move_override in board.legal_moves:
            self.client.send_move(self.game_id, move_override.uci())
            return

        try:
            book_speed = self.context is not None and is_book_speed(self.context, board)
            sims, think_s = self.timeman.budget_sims(
                remaining_ms, increment_ms,
                cap_sims=self.cfg.get("mcts_sims_live", 256),
                book_speed=book_speed,
            )
            # Hard wall-clock deadline: 2x think budget, minimum 5 s
            move_deadline = time.time() + max(think_s * 2.0, 5.0)

            # Root prior bias from opponent profile
            root_bias = None
            if self.context is not None:
                root_bias = get_root_bias(self.context, board, legal_moves)

            with Timer() as t:
                move, visit_dist = self.mcts.search(
                    board, sims,
                    add_noise=add_noise,
                    root_prior_bias=root_bias,
                    excluded_moves=excluded if excluded else None,
                    forced_tau=forced_tau,
                    deadline=move_deadline,
                )
            self.timeman.record(sims, t.elapsed)

            # Get root value for resign/draw checks
            root_val = 0.0
            if self.mcts._root and self.mcts._root.children:
                best = max(self.mcts._root.children, key=lambda c: c.n)
                root_val = -best.q  # from our perspective

            self._our_move_values.append(root_val)

            # Resignation check
            if root_val < RESIGN_THRESHOLD:
                self._resign_count += 1
            else:
                self._resign_count = 0

            opp_elo = self.context.last_elo if self.context else 1500
            our_elo = int(self.cfg.get("ladder_elo", 1500))
            if (self._resign_count >= RESIGN_CONSECUTIVE and
                    opp_elo - our_elo > self.cfg.get("resign_elo_gap", 300)):
                log.info("Resigning game %s (value %.2f)", self.game_id, root_val)
                self.client.resign_game(self.game_id)
                self._result = "loss"
                self._done.set()
                return

            self.client.send_move(self.game_id, move.uci())

            if self.antibot:
                self.antibot.record_our_move(move, root_val)

            self._our_move_ucis.append(move.uci())

        except Exception as e:
            log.error("MCTS error in game %s (%s); playing random legal move", self.game_id, e)
            import random as _rand
            fallback = _rand.choice(legal_moves)
            self.client.send_move(self.game_id, fallback.uci())

    def _handle_draw_offer(self, state: dict) -> None:
        if self.mcts._root:
            root_val = 0.0
            if self.mcts._root.children:
                best = max(self.mcts._root.children, key=lambda c: c.n)
                root_val = -best.q
            if abs(root_val) <= DRAW_THRESHOLD:
                try:
                    self.client.post(f"/api/bot/game/{self.game_id}/draw/yes")
                except Exception:
                    pass

    def _handle_chat(self, event: dict) -> None:
        pass  # no chat processing needed

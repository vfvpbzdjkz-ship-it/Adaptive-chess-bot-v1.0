"""Compact periodic status line."""
import time
import threading
import logging
from typing import Optional

log = logging.getLogger(__name__)

_state: dict = {
    "mode": "init",
    "live_game": None,
    "buffer_fill": 0,
    "buffer_cap": 1_000_000,
    "train_steps": 0,
    "last_loss": None,
    "lichess_games": 0,
    "selfplay_games": 0,
    "checkpoint": "none",
    "ladder_elo": 1500.0,
    "start_time": time.time(),
    "sims_per_sec": 0.0,
}
_lock = threading.Lock()
_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


def update(**kwargs) -> None:
    with _lock:
        _state.update(kwargs)


def _format_uptime(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _print_status() -> None:
    with _lock:
        elapsed = time.time() - _state["start_time"]
        uptime = _format_uptime(elapsed)
        mode = _state["mode"]
        live = _state["live_game"] or "—"
        buf = _state["buffer_fill"]
        cap = _state["buffer_cap"]
        buf_pct = 100 * buf / max(cap, 1)
        steps = _state["train_steps"]
        loss = _state["last_loss"]
        loss_str = f"{loss:.4f}" if loss is not None else "—"
        lg = _state["lichess_games"]
        sp = _state["selfplay_games"]
        ckpt = _state["checkpoint"]
        elo = _state["ladder_elo"]

    print(
        f"\r[{uptime}] mode={mode} | game={live} | "
        f"buf={buf_pct:.1f}% | steps={steps} | loss={loss_str} | "
        f"lichess={lg} sp={sp} | ckpt={ckpt} | "
        f"internal_elo={elo:.0f} (self-relative)",
        end="",
        flush=True,
    )


def _run(interval: float) -> None:
    while not _stop_event.wait(interval):
        _print_status()


def start(interval: float = 30.0) -> None:
    global _thread
    _state["start_time"] = time.time()
    _stop_event.clear()
    _thread = threading.Thread(target=_run, args=(interval,), daemon=True)
    _thread.start()


def stop() -> None:
    _stop_event.set()
    if _thread:
        _thread.join(timeout=5)
    _print_status()
    print()

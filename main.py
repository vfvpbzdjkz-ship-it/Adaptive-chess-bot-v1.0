"""OUROBOROS entry point. Wizard → mode dispatch."""
import argparse
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

# Ensure project root is in sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OUROBOROS self-learning Lichess bot")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["auto", "play", "train"],
        default=None,
        help="Run mode (overrides config). Default: auto",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def _setup() -> dict:
    """Run wizard (interactive) or cloud setup (env vars), then return config."""
    from ouroboros import config as cfg_mod
    if cfg_mod.is_configured():
        return cfg_mod.load()
    # Cloud / headless: LICHESS_TOKEN env var present → no interactive input needed
    from ouroboros.cloud_setup import is_cloud_mode
    if is_cloud_mode():
        from ouroboros.cloud_setup import run_cloud_setup
        return run_cloud_setup()
    # Local: interactive wizard
    from ouroboros.wizard import run_wizard
    return run_wizard()


def _load_net(cfg: dict):
    """Load best.pt (or latest.pt) into a network ready for inference."""
    from ouroboros.engine.network import build_net, load_checkpoint, best_path, latest_path
    device = cfg.get("device", "cpu")
    net = build_net(cfg, device)
    bp = best_path()
    lp = latest_path()
    if bp.exists():
        load_checkpoint(net, bp, device)
        logging.getLogger(__name__).info("Loaded best.pt")
    elif lp.exists():
        load_checkpoint(net, lp, device)
        logging.getLogger(__name__).info("Loaded latest.pt (no best.pt yet)")
    else:
        logging.getLogger(__name__).warning("No checkpoint found; using random weights")
    net.eval()

    # Warm up PyTorch: the first forward pass initialises kernels and can take
    # many seconds on CPU; doing it now avoids a freeze on the first game move.
    try:
        import torch
        import chess as _chess
        from ouroboros.engine.encoding import board_to_tensor
        with torch.inference_mode():
            dummy = board_to_tensor(_chess.Board()).unsqueeze(0).to(device)
            net(dummy)
        logging.getLogger(__name__).info("Network warmup complete")
    except Exception as _e:
        logging.getLogger(__name__).debug("Warmup skipped: %s", _e)

    return net, device


def run_auto(cfg: dict) -> None:
    """Auto mode: Lichess + background self-play + training."""
    from ouroboros.lichess.client import LichessClient
    from ouroboros.lichess.events import EventLoop
    from ouroboros.lichess.matchmaker import Matchmaker
    from ouroboros.learning.buffer import ReplayBuffer
    from ouroboros.learning.selfplay import SelfPlayManager
    from ouroboros.learning.trainer import Trainer
    from ouroboros.learning.online import process_finished_game
    from ouroboros.persistence import meta_get
    from ouroboros import status as st
    from ouroboros.sync import pull_latest, PeriodicSync
    from ouroboros.web_viewer import (
        update_game, update_training_stats, load_elo_history,
        set_force_game_callback, set_challenge_callback, set_native_status,
        set_reset_callback,
    )
    from ouroboros.scheduler import PlayScheduler

    log = logging.getLogger(__name__)

    # Pull latest weights + DB from HF Hub before loading (no-op if not configured)
    pull_latest(cfg)

    # Surface whether the native (Rust) encoding acceleration is active.
    try:
        from ouroboros.engine.encoding import HAS_NATIVE
        set_native_status(HAS_NATIVE)
        log.info("Encoding backend: %s", "native (rust)" if HAS_NATIVE else "pure-python")
    except Exception:
        pass

    net, device = _load_net(cfg)
    client = LichessClient(cfg["lichess_token"])
    buffer = ReplayBuffer(capacity=cfg.get("buffer_capacity", 1_000_000))
    trainer = Trainer(net, buffer, cfg, device)
    sp_manager = SelfPlayManager(cfg, buffer)
    matchmaker = Matchmaker(client, cfg)

    # Status updates
    elo = float(meta_get("ladder_elo", "1500"))
    st.update(
        mode="auto",
        buffer_cap=cfg.get("buffer_capacity", 1_000_000),
        ladder_elo=elo,
    )
    st.start(interval=30)

    # Seed ELO history from DB so it shows up immediately on the web viewer
    load_elo_history()

    def on_game_start(game_id: str) -> None:
        log.info("Game started: %s — throttling self-play", game_id)
        sp_manager.throttle(True)
        matchmaker.set_in_game(True)
        st.update(live_game=game_id)
        update_game(game_id)

    def on_game_finish(game_id: str, result, opp_username, opp_elo, opp_is_bot, our_color: str = "white") -> None:
        log.info("Game %s finished: %s vs %s (we played %s)", game_id, result, opp_username, our_color)
        sp_manager.throttle(False)
        matchmaker.set_in_game(False)
        st.update(live_game=None, lichess_games=st._state.get("lichess_games", 0) + 1)
        update_game(None)
        _fetch_and_process_game(client, buffer, game_id, result, opp_username, opp_elo, opp_is_bot, cfg, our_color)

    def _update_status(steps: int, loss: float,
                       policy_loss: float = None, value_loss: float = None) -> None:
        st.update(
            train_steps=steps,
            last_loss=loss,
            buffer_fill=buffer.count,
            selfplay_games=sp_manager.total_games,
            checkpoint=f"ckpt_{trainer.train_step_count}",
        )
        try:
            update_training_stats(
                steps, loss, buffer.count, buffer.capacity, sp_manager.total_games,
                policy_loss=policy_loss, value_loss=value_loss,
                source_counts=buffer.source_counts(),
            )
        except Exception:
            pass

    # Start everything
    sp_manager.start()
    trainer.start_background(status_fn=_update_status)
    play_scheduler = PlayScheduler(matchmaker)
    play_scheduler.start()   # starts in Lichess mode; matchmaker started inside
    set_force_game_callback(play_scheduler.force_one_game)
    set_challenge_callback(play_scheduler.challenge_with_options)

    def _do_reset() -> dict:
        """Delete checkpoints + clear buffer so the bot relearns from scratch."""
        import glob as _glob
        removed = 0
        for pattern in ["data/models/*.pt", "data/buffer/*.npy", "data/buffer/*.npz"]:
            for f in _glob.glob(pattern):
                try:
                    os.remove(f)
                    removed += 1
                    log.warning("reset-model: removed %s", f)
                except OSError as e:
                    log.warning("reset-model: could not remove %s: %s", f, e)
        buffer.clear()
        # Reinitialise network weights in-place. Acquire the trainer's step lock so
        # we wait for any in-flight forward/backward pass to finish before overwriting
        # parameters — avoids corrupting gradients or producing a torn read in MCTS.
        with trainer._step_lock:
            from ouroboros.engine.network import build_net as _bn
            fresh = _bn(cfg, device)
            net.load_state_dict(fresh.state_dict())
            net.eval()
        log.warning("reset-model: weights reset to random; %d file(s) deleted", removed)
        return {"removed": removed}

    set_reset_callback(_do_reset)
    periodic_sync = PeriodicSync(cfg, interval_minutes=15)
    periodic_sync.start()

    event_loop = EventLoop(
        client, net, device, cfg,
        on_game_start=on_game_start,
        on_game_finish=on_game_finish,
    )

    def _on_signal(sig, frame):
        log.info("Shutdown signal received; stopping...")
        event_loop.stop()
        play_scheduler.stop()
        matchmaker.stop()
        trainer.stop()
        sp_manager.stop()
        periodic_sync.stop()
        buffer.flush()
        st.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    log.info("Auto mode running. Press Ctrl+C to stop.")
    event_loop.run()


def run_play(cfg: dict) -> None:
    """Play mode: Lichess only, no training."""
    from ouroboros.lichess.client import LichessClient
    from ouroboros.lichess.events import EventLoop
    from ouroboros import status as st
    from ouroboros.persistence import meta_get

    log = logging.getLogger(__name__)
    net, device = _load_net(cfg)
    client = LichessClient(cfg["lichess_token"])

    elo = float(meta_get("ladder_elo", "1500"))
    st.update(mode="play", ladder_elo=elo)
    st.start()

    event_loop = EventLoop(client, net, device, cfg)

    def _on_signal(sig, frame):
        event_loop.stop()
        st.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    log.info("Play mode running.")
    event_loop.run()


def run_train(cfg: dict) -> None:
    """Train mode: pure offline self-play + training, no network needed."""
    from ouroboros.learning.buffer import ReplayBuffer
    from ouroboros.learning.selfplay import SelfPlayManager
    from ouroboros.learning.trainer import Trainer
    from ouroboros import status as st
    from ouroboros.persistence import meta_get
    from ouroboros.engine.network import build_net, load_checkpoint, latest_path

    log = logging.getLogger(__name__)

    device = cfg.get("device", "cpu")
    net = build_net(cfg, device)
    lp = latest_path()
    if lp.exists():
        load_checkpoint(net, lp, device)

    buffer = ReplayBuffer(capacity=cfg.get("buffer_capacity", 1_000_000))
    trainer = Trainer(net, buffer, cfg, device)
    sp_manager = SelfPlayManager(cfg, buffer)

    elo = float(meta_get("ladder_elo", "1500"))
    st.update(mode="train", buffer_cap=cfg.get("buffer_capacity", 1_000_000), ladder_elo=elo)
    st.start()

    def _update_status(steps: int, loss: float) -> None:
        st.update(
            train_steps=steps,
            last_loss=loss,
            buffer_fill=buffer.count,
            selfplay_games=sp_manager.total_games,
        )

    sp_manager.start()
    trainer.start_background(status_fn=_update_status)

    def _on_signal(sig, frame):
        log.info("Stopping training...")
        trainer.stop()
        sp_manager.stop()
        buffer.flush()
        st.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    log.info("Train mode running. Press Ctrl+C to stop.")
    while True:
        time.sleep(60)


def _fetch_and_process_game(
    client, buffer, game_id: str, result, opp_username, opp_elo, opp_is_bot, cfg,
    our_color: str = "white",
) -> None:
    """Fetch completed game PGN from Lichess and process it."""
    import logging
    log = logging.getLogger(__name__)
    try:
        import requests
        resp = requests.get(
            f"https://lichess.org/game/export/{game_id}",
            headers={
                "Authorization": f"Bearer {cfg['lichess_token']}",
                "Accept": "application/x-chess-pgn",
            },
            timeout=15,
        )
        if resp.status_code == 200:
            pgn = resp.text
            from ouroboros.learning.online import process_finished_game
            process_finished_game(
                buffer=buffer,
                game_id=game_id,
                pgn=pgn,
                our_color=our_color,
                result=result or "draw",
                opponent_username=opp_username,
                opponent_elo=opp_elo,
                opponent_is_bot=opp_is_bot,
            )
        else:
            log.warning("Could not fetch PGN for game %s (HTTP %d)", game_id, resp.status_code)
    except Exception as e:
        log.warning("Could not fetch/process game %s: %s", game_id, e)


def main() -> None:
    args = _parse_args()

    # Setup logging before anything else
    from ouroboros.logging_setup import setup_logging
    import logging as _logging
    setup_logging(level=_logging.DEBUG if args.debug else _logging.INFO)
    log = _logging.getLogger(__name__)

    # Start the spectator web server immediately — before any setup that might
    # take time — so Railway health checks pass from the first second.
    _viewer = None
    if os.environ.get("LICHESS_TOKEN") and os.environ.get("PORT"):
        try:
            from ouroboros.web_viewer import WebViewer
            _viewer = WebViewer()
            _viewer.start()
            print(f"Web viewer started on port {_viewer.port}", flush=True)
        except Exception as exc:
            print(f"Web viewer could not start: {exc}", flush=True)

    # Ensure data directories exist
    for d in ["data", "data/models", "data/buffer", "data/logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # RESET_ON_START: wipe model checkpoints + buffer so the bot relearns from
    # scratch.  Set this env var in Railway, redeploy once, then remove it.
    if os.environ.get("RESET_ON_START"):
        import glob as _glob
        removed = []
        for pattern in ["data/models/*.pt", "data/buffer/*.npy", "data/buffer/*.npz"]:
            for f in _glob.glob(pattern):
                try:
                    os.remove(f)
                    removed.append(f)
                except OSError:
                    pass
        log.warning("RESET_ON_START: removed %d file(s): %s", len(removed),
                    ", ".join(removed) or "(none found)")

    # Always init DB on boot — /tmp is wiped on container restart so tables
    # must be recreated even when config.json already exists on the volume.
    from ouroboros.persistence import init_db
    init_db()

    cfg = _setup()

    # Update viewer with the bot username once config is loaded
    if _viewer is not None:
        from ouroboros.web_viewer import set_username
        set_username(cfg.get("bot_username", ""))

    mode = args.mode or cfg.get("mode", "auto")
    log.info("Starting OUROBOROS in mode: %s", mode)

    if mode == "train":
        run_train(cfg)
    elif mode == "play":
        run_play(cfg)
    else:
        run_auto(cfg)


if __name__ == "__main__":
    main()

"""Load, validate, and provide configuration. Single source of truth."""
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("data/config.json")

# Bump when operational defaults below must override a config.json that was
# already written to a persisted volume (e.g. Railway). load() migrates older
# stored configs up to this version, forcing the operational keys it manages.
CONFIG_VERSION = 2

# Operational keys that the migration re-asserts from code on a version bump.
# These are runtime behaviours (what to play, how hard to think) that should
# track the deployed code, not a stale config baked at first boot.
_MIGRATION_KEYS = (
    "accept_blitz", "accept_rapid", "accept_classical", "accept_bullet",
    "accept_correspondence", "matchmaker_time", "matchmaker_increment",
    "matchmaker_max_per_hour",
)

DEFAULTS: dict[str, Any] = {
    "config_version": CONFIG_VERSION,
    "mode": "auto",
    "lichess_token": "",
    "bot_username": "",
    "hardware_profile": "small",  # small | medium | large
    "device": "cpu",
    "n_workers": 1,
    "mcts_sims_selfplay": 96,
    "mcts_sims_live": 256,
    "c_puct": 1.5,
    "dirichlet_alpha": 0.3,
    "dirichlet_eps": 0.25,
    "batch_size": 256,
    "buffer_capacity": 1_000_000,
    "lr_initial": 0.02,
    "lr_final": 0.002,
    "lr_schedule_steps": 500_000,
    "l2_weight": 1e-4,
    "checkpoint_every": 500,
    "ladder_every": 2000,
    "ladder_games": 20,
    "promotion_threshold": 0.55,
    "accept_rated": True,
    "accept_casual": True,
    # Classical-only: long time controls give the (currently weak) net enough
    # search per move to find tactics and avoid blunders, which is the single
    # biggest lever on actually winning games.
    "accept_blitz": False,
    "accept_rapid": False,
    "accept_classical": True,
    "accept_bullet": False,
    "accept_correspondence": False,
    "max_concurrent_games": 1,
    "matchmaker_enabled": True,
    # 30+0 -> 30 min estimated duration, comfortably in Lichess "classical".
    "matchmaker_time": 30,
    "matchmaker_increment": 0,
    "matchmaker_max_per_hour": 6,
    "chat_enabled": True,
    "winner_imitation": True,
    "resign_threshold": -0.95,
    "resign_consecutive": 6,
    "resign_elo_gap": 300,
    "draw_threshold": 0.05,
    # Opponent adaptation
    "opening_plies": 16,
    "adapt_lambda_max": 0.6,
    "adapt_lambda_root_max": 0.3,
    "adapt_ema_alpha": 0.25,
    "determinism_threshold": 0.7,
    # Book-speed cutoff confidence
    "book_speed_confidence": 0.5,
    # Hugging Face Hub sync (optional — env vars HF_TOKEN / HF_REPO take priority)
    "hf_token": "",
    "hf_repo": "",
}

_PROFILE_SPECS = {
    # sims_live raised: classical clocks leave plenty of time, and MCTS does
    # exact terminal/checkmate detection, so deeper search wins games even with
    # a lightly-trained net. The per-move wall-clock deadline still caps it.
    "small":  {"blocks": 5,  "channels": 64,  "sims_sp": 96,  "sims_live": 1024},
    "medium": {"blocks": 10, "channels": 128, "sims_sp": 256, "sims_live": 1200},
    "large":  {"blocks": 19, "channels": 256, "sims_sp": 600, "sims_live": 1600},
}


def detect_hardware() -> tuple[str, str, int]:
    """Returns (profile, device, n_workers)."""
    cores = multiprocessing.cpu_count()
    try:
        import torch
        if torch.cuda.is_available():
            return "large", "cuda", max(1, cores - 2)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "medium", "mps", max(1, cores - 2)
    except ImportError:
        pass
    return "small", "cpu", max(1, cores - 2)


def load() -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    stored: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            stored = json.load(f)
        cfg.update(stored)

    # Apply profile specs
    spec = _PROFILE_SPECS.get(cfg["hardware_profile"], _PROFILE_SPECS["small"])
    cfg.setdefault("net_blocks", spec["blocks"])
    cfg.setdefault("net_channels", spec["channels"])

    # Check the stored version, not the merged cfg (DEFAULTS already carries the
    # current version, which would otherwise mask a stale stored config).
    if stored and int(stored.get("config_version", 0)) < CONFIG_VERSION:
        # Migrate a stale stored config: re-assert operational keys from code and
        # re-resolve the live search budget from the profile, then persist so the
        # change survives restarts.
        for key in _MIGRATION_KEYS:
            cfg[key] = DEFAULTS[key]
        cfg["mcts_sims_live"] = spec["sims_live"]
        cfg["config_version"] = CONFIG_VERSION
        try:
            save(cfg)
        except Exception:
            pass
        return cfg

    if cfg["mcts_sims_selfplay"] == DEFAULTS["mcts_sims_selfplay"]:
        cfg["mcts_sims_selfplay"] = spec["sims_sp"]
    if cfg["mcts_sims_live"] == DEFAULTS["mcts_sims_live"]:
        cfg["mcts_sims_live"] = spec["sims_live"]
    return cfg


def save(cfg: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(CONFIG_PATH) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG_PATH)
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except OSError:
        pass


def is_configured() -> bool:
    return CONFIG_PATH.exists()


def profile_spec(name: str) -> dict:
    return _PROFILE_SPECS.get(name, _PROFILE_SPECS["small"])

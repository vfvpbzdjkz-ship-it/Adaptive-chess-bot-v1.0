"""Load, validate, and provide configuration. Single source of truth."""
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("data/config.json")

DEFAULTS: dict[str, Any] = {
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
    "checkpoint_every": 1000,
    "ladder_every": 5000,
    "ladder_games": 40,
    "promotion_threshold": 0.55,
    "accept_rated": True,
    "accept_casual": True,
    "accept_blitz": True,
    "accept_rapid": True,
    "accept_classical": True,
    "accept_bullet": False,
    "max_concurrent_games": 1,
    "matchmaker_enabled": True,
    "matchmaker_time": 5,
    "matchmaker_increment": 3,
    "matchmaker_max_per_hour": 10,
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
    "small":  {"blocks": 5,  "channels": 64,  "sims_sp": 96,  "sims_live": 256},
    "medium": {"blocks": 10, "channels": 128, "sims_sp": 256, "sims_live": 800},
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
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            stored = json.load(f)
        cfg.update(stored)
    # Apply profile specs
    spec = _PROFILE_SPECS.get(cfg["hardware_profile"], _PROFILE_SPECS["small"])
    cfg.setdefault("net_blocks", spec["blocks"])
    cfg.setdefault("net_channels", spec["channels"])
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

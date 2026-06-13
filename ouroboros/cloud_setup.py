"""Headless configuration from environment variables (no interactive wizard).

Used automatically when LICHESS_TOKEN env var is set and data/config.json
does not exist — i.e., first boot in a cloud environment like Railway.
"""
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_bool(key: str, default: bool = True) -> bool:
    val = os.environ.get(key, "").strip().lower()
    if not val:
        return default
    return val not in ("0", "false", "no", "off")


def _env_int(key: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def is_cloud_mode() -> bool:
    """True when LICHESS_TOKEN env var is set (cloud / headless deployment)."""
    return bool(os.environ.get("LICHESS_TOKEN", "").strip())


def _validate_token(token: str) -> dict:
    import requests
    resp = requests.get(
        "https://lichess.org/api/account",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    if resp.status_code == 401:
        raise RuntimeError(
            "LICHESS_TOKEN is invalid or expired (got 401).\n"
            "Go to Railway → Variables → update LICHESS_TOKEN with a fresh token from:\n"
            "  https://lichess.org/account/oauth/token/create\n"
            "Required scopes: bot:play  challenge:read  challenge:write"
        )
    resp.raise_for_status()
    return resp.json()


def _upgrade_to_bot(token: str) -> None:
    import requests
    resp = requests.post(
        "https://lichess.org/api/bot/account/upgrade",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    if resp.status_code == 200:
        log.info("Account upgraded to BOT.")
    elif resp.status_code == 400:
        data = resp.json()
        msg = data.get("error", str(data))
        if "already" in msg.lower():
            log.info("Account is already a BOT.")
        else:
            raise RuntimeError(f"BOT upgrade failed: {msg}")
    else:
        raise RuntimeError(f"BOT upgrade HTTP {resp.status_code}")


def run_cloud_setup() -> dict:
    """Build config from environment variables; run seed if first boot."""
    token = _env("LICHESS_TOKEN")
    if not token:
        raise RuntimeError("LICHESS_TOKEN environment variable is required.")

    print("=== OUROBOROS Cloud Setup ===")
    print("Reading configuration from environment variables...")

    # Validate token
    print("Validating Lichess token...")
    account = _validate_token(token)
    username = account.get("username", "")
    title = account.get("title", "") or ""
    print(f"Connected as: {username}")

    # BOT upgrade if needed
    if title != "BOT":
        print("Upgrading account to BOT...")
        _upgrade_to_bot(token)
    else:
        print("Account is already a BOT.")

    # Hardware (Railway free = CPU only)
    profile = _env("HARDWARE_PROFILE", "small")
    device = "cpu"
    import multiprocessing
    n_workers = max(1, multiprocessing.cpu_count() - 1)

    # Try GPU just in case a paid Railway instance has one
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            if profile == "small":
                profile = "medium"
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            if profile == "small":
                profile = "medium"
    except Exception:
        pass

    print(f"Hardware profile: {profile} | device: {device} | workers: {n_workers}")

    from ouroboros.config import DEFAULTS, profile_spec, save as save_cfg
    spec = profile_spec(profile)

    cfg = dict(DEFAULTS)
    cfg.update({
        "lichess_token": token,
        "bot_username": username,
        "hardware_profile": profile,
        "device": device,
        "n_workers": n_workers,
        "net_blocks": spec["blocks"],
        "net_channels": spec["channels"],
        "mcts_sims_selfplay": spec["sims_sp"],
        "mcts_sims_live": spec["sims_live"],
        "mode": _env("MODE", "auto"),
        "matchmaker_enabled": _env_bool("MATCHMAKER_ENABLED", True),
        "accept_rated": _env_bool("ACCEPT_RATED", True),
        "accept_casual": _env_bool("ACCEPT_CASUAL", True),
        "accept_bullet": _env_bool("ACCEPT_BULLET", False),
        "max_concurrent_games": _env_int("MAX_CONCURRENT_GAMES", 1),
        "chat_enabled": _env_bool("CHAT_ENABLED", True),
        "winner_imitation": _env_bool("WINNER_IMITATION", True),
    })

    # Ensure directories exist
    for d in ["data/models", "data/buffer", "data/logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    from ouroboros.persistence import init_db
    init_db()

    # Run seed if no checkpoint exists yet
    from ouroboros.engine.network import best_path
    if not best_path().exists():
        n_games = _env_int("SEED_GAMES", 5_000)
        train_steps = _env_int("SEED_TRAIN_STEPS", 1_000)
        print(f"Running seed bootstrap ({n_games} games, {train_steps} steps)...")
        print("This is a one-time step. Progress is saved — safe to restart.")
        from ouroboros.learning.seed import run_seed
        run_seed(cfg, n_games=n_games, train_steps=train_steps)
    else:
        print("Existing checkpoint found — skipping seed.")

    save_cfg(cfg)
    print("Cloud setup complete.")
    return cfg

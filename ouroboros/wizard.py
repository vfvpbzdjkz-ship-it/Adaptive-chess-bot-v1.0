"""First-run interactive setup wizard."""
import getpass
import json
import os
import sys
from pathlib import Path

import requests

EXPECTATIONS = """
╔══════════════════════════════════════════════════════════════════╗
║          OUROBOROS — Self-Learning Lichess Chess Bot             ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  REALISTIC EXPECTATIONS (please read before continuing):         ║
║                                                                  ║
║  Ouroboros starts nearly random and learns purely from games.    ║
║  On consumer hardware, the realistic improvement timeline is:    ║
║                                                                  ║
║   • Days 1–3:   Stops hanging pieces randomly                    ║
║   • Weeks 1–3:  Coherent, club-ish play emerges                  ║
║   • Months+:    Continues improving, diminishing returns         ║
║                                                                  ║
║  It will NOT beat Stockfish through general strength.            ║
║  Its real superpower: opponent-adaptive play. Given enough       ║
║  games, it can exploit specific deterministic bots well above    ║
║  its general strength by learning exactly what beats them.       ║
║                                                                  ║
║  The internal Elo shown is self-relative, NOT a real rating.     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

TOKEN_INSTRUCTIONS = """
To get your Lichess API token:
  1. Log into your NEW, EMPTY Lichess bot account (NOT your main account!)
  2. Go to: https://lichess.org/account/oauth/token/create
  3. Enable scopes:  bot:play   challenge:read   challenge:write
  4. Create the token and paste it below.

WARNING: Only use an account with ZERO games played.
BOT upgrade is permanent and irreversible.
"""


def _validate_token(token: str) -> tuple[bool, dict]:
    try:
        resp = requests.get(
            "https://lichess.org/api/account",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if resp.status_code == 200:
            return True, resp.json()
        return False, {}
    except Exception as e:
        print(f"  Network error: {e}")
        return False, {}


def _detect_hardware() -> tuple[str, str, int]:
    import multiprocessing
    cores = multiprocessing.cpu_count()
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            print(f"  Detected GPU: {gpu}")
            return "large", "cuda", max(1, cores - 2)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("  Detected Apple MPS")
            return "medium", "mps", max(1, cores - 2)
    except ImportError:
        pass
    print(f"  No GPU detected. Using CPU ({cores} cores).")
    return "small", "cpu", max(1, cores - 2)


_PROFILE_DESCRIPTIONS = {
    "small":  "5 blocks × 64 ch  | 96 self-play sims | 256 live sims  (CPU)",
    "medium": "10 blocks × 128 ch | 256 self-play sims | 800 live sims  (laptop GPU / MPS)",
    "large":  "19 blocks × 256 ch | 600 self-play sims | 1600 live sims (desktop GPU)",
}


def run_wizard() -> dict:
    print(EXPECTATIONS)
    input("Press Enter to begin setup...")
    print()

    # ── Step 1: Token ──────────────────────────────────────────────────────────
    print("── Step 1: Lichess API Token ──")
    print(TOKEN_INSTRUCTIONS)
    account = None
    token = ""
    while True:
        token = getpass.getpass("Paste your Lichess API token: ").strip()
        if not token:
            print("Token cannot be empty.")
            continue
        print("  Validating token...")
        ok, account = _validate_token(token)
        if ok:
            username = account.get("username", "?")
            print(f"  ✓ Connected as: {username}")
            break
        else:
            print("  ✗ Token invalid or network error. Please try again.")

    # ── Step 2: BOT upgrade ────────────────────────────────────────────────────
    print("\n── Step 2: BOT Account Upgrade ──")
    title = account.get("title", "") or ""
    if title == "BOT":
        print("  ✓ Account is already a BOT.")
    else:
        print("""
  IMPORTANT: Upgrading to BOT is PERMANENT and IRREVERSIBLE.
  This only works on accounts that have played ZERO games.
  After upgrade, this account can ONLY be used as a bot.
        """)
        confirm = input("  Type 'yes' to upgrade this account to BOT: ").strip().lower()
        if confirm != "yes":
            print("  Upgrade cancelled. Exiting.")
            sys.exit(0)
        try:
            resp = requests.post(
                "https://lichess.org/api/bot/account/upgrade",
                headers={"Authorization": f"Bearer {token}"},
                timeout=15,
            )
            if resp.status_code == 200:
                print("  ✓ Account upgraded to BOT successfully!")
            elif resp.status_code == 400:
                data = resp.json()
                msg = data.get("error", str(data))
                if "already played" in msg.lower() or "games" in msg.lower():
                    print("\n  ✗ ERROR: This account has already played games.")
                    print("    BOT upgrade requires an account with ZERO games played.")
                    print("    Please create a new Lichess account and use that token.")
                else:
                    print(f"\n  ✗ Upgrade failed: {msg}")
                sys.exit(1)
            else:
                print(f"  ✗ Upgrade failed with status {resp.status_code}")
                sys.exit(1)
        except Exception as e:
            print(f"  ✗ Network error during upgrade: {e}")
            sys.exit(1)

    # ── Step 3: Hardware profile ───────────────────────────────────────────────
    print("\n── Step 3: Hardware Profile ──")
    detected_profile, detected_device, detected_workers = _detect_hardware()
    print(f"\n  Recommended profile: {detected_profile}")
    print(f"  {_PROFILE_DESCRIPTIONS[detected_profile]}")
    print()
    print("  Available profiles:")
    for name, desc in _PROFILE_DESCRIPTIONS.items():
        print(f"    {name}: {desc}")
    choice = input(f"\n  Press Enter to accept '{detected_profile}', or type small/medium/large: ").strip().lower()
    if choice in _PROFILE_DESCRIPTIONS:
        profile = choice
    else:
        profile = detected_profile

    from ouroboros.config import profile_spec
    spec = profile_spec(profile)
    print(f"  ✓ Using profile: {profile} ({spec['blocks']} blocks × {spec['channels']} channels)")

    # ── Step 4: Challenge policy ───────────────────────────────────────────────
    print("\n── Step 4: Challenge Policy ──")
    print("""
  Defaults:
    • Accept standard chess only
    • Casual AND rated games
    • Blitz / Rapid / Classical (no bullet on CPU)
    • Max 1 concurrent game
    """)
    policy_change = input("  Press Enter to accept defaults, or type 'edit' to change: ").strip().lower()
    accept_bullet = False
    if policy_change == "edit":
        accept_bullet = input("  Accept bullet games? (y/N): ").strip().lower() == "y"
        print("  Other defaults kept.")

    # ── Step 5: Matchmaker ─────────────────────────────────────────────────────
    print("\n── Step 5: Matchmaker ──")
    mm_answer = input(
        "  When idle, should I challenge other online Lichess bots to generate\n"
        "  real training games? (recommended: yes) [Y/n]: "
    ).strip().lower()
    matchmaker_enabled = mm_answer != "n"

    # ── Step 6: Seed ───────────────────────────────────────────────────────────
    print("\n── Step 6: Initial Seed Bootstrap ──")
    print("  This generates seed training games using a simple heuristic.")
    print("  Takes approximately 10–30 minutes on CPU (less on GPU).")
    print("  You can interrupt and restart — setup will pick up where it left off.")

    n_games = 30_000
    train_steps = 3_000
    quick = input("  Use quick seed (5k games, faster but weaker start)? [y/N]: ").strip().lower()
    if quick == "y":
        n_games = 5_000
        train_steps = 1_000

    # Build config
    from ouroboros.config import detect_hardware, DEFAULTS
    cfg = dict(DEFAULTS)
    cfg.update({
        "lichess_token": token,
        "bot_username": account.get("username", ""),
        "hardware_profile": profile,
        "device": detected_device,
        "n_workers": detected_workers,
        "net_blocks": spec["blocks"],
        "net_channels": spec["channels"],
        "mcts_sims_selfplay": spec["sims_sp"],
        "mcts_sims_live": spec["sims_live"],
        "accept_bullet": accept_bullet,
        "matchmaker_enabled": matchmaker_enabled,
    })

    # Init DB and directories
    from ouroboros.persistence import init_db
    from ouroboros.config import save as save_cfg
    Path("data/models").mkdir(parents=True, exist_ok=True)
    Path("data/buffer").mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    init_db()

    # Run seed
    print("\n  Starting seed bootstrap...")
    from ouroboros.learning.seed import run_seed
    run_seed(cfg, n_games=n_games, train_steps=train_steps)

    # Save config
    save_cfg(cfg)
    print("\n✓ Setup complete!")
    print("""
  To change settings later:
    • Delete data/config.json to re-run this wizard
    • Or edit data/config.json directly (the ONLY file you ever need to touch)

  Run ./run.sh (or run.bat on Windows) to start the bot.
    """)
    return cfg

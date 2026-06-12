"""Push/pull learning artifacts to Hugging Face Hub.

Synced:  best.pt, latest.pt, ckpt_*.pt, ouroboros.db
Skipped: replay buffer (large, regenerates fast), config.json (contains token)

Set HF_TOKEN and HF_REPO env vars (or config keys) to enable.
HF_REPO format: "username/repo-name"  e.g. "alice/ouroboros-brain"
"""
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_SYNC_LOCK = threading.Lock()


def _get_credentials(cfg: dict) -> tuple[Optional[str], Optional[str]]:
    token = os.environ.get("HF_TOKEN") or cfg.get("hf_token", "")
    repo = os.environ.get("HF_REPO") or cfg.get("hf_repo", "")
    return token or None, repo or None


def is_enabled(cfg: dict) -> bool:
    token, repo = _get_credentials(cfg)
    return bool(token and repo)


def _api(token: str):
    from huggingface_hub import HfApi
    return HfApi(token=token)


def ensure_repo(cfg: dict) -> bool:
    """Create the HF repo if it doesn't exist yet. Returns True on success."""
    token, repo = _get_credentials(cfg)
    if not token or not repo:
        return False
    try:
        api = _api(token)
        api.create_repo(repo_id=repo, private=True, exist_ok=True, repo_type="model")
        log.info("HF repo ready: %s", repo)
        return True
    except Exception as e:
        log.warning("Could not create HF repo: %s", e)
        return False


def push(cfg: dict, paths: list[Path], commit_message: str = "checkpoint") -> bool:
    """Upload files to HF Hub. paths are local paths under data/."""
    token, repo = _get_credentials(cfg)
    if not token or not repo:
        return False
    with _SYNC_LOCK:
        try:
            api = _api(token)
            for path in paths:
                if not path.exists():
                    continue
                path_in_repo = str(path)   # preserves data/models/best.pt etc.
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=path_in_repo,
                    repo_id=repo,
                    repo_type="model",
                    commit_message=commit_message,
                )
                log.info("Pushed %s → %s/%s", path, repo, path_in_repo)
            return True
        except Exception as e:
            log.warning("HF push failed: %s", e)
            return False


def pull(cfg: dict, paths: list[str]) -> bool:
    """Download files from HF Hub into their local paths."""
    token, repo = _get_credentials(cfg)
    if not token or not repo:
        return False
    with _SYNC_LOCK:
        try:
            from huggingface_hub import hf_hub_download
            for path_in_repo in paths:
                local_path = Path(path_in_repo)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    downloaded = hf_hub_download(
                        repo_id=repo,
                        filename=path_in_repo,
                        repo_type="model",
                        token=token,
                        local_dir=".",
                        local_dir_use_symlinks=False,
                    )
                    log.info("Pulled %s/%s → %s", repo, path_in_repo, downloaded)
                except Exception as e:
                    log.debug("Could not pull %s: %s", path_in_repo, e)
            return True
        except Exception as e:
            log.warning("HF pull failed: %s", e)
            return False


# ── Convenience wrappers ───────────────────────────────────────────────────────

_CORE_ARTIFACTS = [
    "data/models/best.pt",
    "data/models/latest.pt",
    "data/ouroboros.db",
]


def pull_latest(cfg: dict) -> None:
    """Pull best.pt, latest.pt, and ouroboros.db on startup."""
    if not is_enabled(cfg):
        return
    log.info("Pulling latest artifacts from HF Hub...")
    pull(cfg, _CORE_ARTIFACTS)


def push_checkpoint(cfg: dict, step: int) -> None:
    """Push model weights after a checkpoint save."""
    if not is_enabled(cfg):
        return
    paths = [
        Path("data/models/best.pt"),
        Path("data/models/latest.pt"),
        Path(f"data/models/ckpt_{step}.pt"),
    ]
    push(cfg, [p for p in paths if p.exists()],
         commit_message=f"checkpoint step {step}")


def push_db(cfg: dict) -> None:
    """Push the database (opponent profiles + game history)."""
    if not is_enabled(cfg):
        return
    push(cfg, [Path("data/ouroboros.db")], commit_message="db update")


def push_promotion(cfg: dict, step: int) -> None:
    """Push after a ladder promotion (new best.pt)."""
    if not is_enabled(cfg):
        return
    push(cfg, [Path("data/models/best.pt")],
         commit_message=f"promoted best at step {step}")


# ── Background periodic DB sync ────────────────────────────────────────────────

class PeriodicSync:
    """Pushes the DB every N minutes in the background."""

    def __init__(self, cfg: dict, interval_minutes: float = 15):
        self.cfg = cfg
        self.interval = interval_minutes * 60
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not is_enabled(self.cfg):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("Periodic HF sync started (every %.0f min)", self.interval / 60)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            push_db(self.cfg)

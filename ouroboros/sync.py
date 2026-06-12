"""Push/pull learning artifacts to a remote store.

Supported backends (first one configured wins):
  1. Google Drive  — set GDRIVE_KEY + GDRIVE_FOLDER_ID
  2. Hugging Face  — set HF_TOKEN + HF_REPO

Synced:  best.pt, latest.pt, ckpt_*.pt, ouroboros.db
Skipped: replay buffer (large, regenerates fast), config.json (contains token)

Size reality check:
  small-profile best.pt ≈ 10 MB   medium ≈ 60 MB   large ≈ 380 MB
  ouroboros.db           < 100 MB
  All artifacts combined < 1 GB — Google Drive 15 GB free tier is ample.
"""
import logging
import os
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_SYNC_LOCK = threading.Lock()

# ── Backend detection ──────────────────────────────────────────────────────────

def _backend() -> str:
    """Return 'gdrive', 'hf', or 'none'."""
    if os.environ.get("GDRIVE_KEY") and os.environ.get("GDRIVE_FOLDER_ID"):
        return "gdrive"
    if os.environ.get("HF_TOKEN") and os.environ.get("HF_REPO"):
        return "hf"
    return "none"


def is_enabled(cfg: dict = None) -> bool:
    return _backend() != "none"


# ── HF helpers ─────────────────────────────────────────────────────────────────

def _hf_credentials(cfg: dict) -> tuple[Optional[str], Optional[str]]:
    token = os.environ.get("HF_TOKEN") or (cfg or {}).get("hf_token", "")
    repo  = os.environ.get("HF_REPO")  or (cfg or {}).get("hf_repo", "")
    return token or None, repo or None


def _hf_api(token: str):
    from huggingface_hub import HfApi
    return HfApi(token=token)


def _hf_ensure_repo(cfg: dict) -> bool:
    token, repo = _hf_credentials(cfg)
    if not token or not repo:
        return False
    try:
        _hf_api(token).create_repo(repo_id=repo, private=True,
                                    exist_ok=True, repo_type="model")
        return True
    except Exception as e:
        log.warning("Could not create HF repo: %s", e)
        return False


def _hf_push(cfg: dict, paths: list[Path], message: str) -> bool:
    token, repo = _hf_credentials(cfg)
    if not token or not repo:
        return False
    try:
        api = _hf_api(token)
        for p in paths:
            if not p.exists():
                continue
            api.upload_file(path_or_fileobj=str(p), path_in_repo=str(p),
                            repo_id=repo, repo_type="model",
                            commit_message=message)
            log.info("HF push: %s", p)
        return True
    except Exception as e:
        log.warning("HF push failed: %s", e)
        return False


def _hf_pull(cfg: dict, remote_paths: list[str]) -> bool:
    token, repo = _hf_credentials(cfg)
    if not token or not repo:
        return False
    try:
        from huggingface_hub import hf_hub_download
        for rp in remote_paths:
            local = Path(rp)
            local.parent.mkdir(parents=True, exist_ok=True)
            try:
                hf_hub_download(repo_id=repo, filename=rp, repo_type="model",
                                token=token, local_dir=".",
                                local_dir_use_symlinks=False)
                log.info("HF pull: %s", rp)
            except Exception as e:
                log.debug("HF pull skip %s: %s", rp, e)
        return True
    except Exception as e:
        log.warning("HF pull failed: %s", e)
        return False


# ── GDrive helpers ─────────────────────────────────────────────────────────────

def _gdrive_push(paths: list[Path], message: str) -> bool:
    from ouroboros.sync_gdrive import push as gd_push
    ok = True
    for p in paths:
        if p.exists():
            ok = gd_push(p) and ok
    return ok


def _gdrive_pull(remote_paths: list[str]) -> bool:
    from ouroboros.sync_gdrive import pull as gd_pull
    ok = True
    for rp in remote_paths:
        # remote name = just the filename; local path = full relative path
        ok = gd_pull(Path(rp).name, Path(rp)) and ok
    return ok


# ── Public API ─────────────────────────────────────────────────────────────────

_CORE_ARTIFACTS = [
    "data/models/best.pt",
    "data/models/latest.pt",
    "data/ouroboros.db",
]


def ensure_repo(cfg: dict) -> bool:
    b = _backend()
    if b == "hf":
        return _hf_ensure_repo(cfg)
    return b == "gdrive"   # GDrive folder must be pre-created by the user


def pull_latest(cfg: dict = None) -> None:
    """Pull core artifacts on startup (no-op if no backend configured)."""
    b = _backend()
    if b == "none":
        return
    log.info("Pulling latest artifacts via %s...", b)
    if b == "gdrive":
        _gdrive_pull(_CORE_ARTIFACTS)
    else:
        _hf_pull(cfg or {}, _CORE_ARTIFACTS)


def push_checkpoint(cfg: dict, step: int) -> None:
    """Push model weights after a checkpoint save."""
    b = _backend()
    if b == "none":
        return
    paths = [Path("data/models/best.pt"), Path("data/models/latest.pt"),
             Path(f"data/models/ckpt_{step}.pt")]
    msg = f"checkpoint step {step}"
    if b == "gdrive":
        _gdrive_push([p for p in paths if p.exists()], msg)
    else:
        _hf_push(cfg, [p for p in paths if p.exists()], msg)


def push_promotion(cfg: dict, step: int) -> None:
    """Push after a ladder promotion (new best.pt)."""
    b = _backend()
    if b == "none":
        return
    paths = [Path("data/models/best.pt")]
    msg = f"promoted best at step {step}"
    if b == "gdrive":
        _gdrive_push(paths, msg)
    else:
        _hf_push(cfg, paths, msg)


def push_db(cfg: dict = None) -> None:
    """Push the DB (opponent profiles + game history)."""
    b = _backend()
    if b == "none":
        return
    paths = [Path("data/ouroboros.db")]
    if b == "gdrive":
        _gdrive_push(paths, "db update")
    else:
        _hf_push(cfg or {}, paths, "db update")


# ── Background periodic DB sync ────────────────────────────────────────────────

class PeriodicSync:
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
        log.info("Periodic sync started via %s (every %.0f min)",
                 _backend(), self.interval / 60)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            push_db(self.cfg)

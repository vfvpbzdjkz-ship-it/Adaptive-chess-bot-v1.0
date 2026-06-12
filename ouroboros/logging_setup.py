import logging
import logging.handlers
import os
from pathlib import Path


def setup_logging(log_dir: str = "data/logs", level: int = logging.INFO) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)

    # Rotating file handler — 10 MB per file, keep 5
    fh = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "ouroboros.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    fh.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(fmt, datefmt))
    ch.setLevel(logging.WARNING)

    if not root.handlers:
        root.addHandler(fh)
        root.addHandler(ch)

"""Disk-backed replay buffer using numpy memmap ring arrays.

Stores up to `capacity` positions. Survives restarts via meta.json.
Each sample: (state_planes int8, policy fp16 dense, value fp16, weight fp16, source uint8)
"""
import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

BUFFER_DIR = Path("data/buffer")
META_PATH = BUFFER_DIR / "meta.json"

IN_PLANES = 19
BOARD_SQ = 8 * 8
STATE_SHAPE = (IN_PLANES, 8, 8)
POLICY_SIZE = 4672

SOURCE_SELFPLAY = 0
SOURCE_LICHESS_LIVE = 1
SOURCE_OPP_IMITATION = 2


class ReplayBuffer:
    def __init__(self, capacity: int = 1_000_000):
        self.capacity = capacity
        self._lock = threading.Lock()
        BUFFER_DIR.mkdir(parents=True, exist_ok=True)
        self._states = self._open_or_create(
            BUFFER_DIR / "states.npy",
            (capacity, IN_PLANES, 8, 8),
            np.int8,
        )
        self._policies = self._open_or_create(
            BUFFER_DIR / "policies.npy",
            (capacity, POLICY_SIZE),
            np.float16,
        )
        self._values = self._open_or_create(
            BUFFER_DIR / "values.npy",
            (capacity,),
            np.float16,
        )
        self._weights = self._open_or_create(
            BUFFER_DIR / "weights.npy",
            (capacity,),
            np.float16,
        )
        self._sources = self._open_or_create(
            BUFFER_DIR / "sources.npy",
            (capacity,),
            np.uint8,
        )
        meta = self._load_meta()
        self._write_ptr: int = meta.get("write_ptr", 0)
        self._count: int = meta.get("count", 0)

    def _open_or_create(self, path: Path, shape: tuple, dtype) -> np.ndarray:
        if path.exists():
            return np.memmap(str(path), dtype=dtype, mode="r+", shape=shape)
        arr = np.memmap(str(path), dtype=dtype, mode="w+", shape=shape)
        arr[:] = 0
        arr.flush()
        return arr

    def _load_meta(self) -> dict:
        if META_PATH.exists():
            with open(META_PATH) as f:
                return json.load(f)
        return {}

    def _save_meta(self) -> None:
        tmp = str(META_PATH) + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"write_ptr": self._write_ptr, "count": self._count}, f)
        os.replace(tmp, META_PATH)

    def add(
        self,
        state: np.ndarray,        # (19, 8, 8) float32 or int8
        policy: np.ndarray,        # (4672,) float32
        value: float,
        weight: float = 1.0,
        source: int = SOURCE_SELFPLAY,
    ) -> None:
        with self._lock:
            idx = self._write_ptr % self.capacity
            self._states[idx] = (state * 127).astype(np.int8) if state.dtype != np.int8 else state
            self._policies[idx] = policy.astype(np.float16)
            self._values[idx] = np.float16(value)
            self._weights[idx] = np.float16(weight)
            self._sources[idx] = np.uint8(source)
            self._write_ptr += 1
            self._count = min(self._count + 1, self.capacity)
            if self._write_ptr % 1000 == 0:
                self._flush_and_save()

    def add_batch(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
        weights: Optional[np.ndarray] = None,
        source: int = SOURCE_SELFPLAY,
    ) -> None:
        n = len(states)
        if weights is None:
            weights = np.ones(n, dtype=np.float32)
        for i in range(n):
            self.add(states[i], policies[i], float(values[i]), float(weights[i]), source)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (states, policies, values, weights) as float32 arrays."""
        with self._lock:
            count = self._count
        if count == 0:
            raise RuntimeError("Buffer is empty")
        count = min(count, self.capacity)

        # Weighted sampling using stored weights
        w = self._weights[:count].astype(np.float32)
        w = np.clip(w, 0, None)
        w_sum = w.sum()
        if w_sum <= 0:
            probs = np.ones(count) / count
        else:
            probs = w / w_sum

        idxs = np.random.choice(count, size=batch_size, replace=True, p=probs)

        states = self._states[idxs].astype(np.float32) / 127.0
        policies = self._policies[idxs].astype(np.float32)
        values = self._values[idxs].astype(np.float32)
        weights = self._weights[idxs].astype(np.float32)
        return states, policies, values, weights

    def _flush_and_save(self) -> None:
        self._states.flush()
        self._policies.flush()
        self._values.flush()
        self._weights.flush()
        self._sources.flush()
        self._save_meta()

    def flush(self) -> None:
        with self._lock:
            self._flush_and_save()

    @property
    def count(self) -> int:
        return self._count

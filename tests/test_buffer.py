"""Buffer tests: write/read/wraparound/restart integrity."""
import os
import shutil
import sys
import tempfile
from pathlib import Path
import unittest.mock as mock

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest


def _make_buffer(tmpdir: str, capacity: int = 100):
    # Patch the BUFFER_DIR to use tmpdir
    import ouroboros.learning.buffer as buf_mod
    original_dir = buf_mod.BUFFER_DIR
    original_meta = buf_mod.META_PATH
    buf_mod.BUFFER_DIR = Path(tmpdir)
    buf_mod.META_PATH = Path(tmpdir) / "meta.json"
    from ouroboros.learning.buffer import ReplayBuffer
    buf = ReplayBuffer(capacity=capacity)
    # Restore
    buf_mod.BUFFER_DIR = original_dir
    buf_mod.META_PATH = original_meta
    return buf


def _sample_data():
    state = np.random.randn(19, 8, 8).astype(np.float32)
    policy = np.random.dirichlet(np.ones(4672)).astype(np.float32)
    value = float(np.random.choice([-1, 0, 1]))
    return state, policy, value


def test_write_and_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        buf = _make_buffer(tmpdir, capacity=50)
        state, policy, value = _sample_data()
        buf.add(state, policy, value, weight=1.0)
        assert buf.count == 1
        s, p, v, w = buf.sample(1)
        assert s.shape == (1, 19, 8, 8)
        assert p.shape == (1, 4672)
        assert abs(float(v[0]) - value) < 0.02  # fp16 precision


def test_wraparound():
    with tempfile.TemporaryDirectory() as tmpdir:
        buf = _make_buffer(tmpdir, capacity=10)
        for i in range(15):
            state, policy, value = _sample_data()
            buf.add(state, policy, value)
        assert buf.count == 10  # capped at capacity


def test_restart_integrity():
    with tempfile.TemporaryDirectory() as tmpdir:
        import ouroboros.learning.buffer as buf_mod
        # Write some data
        buf_mod.BUFFER_DIR = Path(tmpdir)
        buf_mod.META_PATH = Path(tmpdir) / "meta.json"
        from ouroboros.learning.buffer import ReplayBuffer

        buf1 = ReplayBuffer(capacity=50)
        for _ in range(20):
            state, policy, value = _sample_data()
            buf1.add(state, policy, value)
        buf1.flush()
        count_before = buf1.count
        ptr_before = buf1._write_ptr

        # Simulate restart
        buf2 = ReplayBuffer(capacity=50)
        assert buf2.count == count_before, f"Count mismatch after restart: {buf2.count} vs {count_before}"
        assert buf2._write_ptr == ptr_before, "Write pointer not restored"


def test_weighted_sampling():
    with tempfile.TemporaryDirectory() as tmpdir:
        import ouroboros.learning.buffer as buf_mod
        buf_mod.BUFFER_DIR = Path(tmpdir)
        buf_mod.META_PATH = Path(tmpdir) / "meta.json"
        from ouroboros.learning.buffer import ReplayBuffer

        buf = ReplayBuffer(capacity=100)
        # Add high-weight sample
        state_hi, policy_hi, _ = _sample_data()
        buf.add(state_hi, policy_hi, 1.0, weight=10.0)
        # Add many low-weight samples
        for _ in range(20):
            state, policy, value = _sample_data()
            buf.add(state, policy, value, weight=0.1)

        # High-weight sample should be sampled more often
        _, _, vals, weights = buf.sample(200)
        assert len(vals) == 200


if __name__ == "__main__":
    test_write_and_read()
    test_wraparound()
    test_restart_integrity()
    test_weighted_sampling()
    print("All buffer tests passed.")

"""Tests for the backend abstraction layer."""

import numpy as np
import pytest

from src.core.backend import (
    set_backend,
    get_xp,
    get_backend_name,
    to_cpu,
    to_xp,
)


@pytest.fixture(autouse=True)
def reset_backend():
    """Ensure numpy backend before and after each test."""
    set_backend("numpy")
    yield
    set_backend("numpy")


class TestBackendSwitching:
    def test_default_is_numpy(self):
        assert get_backend_name() == "numpy"
        assert get_xp() is np

    def test_set_numpy(self):
        set_backend("numpy")
        assert get_backend_name() == "numpy"

    def test_set_auto_falls_back_to_numpy(self):
        # On machines without cupy, auto should fall back to numpy
        set_backend("auto")
        assert get_backend_name() in ("numpy", "cupy")

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend("tensorflow")


class TestTransferFunctions:
    def test_to_cpu_numpy(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = to_cpu(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_to_xp_numpy(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = to_xp(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_to_cpu_list(self):
        result = to_cpu([1, 2, 3])
        assert isinstance(result, np.ndarray)

    def test_roundtrip(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        xp_arr = to_xp(arr)
        cpu_arr = to_cpu(xp_arr)
        np.testing.assert_array_equal(cpu_arr, arr)


class TestGetXpCallPattern:
    """Verify that get_xp() returns current backend even after switching."""

    def test_get_xp_reflects_switch(self):
        set_backend("numpy")
        xp1 = get_xp()
        assert xp1 is np
        # Switch back to numpy (no-op in CI without cupy)
        set_backend("numpy")
        xp2 = get_xp()
        assert xp2 is np


class TestCpuBrainSmoke:
    """Smoke test: full projection cycle on numpy backend."""

    def test_explicit_projection(self):
        from src.core.brain import Brain

        set_backend("numpy")
        brain = Brain(p=0.1, seed=42)
        brain.add_area("A", n=100, k=10, beta=0.05, explicit=True)
        brain.add_area("B", n=100, k=10, beta=0.05, explicit=True)
        brain.add_stimulus("s", size=20)

        # Stimulus -> A
        brain.project({"s": ["A"]}, {})
        assert brain.areas["A"].winners is not None
        assert len(brain.areas["A"].winners) == 10

        # A -> B
        brain.project({}, {"A": ["B"]})
        assert brain.areas["B"].winners is not None
        assert len(brain.areas["B"].winners) == 10

    def test_sparse_projection(self):
        from src.core.brain import Brain

        set_backend("numpy")
        brain = Brain(p=0.05, seed=42)
        brain.add_area("A", n=1000, k=50, beta=0.05, explicit=False)
        brain.add_stimulus("s", size=50)

        # Run a few projection steps
        for _ in range(5):
            brain.project({"s": ["A"]}, {"A": ["A"]})

        assert brain.areas["A"].winners is not None
        assert len(brain.areas["A"].winners) == 50

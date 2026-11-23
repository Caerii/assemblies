"""Local protocol unit tests; native numerical gate is a separate execution."""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from research.multimodal import native_retention as NR


@pytest.mark.parametrize("cfg", [
    NR.NativeConfig(n=4, k=5),
    NR.NativeConfig(p=0),
    NR.NativeConfig(beta=-.1),
    NR.NativeConfig(loads=(8, 4)),
    NR.NativeConfig(loads=(4, 4)),
    NR.NativeConfig(seeds=(1, 1)),
    NR.NativeConfig(arms=("invalid",)),
    NR.NativeConfig(read_rounds=0),
])
def test_invalid_config(cfg):
    with pytest.raises(ValueError):
        cfg.validate()


def test_overlap_and_winner_ids():
    assert NR.overlap(np.array([1, 2]), np.array([2, 3])) == .5
    np.testing.assert_array_equal(NR.canonical([4, 0], 2), [0, 4])
    with pytest.raises(AssertionError):
        NR.canonical([2, 2], 2)


def test_protocol_scopes_are_used(monkeypatch):
    calls = []

    class FakeBrain:
        def __init__(self, **kwargs):
            self.seed = kwargs["seed"]
            self.reading = 0
            self.frozen_state = 0

        def add_area(self, *args, **kwargs):
            pass

        def add_stimulus(self, *args, **kwargs):
            pass

        def clear_activity(self, areas):
            assert areas == ["memory"]
            calls.append(("clear", bool(self.reading), bool(self.frozen_state)))

        @contextmanager
        def read_only(self, *, seed):
            assert isinstance(seed, int)
            self.reading += 1
            try:
                yield self
            finally:
                self.reading -= 1

        @contextmanager
        def frozen(self):
            assert self.reading
            self.frozen_state += 1
            try:
                yield self
            finally:
                self.frozen_state -= 1

    def project(brain, stimulus, target, rounds, recurrent):
        assert target == "memory"
        assert rounds >= 1
        reading = bool(brain.reading)
        assert not reading or bool(brain.frozen_state)
        idx = int(stimulus.split("_")[-1])
        # Fake stable, deterministic code; this is NOT a simulated brain.
        n = 32
        winners = np.array([(idx * 3 + j) % n for j in range(4)], dtype=np.uint32)
        calls.append(("project", reading, recurrent))
        return SimpleNamespace(winners=winners)

    monkeypatch.setattr(NR, "_native", lambda: (FakeBrain, project))
    cfg = NR.NativeConfig(n=32, k=4, loads=(2, 4), seeds=(0, 1))
    first = NR.run(cfg)
    assert first["status"] == "native_brain_identity_retention_not_binding"
    assert len(first["records"]) == 2 * 3 * 2
    assert all(row["mean_overlap"] == 1. for row in first["records"])
    assert all(row["exact_recovery"] == 1. for row in first["records"])
    assert any(row[0] == "clear" and row[1] and row[2] for row in calls)
    assert all(not row[1] or row[2] for row in calls if row[0] == "clear")
    assert any(row[0] == "project" and row[1] for row in calls)
    assert any(row[0] == "project" and not row[1] for row in calls)


def test_read_only_contract_documented():
    cfg = NR.NativeConfig()
    assert cfg.validate() == cfg
    assert cfg.arms == ("recurrent", "feedforward", "no_plasticity")

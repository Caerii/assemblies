"""Contract tests for paired refraction readout (not native numerical evidence)."""
from contextlib import contextmanager

import numpy as np
import pytest

from research.multimodal import refraction_paired as R


@pytest.mark.parametrize("cfg", [
    R.PairedConfig(n=4, k=5),
    R.PairedConfig(p=0),
    R.PairedConfig(k=7, stimulus_parts=2),
    R.PairedConfig(refraction_factor=2),
    R.PairedConfig(loads=(8, 8)),
    R.PairedConfig(loads=(0,)),
    R.PairedConfig(seeds=(1, 1)),
])
def test_rejects_invalid_protocol(cfg):
    with pytest.raises(ValueError):
        cfg.validate()


class FakeBrain:
    def __init__(self):
        self.flags = []
        self.reads = 0
        self.frozen_count = 0

    def set_masked_readout(self, area, flag):
        assert area == "memory"
        self.flags.append(flag)

    @contextmanager
    def read_only(self, *, seed):
        assert seed >= 0
        self.reads += 1
        try:
            yield self
        finally:
            self.reads -= 1

    @contextmanager
    def frozen(self):
        assert self.reads
        self.frozen_count += 1
        try:
            yield self
        finally:
            self.frozen_count -= 1


class FakeEngine:
    def __init__(self, brain, fail=False):
        self.brain = brain
        self.current = np.array([0, 1], dtype=np.int64)
        self.fail = fail

    def set_winners(self, area, value):
        assert area == "memory"
        assert self.brain.reads and self.brain.frozen_count
        self.current = np.asarray(value).copy()

    def project_into(self, area, stimuli, fibers, *, plasticity_enabled):
        assert area == "memory" and stimuli == [] and fibers == ["memory"]
        assert not plasticity_enabled
        assert self.brain.reads and self.brain.frozen_count
        if self.fail:
            raise RuntimeError("projection failed")
        self.current = np.array([0, 1], dtype=np.int64)

    def get_winners(self, area):
        return self.current


def test_readout_scoped_and_mask_reset():
    brain = FakeBrain()
    eng = FakeEngine(brain)
    stored = [np.array([0, 1]), np.array([2, 3])]
    hit, overlap = R._recall_one(brain, eng, stored, 0, True, 2, 71)
    assert hit and overlap == 1.
    assert brain.flags == [True, False]
    assert brain.reads == 0 and brain.frozen_count == 0


def test_mask_reset_even_on_exception():
    brain = FakeBrain()
    eng = FakeEngine(brain, fail=True)
    stored = [np.array([0, 1]), np.array([2, 3])]
    with pytest.raises(RuntimeError):
        R._recall_one(brain, eng, stored, 0, True, 2, 71)
    assert brain.flags == [True, False]
    assert brain.reads == 0 and brain.frozen_count == 0


def test_paired_uses_same_sample_and_seed(monkeypatch):
    pairs = []
    class MinimalBrain:
        def __init__(self, **kwargs):
            self.areas = {"memory": object()}
        def add_area(self, *args, **kwargs): pass
        def materialize_area(self, *args, **kwargs): pass
        def set_refracted(self, *args, **kwargs): pass
        def add_stimulus(self, *args, **kwargs): pass
        def clear_activity(self, *args, **kwargs): pass
        def _engine_for(self, *args, **kwargs): return self.engine

    class MinimalEngine:
        def __init__(self): self.i = 0
        def project_into(self, *args, **kwargs): self.i += 1
        def get_winners(self, *args): return [0, 1, 2, 3]

    def build(**kwargs):
        b = MinimalBrain(**kwargs)
        b.engine = MinimalEngine()
        return b

    def fake_recall(brain, eng, stored, item, masked, rounds, seed):
        pairs.append((len(stored), item, masked, seed))
        return (True, 1.0)

    monkeypatch.setattr(R, "_native", lambda: build)
    monkeypatch.setattr(R, "_recall_one", fake_recall)
    cfg = R.PairedConfig(n=16, k=4, stimulus_parts=2, loads=(2, 4),
                         recall_sample=2, seeds=(1,))
    rows = R.trial(1, True, cfg)
    assert len(rows) == 2
    assert all(r["paired_delta_rank1"] == 0 for r in rows)
    for load in (2, 4):
        net = {(j, seed) for m, j, masked, seed in pairs if m == load and not masked}
        masked = {(j, seed) for m, j, mode, seed in pairs if m == load and mode}
        assert net == masked

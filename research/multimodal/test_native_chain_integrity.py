"""Multi-step native adapter integrity tests. Fake native Brain only."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from research.multimodal.native_affine_chain import NativeAffineChain
from research.multimodal.noncommutative_composition import Protocol, public_view


def inject(monkeypatch, fake_type):
    monkeypatch.setitem(sys.modules, "research.multimodal.native_teacher_drive",
                        SimpleNamespace(NativeTeacherDriven=fake_type))
    monkeypatch.setitem(sys.modules, "research.multimodal.zero_shot_gate",
                        SimpleNamespace(Config=lambda **kwargs: SimpleNamespace(**kwargs)))
    return NativeAffineChain(public_view(Protocol(n=64, k=8, data_seed=3,
                                                      model_seed=901,
                                                      atomic_examples=3,
                                                      test_cues=3)))


class Fake:
    def __init__(self, cfg, **kwargs):
        assert cfg.seed == 901
        self.calls = []
        self.ledger = 0

    def state_fingerprint(self):
        return str(self.ledger)

    def train(self, index, role, cue, target):
        self.calls.append(("train", role))
        self.ledger += 1

    def predict(self, index, role, cue):
        self.calls.append(("predict", role))
        return np.sort((cue + (1 if role == "A" else 2)) % 64)


def test_order_and_unchanged_full_graph(monkeypatch):
    m = inject(monkeypatch, Fake)
    cue = np.arange(8, dtype=np.uint32)
    src = cue.copy()
    before = m.fingerprint()
    answer = m.predict(("B", "A"), cue)
    np.testing.assert_array_equal(answer, np.sort((cue + 3) % 64))
    np.testing.assert_array_equal(src, cue)
    assert m.inner.calls == [("predict", "B"), ("predict", "A")]
    assert m.fingerprint() == before


def test_detects_first_mutation_even_if_later_step_would_restore(monkeypatch):
    class MutateThenRestore(Fake):
        def predict(self, index, role, cue):
            self.ledger += 1 if role == "A" else -1
            return super().predict(index, role, cue)

    model = inject(monkeypatch, MutateThenRestore)
    with pytest.raises(AssertionError, match="step 0"):
        model.predict(("A", "B"), np.arange(8, dtype=np.uint32))
    assert model.inner.calls == [("predict", "A")]


def test_detects_late_mutation(monkeypatch):
    class LateWrite(Fake):
        def predict(self, index, role, cue):
            if role == "B":
                self.ledger += 1
            return super().predict(index, role, cue)

    model = inject(monkeypatch, LateWrite)
    with pytest.raises(AssertionError, match=r"step 1 \\(B\\)"):
        model.predict(("A", "B"), np.arange(8, dtype=np.uint32))


def test_invalid_intermediate_output_fails_before_next_step(monkeypatch):
    class Broken(Fake):
        def predict(self, index, role, cue):
            self.calls.append(("predict", role))
            return np.zeros(8, dtype=np.uint32)

    model = inject(monkeypatch, Broken)
    with pytest.raises(ValueError, match="invalid k-of-n"):
        model.predict(("A", "B"), np.arange(8, dtype=np.uint32))
    assert model.inner.calls == [("predict", "A")]


def test_rejects_bad_input_without_calling_native(monkeypatch):
    model = inject(monkeypatch, Fake)
    for bad in ([0] * 8, [-1] + list(range(1, 8)), list(range(9))):
        with pytest.raises(ValueError):
            model.predict(("A",), bad)
    with pytest.raises(ValueError, match="known role"):
        model.predict(("C",), np.arange(8, dtype=np.uint32))
    with pytest.raises(ValueError, match="unknown atomic role"):
        model.train("C", np.arange(8), np.arange(8))
    assert model.inner.calls == []


def test_train_uses_copied_valid_assemblies(monkeypatch):
    class MutatingTrainer(Fake):
        def train(self, index, role, cue, target):
            cue[:] = 0
            target[:] = 1
            return super().train(index, role, cue, target)

    model = inject(monkeypatch, MutatingTrainer)
    cue, target = np.arange(8, dtype=np.uint32), np.arange(8, 16, dtype=np.uint32)
    model.train("A", cue, target)
    np.testing.assert_array_equal(cue, np.arange(8, dtype=np.uint32))
    np.testing.assert_array_equal(target, np.arange(8, 16, dtype=np.uint32))
    assert model.writes == 1


def test_none_early_stop_has_no_later_predict(monkeypatch):
    class EarlyStop(Fake):
        def predict(self, index, role, cue):
            self.calls.append(("predict", role))
            return None

    model = inject(monkeypatch, EarlyStop)
    assert model.predict(("A", "B"), np.arange(8, dtype=np.uint32)) is None
    assert model.inner.calls == [("predict", "A")]

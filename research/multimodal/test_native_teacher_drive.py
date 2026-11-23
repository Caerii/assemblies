"""R1h protocol tests with fake Brain; no numerical native claims."""
from contextlib import contextmanager
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from research.multimodal import native_teacher_drive as M
from research.multimodal.zero_shot_gate import Config


def test_supervision_only_during_training(monkeypatch):
    events = []
    class Brain:
        def __init__(self, **kwargs):
            self.reading = False
            self.frozen_state = False
            self.areas = {}
            self.fibers = {}
            self.current_cue = np.array([], dtype=np.uint32)
        def add_area(self, name, **kwargs):
            assert kwargs["explicit"] is True
            self.areas[name] = SimpleNamespace(winners=np.array([], dtype=np.uint32))
            if name != "LEX":
                self.fibers[name] = SimpleNamespace(
                    weights=np.ones((kwargs["n"], kwargs["n"]), dtype=np.float32))
            events.append(("area", name))
        def engine_for(self, role):
            return SimpleNamespace(_area_conns={"LEX": {role: self.fibers[role]}})
        def clear_activity(self, names):
            self.areas[names[0]].winners = np.array([], dtype=np.uint32)
            events.append(("clear", tuple(names)))
        def project(self, a, b, external_drive=None):
            if self.reading:
                assert self.frozen_state and external_drive is None
            role = b["LEX"][0]
            if external_drive is not None:
                target_ids = np.flatnonzero(external_drive[role])
                self.areas[role].winners = target_ids.astype(np.uint32)
                matrix = self.fibers[role].weights
                matrix[np.ix_(self.current_cue, target_ids)] *= 1.1
            events.append(("project", tuple(b["LEX"]), external_drive))
        @contextmanager
        def read_only(self, *, seed):
            self.reading = True
            try:
                yield self
            finally:
                self.reading = False
        @contextmanager
        def frozen(self):
            assert self.reading
            self.frozen_state = True
            try:
                yield self
            finally:
                self.frozen_state = False

    class Assembly:
        def __init__(self, area, ids):
            self.area = area
            self.neuron_ids = ids
        @staticmethod
        def from_area(brain, role):
            assert brain.reading and brain.frozen_state
            return Assembly(role, np.array([1, 2, 3], dtype=np.uint32))
    def activate(brain, assembly):
        brain.current_cue = np.asarray(assembly.neuron_ids, dtype=np.uint32).copy()
        events.append(("activate", assembly.area))
    for name, value in {
        "neural_assemblies.core.brain": SimpleNamespace(Brain=Brain),
        "neural_assemblies.core.index_spaces": SimpleNamespace(NeuronIds=lambda x: x),
        "neural_assemblies.assembly_calculus.assembly": SimpleNamespace(Assembly=Assembly),
        "neural_assemblies.assembly_calculus.ops": SimpleNamespace(activate_assembly=activate),
    }.items():
        monkeypatch.setitem(sys.modules, name, value)

    model = M.NativeTeacherDriven(Config(n=20, k=3, contents=4),
                                  teacher_gain=50., presentations=2)
    cue = np.array([1, 2, 3], dtype=np.uint32)
    target = np.array([4, 5, 6], dtype=np.uint32)
    model.train(0, "SUBJ", cue, target)
    assert model.training_calls == 1
    assert model.train_pairs == {(0, "SUBJ")}
    assert model.clamp_diagnostics[0]["teacher_target_overlap"] == 1.
    assert model.clamp_diagnostics[0]["active_edge_changed"] == 9
    before_read_hash = model.state_fingerprint()
    train_projections = [e for e in events if e[0] == "project"]
    assert len(train_projections) == 2
    for _, _, drive in train_projections:
        np.testing.assert_array_equal(np.flatnonzero(drive["SUBJ"]), target)
        assert np.all(drive["SUBJ"][target] == 50.)
    assert model.predict(0, "SUBJ", cue).tolist() == [1, 2, 3]
    assert model.state_fingerprint() == before_read_hash
    assert events[-1] == ("project", ("SUBJ",), None)


@pytest.mark.parametrize("gain,presentations", [(0, 1), (-1, 1), (float("nan"), 1),
                                               (10, 0), (10, True)])
def test_invalid_training_configuration(gain, presentations):
    with pytest.raises(ValueError):
        M.NativeTeacherDriven(Config(), teacher_gain=gain,
                              presentations=presentations)

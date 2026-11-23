"""Native adapter scheduling tests. Mocks are not native numerical evidence."""
from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
import pytest
from research.multimodal import native_zero_shot as N
from research.multimodal.zero_shot_gate import Config


def test_native_adapter_does_not_receive_hidden_teacher(monkeypatch):
    events = []
    class Brain:
        def __init__(self, **kw):
            self.winners = {}
            self.is_read = False
            self.is_frozen = False
        def add_area(self, name, **kw):
            events.append(("area", name))
        def materialize_area(self, name, storage):
            events.append(("materialize", name, storage))
        def clear_activity(self, names):
            events.append(("clear", tuple(names)))
        def project(self, a, b):
            events.append(("project", tuple(b["LEX"])))
        @contextmanager
        def read_only(self, *, seed):
            self.is_read = True
            try:
                yield
            finally:
                self.is_read = False
        @contextmanager
        def frozen(self):
            self.is_frozen = True
            try:
                yield
            finally:
                self.is_frozen = False

    import sys
    import types
    modules = {
        "neural_assemblies.core.brain": SimpleNamespace(Brain=Brain),
        "neural_assemblies.core.index_spaces": SimpleNamespace(NeuronIds=lambda x: x),
    }
    class Assembly:
        def __init__(self, area, ids):
            self.area, self.neuron_ids = area, ids
        @staticmethod
        def from_area(brain, role):
            assert brain.is_read and brain.is_frozen
            return Assembly(role, np.array([0, 1, 2], dtype=np.uint32))
    modules["neural_assemblies.assembly_calculus.assembly"] = SimpleNamespace(Assembly=Assembly)
    def activate(brain, assembly):
        events.append(("activate", assembly.area))
    modules["neural_assemblies.assembly_calculus.ops"] = SimpleNamespace(activate_assembly=activate)
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)

    cfg = Config(n=20, k=3, contents=4)
    model = N.NativeProjectionBaseline(cfg)
    cue = np.array([3, 4, 5], dtype=np.uint32)
    target = np.array([10, 11, 12], dtype=np.uint32)
    model.train(0, "SUBJ", cue, target)
    assert model.writes == 1
    assert model.predict(0, "SUBJ", cue).tolist() == [0, 1, 2]
    assert events.count(("activate", "LEX")) == 2
    assert events.count(("project", ("SUBJ",))) == 2
    assert len([x for x in events if x[0] == "materialize"]) == 3


def test_config_validated_before_native_import():
    with pytest.raises(ValueError):
        N.run(Config(n=2, k=3))

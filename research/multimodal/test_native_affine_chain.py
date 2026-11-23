"""Native bridge contracts using injected fake. No native numerical evidence."""
import numpy as np
import sys
from types import SimpleNamespace
from research.multimodal import native_affine_chain as N
from research.multimodal.noncommutative_composition import Protocol, PrivateDataset, public_view


def test_native_adapter_uses_atomic_training_and_preserves_role_order(monkeypatch):
    events = []
    class FakeNative:
        def __init__(self, config, **kw):
            assert config.seed == 901  # PUBLIC learner seed
            self.seed = config.seed
            self.writes = 0
            self.ledger = []
        def train(self, index, role, cue, target):
            events.append(('train', index, role))
            self.ledger.append((index, role, tuple(cue), tuple(target)))
            self.writes += 1
        def predict(self, index, role, cue):
            events.append(('predict', index, role))
            return np.sort((np.asarray(cue,dtype=np.uint32) + (1 if role=='A' else 2)) % 64)
        def state_fingerprint(self):
            return str(self.ledger)
    monkeypatch.setitem(sys.modules, 'research.multimodal.native_teacher_drive',
                        SimpleNamespace(NativeTeacherDriven=FakeNative))
    monkeypatch.setitem(sys.modules, 'research.multimodal.zero_shot_gate',
                        SimpleNamespace(Config=lambda **kw: SimpleNamespace(**kw)))
    spec = Protocol(n=64, k=8, data_seed=3, model_seed=901, atomic_examples=3, test_cues=3)
    model = N.NativeAffineChain(public_view(spec))
    d = PrivateDataset(spec)
    for role in spec.roles:
        for cue in d.atomic[role]:
            model.train(role, cue, d.target(cue,(role,)))
    assert model.writes == 6
    assert [e[0] for e in events] == ['train']*6
    first = d.test[0]
    before = model.fingerprint()
    out = model.predict(('B','A'), first)
    assert events[-2:] == [('predict', 0, 'B'), ('predict', 0, 'A')]
    assert np.array_equal(out, np.sort((first+3) % 64))
    assert before == model.fingerprint()


def test_optional_native_evaluator_does_not_train_composites(monkeypatch):
    events=[]
    class Fake:
        def __init__(self, config, **kwargs): self.count=0
        def train(self, index, role, cue, target):
            self.count += 1
            events.append(('train',role))
        def predict(self, index, role, cue):
            return np.asarray(cue,dtype=np.uint32).copy()
        def state_fingerprint(self): return str(self.count)
    monkeypatch.setitem(sys.modules, 'research.multimodal.native_teacher_drive',
                        SimpleNamespace(NativeTeacherDriven=Fake))
    monkeypatch.setitem(sys.modules, 'research.multimodal.zero_shot_gate',
                        SimpleNamespace(Config=lambda **kw: SimpleNamespace(**kw)))
    out = N.run_native(Protocol(n=32,k=5,data_seed=3,model_seed=901,
                                atomic_examples=3,test_cues=3))
    assert out['status'] == 'native_supervised_affine_composition_smoke'
    assert len(events) == 6
    assert all(e[0]=='train' and e[1] in ('A','B') for e in events)
    assert out['dataset']['composite_writes'] == 0
    assert len(out['summary']) == 3

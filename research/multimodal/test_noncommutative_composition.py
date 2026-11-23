"""R1o contracts: teach atomics, independently test noncommutative sequences."""
from dataclasses import replace
import numpy as np
import pytest
from research.multimodal import noncommutative_composition as M


@pytest.mark.parametrize('seed', [0, 1, 17, 23, 42])
def test_noncommutative_private_operator_and_split(seed):
    p = M.Protocol(data_seed=seed)
    d = M.PrivateDataset(p)
    train = [tuple(x) for role in p.roles for x in d.atomic[role]]
    test = [tuple(x) for x in d.test]
    assert len(train) == p.atomic_examples * 2
    assert len(set(train + test)) == len(train) + len(test)
    assert all(not np.array_equal(d.target(x, ('A', 'B')),
                                      d.target(x, ('B', 'A'))) for x in d.test)
    assert d.manifest()['composite_writes'] == 0
    assert d.manifest() == M.PrivateDataset(p).manifest()


@pytest.mark.parametrize('seed', [0, 1, 23, 42])
def test_structural_law_recovered_without_composite_writes(seed):
    p = M.Protocol(data_seed=seed, n=64, k=8, atomic_examples=8, test_cues=6)
    r = M.run_one(p)
    score = {(x['model'], x['partition']): x['exact'] for x in r['summary']}
    assert score['affine_rule', 'atomic_train'] == 1.0
    assert score['affine_rule', 'heldout_atomic'] == 1.0
    assert score['affine_rule', 'heldout_composed'] == 1.0
    assert score['order_blind', 'heldout_composed'] < 1.0
    assert score['hebbian_pairs', 'atomic_train'] == 1.0
    assert score['hebbian_pairs', 'heldout_composed'] == 0.0
    assert score['atomic_lookup', 'heldout_composed'] == 0.0
    assert score['copy_input', 'heldout_composed'] == 0.0
    assert r == M.run_one(p)


def test_order_blind_misses_noncommuting_path_even_with_correct_atomic_learning():
    p = M.Protocol()
    d = M.PrivateDataset(p)
    m = M.OrderBlindAffine(M.public_view(p))
    for role in p.roles:
        for cue in d.atomic[role]:
            m.train(role, cue, d.target(cue, (role,)))
    x = d.test[0]
    assert np.array_equal(m.predict(('A', 'B'), x), d.target(x, ('A', 'B')))
    assert not np.array_equal(m.predict(('B', 'A'), x), d.target(x, ('B', 'A')))


def test_model_cannot_receive_private_seed_or_hidden_parameters():
    p = M.Protocol(data_seed=79, model_seed=1901)
    m = M.public_view(p)
    assert m.seed == 1901
    assert not hasattr(m, 'data_seed')
    assert not hasattr(m, '_params')
    assert not hasattr(m, 'atomic')
    assert not hasattr(m, 'test')


def test_prediction_does_not_mutate_stored_evidence():
    cfg = M.PublicSpec(64, 8, ('A', 'B'), 123)
    cue = np.array([0,1,2,3,4,5,6,7], dtype=np.uint32)
    dst = np.sort((cue * 3 + 9) % 64)
    for cls in (M.AffineRuleLearner, M.ShiftOnly, M.HebbianPairs, M.AtomicKeyMemory, M.CopyInput):
        model = cls(cfg)
        model.train('A', cue, dst)
        before = model.fingerprint()
        value = model.predict(('A','B'), cue.copy())
        if value is not None:
            assert len(value) == 8
        assert model.fingerprint() == before


def test_train_only_atomic_examples(monkeypatch):
    trained, predicted = [], []
    class Spy:
        def __init__(self, public):
            assert isinstance(public, M.PublicSpec)
            self.count = 0
        def train(self, role, cue, target):
            assert isinstance(role, str)
            assert len(cue) == len(target) == 5
            trained.append((role, tuple(cue)))
            self.count += 1
        def predict(self, roles, cue):
            predicted.append(tuple(roles))
            return None
        def fingerprint(self):
            return str(self.count)
        def bytes_state(self):
            return 0
    monkeypatch.setattr(M, 'FACTORIES', (('spy', Spy),))
    result = M.run_one(M.Protocol(n=32,k=5,atomic_examples=4,test_cues=3))
    assert len(trained) == 8
    assert len({(role, cue) for role, cue in trained}) == 8
    assert all(len(role) == 1 for role, _ in trained)
    assert any(len(seq) == 4 for seq in predicted)
    assert result['dataset']['composite_writes'] == 0


@pytest.mark.parametrize('bad', [
    M.Protocol(n=8), M.Protocol(k=1), M.Protocol(k=64),
    M.Protocol(data_seed=4, model_seed=4), M.Protocol(data_seed=-1),
    M.Protocol(roles=('A','A')), M.Protocol(atomic_examples=1),
    M.Protocol(test_cues=1), M.Protocol(k=True),
])
def test_invalid_configuration_rejected(bad):
    with pytest.raises(ValueError):
        bad.validate()


def test_invalid_model_outputs_rejected():
    with pytest.raises(ValueError):
        M._valid_output([1,1,2], 64, 3)
    with pytest.raises(ValueError):
        M._valid_output([1,2,64], 64, 3)
    with pytest.raises(ValueError):
        M._valid_output([1.1,2.2,3.3], 64, 3)
    assert M._valid_output(None, 64, 3) is None

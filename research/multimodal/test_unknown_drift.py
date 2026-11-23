"""Protocol and leak guards; these are synthetic, not native Brain tests."""
import copy
import numpy as np
import pytest
from research.multimodal.unknown_drift import (
    Scenario, Protocol, SCENARIOS, DualTimescale, _private_stream, trial, run,
)


def small(scenarios=SCENARIOS, seeds=(0, 1)):
    return Protocol(n=48, k=4, rounds=48, probes_per_role=3,
                    scenarios=scenarios, seeds=seeds)


def test_default_protocol():
    assert Protocol().validate() == Protocol()
    assert len(Protocol().scenarios) == 4


@pytest.mark.parametrize('p', [
    Protocol(n=4, k=4), Protocol(rounds=0), Protocol(probes_per_role=1),
    Protocol(n=5, k=2, rounds=20, probes_per_role=3),
    Protocol(seeds=()), Protocol(seeds=(1, 1)), Protocol(seeds=(-1,)),
    Protocol(scenarios=()), Protocol(scenarios=(SCENARIOS[0], SCENARIOS[0])),
    Protocol(scenarios=(Scenario('bad', (1,)),)),
    Protocol(scenarios=(Scenario('bad', (48, 32)),)),
    Protocol(scenarios=(Scenario('bad', (), (49,)),)),
])
def test_bad_protocol(p):
    with pytest.raises(ValueError):
        p.validate()


def test_same_probe_and_train_cues_across_scenarios():
    p = small()
    ss = [trial(0, p, sc) for sc in p.scenarios]
    assert len({x['training_cues_sha256'] for x in ss}) == 1
    assert len({x['probe_cues_sha256'] for x in ss}) == 1
    assert len({x['rule_commitment_sha256'] for x in ss}) == 1
    assert len({x['observed_labels_sha256'] for x in ss}) > 1


def test_seeded_replay():
    p = small(scenarios=SCENARIOS[:2])
    assert run(p) == run(p)


def test_corruption_is_not_reported_as_change():
    r = trial(0, small(), SCENARIOS[2])
    assert [x['round'] for x in r['rows'] if x['changed']] == [13, 27, 39]
    assert [x['round'] for x in r['rows'] if x['corrupted_label']] == [7, 18, 32, 44]


def test_models_do_not_see_hidden_data(monkeypatch):
    import research.multimodal.unknown_drift as exp
    made = []
    class Spy:
        def __init__(self):
            self.events = []
            made.append(self)
        def train(self, role, cue, label):
            assert role in ('SUBJ', 'OBJ')
            assert len(cue) == 4 and len(label) == 4
            self.events.append((role, cue.copy(), label.copy()))
        def state_digest(self):
            return str(len(self.events))
        def predict(self, role, cue):
            return None
    monkeypatch.setattr(exp, 'learners', lambda n, k: {'spy': Spy()})
    p = small(scenarios=SCENARIOS[:1], seeds=(0,))
    result = exp.run(p)
    assert len(made) == 1 and len(made[0].events) == p.rounds
    assert all(x['after_exact']['spy'] == 0 for x in result['trials'][0]['rows'])


def test_prediction_mutation_rejected(monkeypatch):
    import research.multimodal.unknown_drift as exp
    class Mutator:
        def __init__(self):
            self.count = 0
        def train(self, role, cue, label): pass
        def state_digest(self): return str(self.count)
        def predict(self, role, cue):
            self.count += 1
            return None
    monkeypatch.setattr(exp, 'learners', lambda n, k: {'mutator': Mutator()})
    with pytest.raises(AssertionError, match='read mutated'):
        exp.run(small(scenarios=SCENARIOS[:1], seeds=(0,)))


def test_quorum_avoids_one_mismatched_candidate():
    n = 32
    a = np.array([1, 3, 5, 8], dtype=np.uint32)
    m = DualTimescale(n, .0, 2)
    m.train('SUBJ', a, (a + 2) % n)
    baseline = m.predict('SUBJ', a).copy()
    m.train('SUBJ', a, (a + 10) % n)
    np.testing.assert_array_equal(m.predict('SUBJ', a), baseline)
    m.train('SUBJ', a, (a + 10) % n)
    np.testing.assert_array_equal(m.predict('SUBJ', a), np.sort((a + 10) % n))


def test_digest_unchanged_on_predict():
    m = DualTimescale(32, .5, 2)
    cue = np.array([1, 2, 3, 4], dtype=np.uint32)
    m.train('OBJ', cue, (cue + 7) % 32)
    before = m.state_digest()
    for _ in range(9):
        m.predict('OBJ', cue)
    assert before == m.state_digest()


@pytest.mark.parametrize('value', [float('nan'), 1., -1., float('inf')])
def test_invalid_decay(value):
    with pytest.raises(ValueError): DualTimescale(32, value, 2)

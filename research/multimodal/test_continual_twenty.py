import json
import numpy as np
import pytest
from research.multimodal.continual_twenty import Protocol, ShiftLearner, HebbianPairs, exact, run, trial


def test_twenty_rounds_per_seed_and_no_probe_training_alias():
    p = Protocol(seeds=(0, 1, 2))
    r = run(p)
    assert len(r['trials']) == 3
    assert len(r['summary']) == 20 * 5
    for t in r['trials']:
        assert [row['round'] for row in t['rows']] == list(range(1,21))
        assert [row['trained_role'] for row in t['rows']] == ['SUBJ','OBJ']*10
        assert len(t['training_sha256']) == len(t['probe_sha256']) == 64
    assert [row['round'] for row in r['trials'][0]['rows'] if row['corrupted_label']] == [7,16]


def test_reproducibility_seed_hashes_and_model_independent_outputs():
    p = Protocol(seeds=(1, 3))
    assert json.dumps(run(p), sort_keys=True) == json.dumps(run(p), sort_keys=True)
    a, b = run(p)['trials']
    assert a['training_sha256'] != b['training_sha256']
    assert a['probe_sha256'] != b['probe_sha256']


def test_learned_shift_adapts_and_global_history_resists_change():
    report = run(Protocol(seeds=(0,1,2,3)))
    def metric(round_id, model, field):
        return next(x[field] for x in report['summary'] if x['round'] == round_id and x['model'] == model)
    assert metric(2, 'recent_rule', 'mean_after_exact') == 1.0
    assert metric(10, 'recent_rule', 'mean_after_exact') == 1.0
    assert metric(15, 'recent_rule', 'mean_after_exact') == 1.0
    assert metric(15, 'global_rule', 'mean_after_exact') < 1.0
    assert metric(20, 'recent_rule', 'mean_after_exact') == 1.0
    assert metric(20, 'global_rule', 'mean_old_retention_exact') > 0.0
    for r in range(1, 21):
        assert metric(r, 'hebbian_pairs', 'mean_after_exact') == 0.0
        assert metric(r, 'copy_input', 'mean_after_exact') == 0.0


def test_output_validation_and_known_shift():
    assert exact(None, np.array([1,2]), 2) == 0.
    assert exact(np.array([1,2]), np.array([2,1]), 2) == 1.
    with pytest.raises(ValueError):
        exact([1,1], [1,2], 2)
    learner = ShiftLearner(64, 4, window=2)
    cue = np.array([1,2,3,4], dtype=np.uint32)
    assert learner.predict('SUBJ', cue) is None
    for shift in (7,7):
        target = np.sort(((cue+shift)%64).astype(np.uint32))
        learner.train('SUBJ',cue,target)
    np.testing.assert_array_equal(learner.predict('SUBJ',cue),np.sort(cue+7))


def test_hebbian_memorizes_observation_not_unseen_shift():
    model=HebbianPairs(128,8)
    cue=np.arange(8,dtype=np.uint32)
    target=np.arange(8,16,dtype=np.uint32)
    model.train('SUBJ',cue,target)
    assert exact(model.predict('SUBJ',cue),target,8) == 1.
    heldout=np.arange(20,28,dtype=np.uint32)
    unseen_target=(heldout+8)%128
    assert exact(model.predict('SUBJ',heldout),unseen_target,8) == 0.


@pytest.mark.parametrize('spec',[
    Protocol(k=1), Protocol(k=128), Protocol(change_after=0),
    Protocol(change_after=20), Protocol(rounds=3), Protocol(probes_per_role=1),
    Protocol(seeds=()), Protocol(seeds=(1,1)), Protocol(seeds=(19331,)),
    Protocol(learner_seed=-1),Protocol(poison_rounds=(21,)),
    Protocol(poison_rounds=(7,7)),Protocol(window=0)
])
def test_invalid_configs(spec):
    with pytest.raises(ValueError): spec.validate()


def test_eval_is_read_only_for_learning_state():
    a = ShiftLearner(128,8,window=3)
    cue = np.arange(8, dtype=np.uint32)
    target = cue + 17
    a.train('SUBJ', cue, target)
    h = a.state_digest()
    for _ in range(10):
        a.predict('SUBJ', cue)
    assert a.state_digest() == h
    b = HebbianPairs(128,8)
    b.train('SUBJ', cue, target)
    before = b.state_digest()
    b.predict('SUBJ', cue)
    assert b.state_digest() == before


def test_seeded_after_switch_prequential_metric_changes():
    t = trial(5, Protocol(seeds=(5,)))
    assert t['rows'][9]['period'] == 'old'
    assert t['rows'][10]['period'] == 'new'
    assert t['rows'][10]['before']['global_rule']['mean_exact'] == 0.0
    assert t['rows'][10]['after']['global_rule']['mean_exact'] == 0.0

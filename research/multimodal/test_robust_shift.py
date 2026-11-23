"""R1m exact stream parity and train-only inference state invariants."""
import numpy as np
import pytest
from research.multimodal.continual_twenty import Protocol, trial as original_trial
from research.multimodal.robust_shift import Quarantine, Decayed, trial, run


def example(shift):
    cue = np.arange(8, dtype=np.uint32)
    return cue, np.sort((cue+shift)%128).astype(np.uint32)


def test_quarantine_ignores_isolated_poison_then_commits_repeated_change():
    m = Quarantine(128)
    cue, old = example(7)
    m.train('SUBJ', cue, old)
    m.train('SUBJ', cue, example(20)[1])
    assert m.active['SUBJ'] == 7
    m.train('SUBJ', cue, old)
    assert m.pending['SUBJ'] is None
    m.train('SUBJ', cue, example(31)[1])
    assert m.active['SUBJ'] == 7
    m.train('SUBJ', cue, example(31)[1])
    assert m.active['SUBJ'] == 31
    assert m.switches['SUBJ'] == 1


def test_predictions_do_not_mutate_evidence():
    for m in [Quarantine(128), Decayed(128, .5), Decayed(128, .8)]:
        cue, target = example(9)
        assert m.predict('SUBJ', cue) is None
        m.train('SUBJ', cue, target)
        old_hash = m.state_digest()
        for _ in range(5):
            np.testing.assert_array_equal(m.predict('SUBJ', cue), target)
        assert m.state_digest() == old_hash


@pytest.mark.parametrize('seed', [0,5,17])
def test_same_private_training_stream_as_r1l(seed):
    p = Protocol(seeds=(seed,))
    new, old = trial(seed,p), original_trial(seed,p)
    for key in ('training_sha256','probe_sha256','shifts_sha256'):
        assert new[key] == old[key]
    assert [v['corrupted_label'] for v in new['rows']] == [v['corrupted_label'] for v in old['rows']]


def test_twenty_round_determinism():
    p = Protocol(seeds=(0,1))
    a=run(p)
    assert a == run(p)
    assert len(a['trials'])==2 and len(a['summary'])==20*3


@pytest.mark.parametrize('args', [(2,0),(2,-1),(128,0)])
def test_invalid_quorum(args):
    with pytest.raises(ValueError): Quarantine(*args)


@pytest.mark.parametrize('gamma', [-.1,1.,2.])
def test_invalid_decay(gamma):
    with pytest.raises(ValueError): Decayed(128,gamma)

"""Pure weight-audit tests; native engine execution is a separate gate."""
from types import SimpleNamespace
import numpy as np
import pytest

from research.multimodal.learning_audit import (
    _fiber, weight_fingerprint, active_synapses, write_diagnostic
)


class Dummy:
    def __init__(self):
        self.weights = {'SUBJ': np.eye(8, dtype=np.float32),
                        'OBJ': np.ones((8, 8), dtype=np.float32)}

    def engine_for(self, role):
        return SimpleNamespace(_area_conns={
            'LEX': {role: SimpleNamespace(weights=self.weights[role])}
        })


def test_fingerprint_change_and_restore_and_role_order():
    brain = Dummy()
    original = weight_fingerprint(brain, ('SUBJ', 'OBJ'))
    assert weight_fingerprint(brain, ('OBJ', 'SUBJ')) == original
    before = active_synapses(brain, 'SUBJ', [1, 2], [4, 5])
    brain.weights['SUBJ'][1, 4] += 1
    after = active_synapses(brain, 'SUBJ', [1, 2], [4, 5])
    assert weight_fingerprint(brain, ('SUBJ', 'OBJ')) != original
    stats = write_diagnostic(before, after, [4, 5], [4, 6])
    assert stats['active_edge_delta_sum'] == 1
    assert stats['active_edge_changed'] == 1
    assert stats['teacher_target_overlap'] == .5
    brain.weights['SUBJ'][1, 4] -= 1
    assert weight_fingerprint(brain, ('SUBJ', 'OBJ')) == original


def test_detects_nonlocal_mutation():
    brain = Dummy()
    prior = weight_fingerprint(brain, ('SUBJ', 'OBJ'))
    brain.weights['OBJ'][7, 7] *= 2
    assert weight_fingerprint(brain, ('SUBJ', 'OBJ')) != prior


def test_invalid_or_missing_fiber_is_error():
    brain = Dummy()
    with pytest.raises(RuntimeError):
        _fiber(brain, 'missing')
    with pytest.raises(ValueError):
        active_synapses(brain, 'SUBJ', [0, 0], [1, 2])
    with pytest.raises(ValueError):
        active_synapses(brain, 'SUBJ', [0, 8], [1, 2])
    with pytest.raises(ValueError):
        weight_fingerprint(brain, ('SUBJ', 'SUBJ'))
    with pytest.raises(ValueError):
        write_diagnostic(np.zeros((2, 2)), np.zeros((3, 2)), [1], [1])


def test_read_only_probe_must_leave_fingerprint_equal():
    brain = Dummy()
    first = weight_fingerprint(brain, ('SUBJ', 'OBJ'))
    _ = brain.weights['OBJ'][[0, 1]].sum(axis=0)
    assert weight_fingerprint(brain, ('SUBJ', 'OBJ')) == first


def test_zero_plasticity_teacher_clamp_does_not_count_as_learning():
    brain = Dummy()
    cue = np.array([1, 2], dtype=np.uint32)
    target = np.array([3, 4], dtype=np.uint32)
    before = active_synapses(brain, 'SUBJ', cue, target)
    # Clamping winners alone should not change a single edge.
    winners = target.copy()
    after = active_synapses(brain, 'SUBJ', cue, target)
    stats = write_diagnostic(before, after, target, winners)
    assert stats['teacher_target_overlap'] == 1.
    assert stats['active_edge_changed'] == 0
    assert stats['active_edge_delta_sum'] == 0.


def test_full_fingerprint_detects_reverse_and_recurrent_updates():
    from research.multimodal.learning_audit import full_explicit_fingerprint

    class Complete:
        def __init__(self):
            self.owner = SimpleNamespace(_area_conns={
                src: {dst: SimpleNamespace(weights=np.ones((5, 5), dtype=np.float32))
                      for dst in ('LEX', 'SUBJ', 'OBJ')}
                for src in ('LEX', 'SUBJ', 'OBJ')})

        def engine_for(self, area):
            return self.owner

    brain = Complete()
    before = full_explicit_fingerprint(brain, ('LEX', 'SUBJ', 'OBJ'))
    brain.owner._area_conns['OBJ']['LEX'].weights[1, 2] += 1
    assert full_explicit_fingerprint(brain, ('LEX', 'SUBJ', 'OBJ')) != before
    with pytest.raises(ValueError):
        full_explicit_fingerprint(brain, ('OBJ', 'OBJ'))

    class Mixed(Complete):
        def engine_for(self, area):
            return self.owner if area == 'LEX' else object()

    with pytest.raises(RuntimeError, match='mixed-engine'):
        full_explicit_fingerprint(Mixed(), ('LEX', 'SUBJ'))

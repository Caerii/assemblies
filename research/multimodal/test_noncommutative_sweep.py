"""Seed- and geometry-paired R1o sweep contracts."""
import pytest
from research.multimodal.noncommutative_sweep import Sweep, run


def test_small_sweep_exact_replay_and_paired_cases():
    s = Sweep(geometries=((32, 5), (64, 8)), seeds=(0, 1),
              atomic_examples=5, test_cues=3)
    a = run(s)
    assert a == run(s)
    assert len(a['cells']) == 2 * 2 * 6
    assert len(a['summary']) == 2 * 6
    by_trial = {}
    for item in a['cells']:
        key = (item['n'], item['k'], item['data_seed'])
        by_trial.setdefault(key, set()).add((item['training_cues_sha256'],
                                            item['test_cues_sha256'],
                                            item['hidden_targets_sha256']))
    assert all(len(v) == 1 for v in by_trial.values())


def test_locked_contrasts_on_small_seed_grid():
    a = run(Sweep(geometries=((64, 8),), seeds=(0, 1, 2, 3), test_cues=4))
    by_model = {s['model']: s for s in a['summary']}
    assert by_model['affine_rule']['composed_exact']['mean'] == 1.0
    assert by_model['affine_rule']['ordered_AB_BA_both_exact']['mean'] == 1.0
    assert by_model['order_blind']['ordered_AB_BA_both_exact']['mean'] == 0.
    assert by_model['hebbian_pairs']['composed_exact']['mean'] == 0.
    assert by_model['atomic_lookup']['composed_exact']['mean'] == 0.


@pytest.mark.parametrize('bad', [
    Sweep(seeds=()), Sweep(seeds=(0, 0)), Sweep(seeds=(8123,)),
    Sweep(geometries=()), Sweep(geometries=((64,8),(64,8))),
    Sweep(geometries=((8,9),)), Sweep(model_seed=-1)
])
def test_bad_grid_rejected(bad):
    with pytest.raises(ValueError): run(bad)

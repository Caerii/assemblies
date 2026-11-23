"""Zero-shot gate contract checks; no native Brain performance is claimed."""
import numpy as np
import pytest
from research.multimodal.zero_shot_gate import (
    Config, Dataset, evaluate, KnownPermutationOracle, ExactPairMemory,
    CopyBaseline, HebbianRoleMap, ShiftRoleLearner, _validate_output, _rank,
)


def test_split_is_strict_and_balanced():
    d = Dataset(Config())
    assert not set(d.train) & set(d.test)
    assert len(d.train) == len(d.test) == 12
    for role in d.cfg.roles:
        assert sum(r == role for _, r in d.train) == 6
        assert sum(r == role for _, r in d.test) == 6
    for item in range(d.cfg.contents):
        assert sum(i == item for i, _ in d.train) == 1
        assert sum(i == item for i, _ in d.test) == 1


@pytest.mark.parametrize("seed", [0, 1, 17, 42])
def test_oracle_and_exact_memory(seed):
    cfg = Config(seed=seed)
    oracle = evaluate(cfg, KnownPermutationOracle, expose_labels=True)
    mem = evaluate(cfg, ExactPairMemory)
    assert oracle["summary"]["heldout"]["exact"] == 1.
    assert oracle["summary"]["heldout"]["strict_rank1"] == 1.
    assert mem["summary"]["train"]["exact"] == 1.
    assert mem["summary"]["heldout"]["exact"] == 0.


def test_copy_is_not_trivially_a_role_transform():
    result = evaluate(Config(), CopyBaseline)
    assert result["summary"]["heldout"]["exact"] == 0.


@pytest.mark.parametrize("seed", [0, 1, 17, 42])
def test_hebbian_memorizes_trained_but_not_heldout_random_role_permutation(seed):
    r = evaluate(Config(seed=seed), HebbianRoleMap)
    assert r["summary"]["train"]["exact"] == 1.
    assert r["summary"]["heldout"]["exact"] == 0.


def test_determinism_and_label_copy():
    cfg = Config()
    assert evaluate(cfg, ExactPairMemory) == evaluate(cfg, ExactPairMemory)
    d = Dataset(cfg)
    arr = d.target(0, cfg.roles[0])
    arr[:] = 0
    assert len(np.unique(d.target(0, cfg.roles[0]))) == cfg.k


def test_strict_tie_rejected():
    got = np.array([1, 2], dtype=np.uint32)
    result = _rank(got, got, [got])
    assert result["exact"] == 1
    assert result["strict_rank1"] == 0
    assert result["margin"] == 0


@pytest.mark.parametrize("invalid", [None, [-1, 2], [2, 2], [1, 2, 3], [1.5, 2.5]])
def test_output_validation(invalid):
    if invalid is None:
        assert _validate_output(invalid, 10, 2) is None
    else:
        with pytest.raises(ValueError):
            _validate_output(invalid, 10, 2)


def test_train_never_sees_heldout_labels():
    cfg = Config(contents=8)
    seen = []

    class Spy:
        def __init__(self, _):
            pass

        def train(self, i, role, cue, target):
            seen.append((i, role))

        def predict(self, i, role, cue):
            return None

    evaluate(cfg, Spy)
    dataset = Dataset(cfg)
    assert set(seen) == set(dataset.train)
    assert not set(seen) & set(dataset.test)


@pytest.mark.parametrize("seed", [0, 1, 17, 42])
def test_learnable_shift_distinguishes_rule_learning_from_pair_memory(seed):
    cfg = Config(seed=seed, transform_family="cyclic_shift")
    learned = evaluate(cfg, ShiftRoleLearner)
    pair = evaluate(cfg, HebbianRoleMap)
    assert learned["summary"]["train"]["exact"] == 1.
    assert learned["summary"]["heldout"]["exact"] == 1.
    assert learned["summary"]["heldout"]["strict_rank1"] == 1.
    assert pair["summary"]["train"]["exact"] == 1.
    assert pair["summary"]["heldout"]["exact"] == 0.


def test_rejects_unknown_transform_family():
    with pytest.raises(ValueError):
        Config(transform_family="unknown").validate()

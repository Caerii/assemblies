"""Calibration contract tests. These are NOT native Brain conformance tests."""
import pytest

from research.multimodal.zero_shot_baselines import Config, split, experiment


def test_disjoint_complete_split():
    cfg = Config(contents=8)
    train, heldout = split(cfg)
    assert len(train) == len(heldout) == 8
    assert not set(train) & set(heldout)
    assert len(set(train + heldout)) == 16


def test_structured_oracle_and_pair_memory_controls():
    for seed in (0, 1, 17, 42):
        result = experiment(Config(seed=seed))
        assert result["summary"] == {
            "factorized_train": 1.0,
            "factorized_heldout": 1.0,
            "lookup_train": 1.0,
            "lookup_heldout": 0.0,
        }
        assert all(row["margin"] > 0 for row in result["results"])


def test_determinism():
    cfg = Config(seed=7)
    assert experiment(cfg) == experiment(cfg)


@pytest.mark.parametrize("cfg", [
    Config(k=0), Config(k=1024), Config(contents=3), Config(contents=5),
    Config(roles=("SUBJ", "SUBJ")),
])
def test_invalid_configuration(cfg):
    with pytest.raises(ValueError):
        cfg.validate()

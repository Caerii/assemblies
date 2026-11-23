"""Contracts for deterministic continual-learning diagnostic."""
import numpy as np
import pytest
from research.multimodal.retention_stream import Config, exact, jaccard, run


def test_reproducibility():
    config = Config(n=64, k=4, loads=(4, 8, 16), seeds=(0, 1))
    assert run(config) == run(config)


def test_stream_prefix_and_accounting():
    result = run(Config(n=64, k=4, loads=(2, 5, 9), seeds=(4,)))
    assert [r["load"] for r in result["observations"]] == [2, 5, 9]
    assert len({r["stream_sha256"] for r in result["observations"]}) == 1
    assert [r["episodic_min_bytes"] for r in result["observations"]] == [64, 160, 288]


def test_scores_bounded():
    for row in run(Config(n=64, k=4, loads=(4, 16), seeds=(0, 7)))["observations"]:
        for key in ("assembly_exact", "assembly_jaccard", "assembly_early_exact",
                    "episodic_exact", "target_input_coverage"):
            assert 0 <= row[key] <= 1


def test_overlap_scoring():
    assert exact(np.array([1, 2]), np.array([1, 2]))
    assert jaccard(np.array([1, 2]), np.array([2, 3])) == pytest.approx(1/3)


@pytest.mark.parametrize("config", [
    Config(n=8, k=9), Config(p=2.), Config(beta=-.1),
    Config(loads=(0,)), Config(seeds=(1, 1)), Config(presentations=0),
])
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        run(config)


def test_dense_connectivity_positive_control():
    result = run(Config(n=48, k=4, p=1., beta=.5, presentations=3,
                        loads=(1,), seeds=(0, 1)))
    assert result["summary"][0]["assembly_exact"]["mean"] == 1.
    assert result["summary"][0]["target_input_coverage"]["mean"] == 1.

"""PFA table weights are not evidence of calibrated neural probabilities."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from neural_assemblies.assembly_calculus import SeedMixtureChoice, PFANetwork
from neural_assemblies.assembly_calculus import pfa as module
from neural_assemblies.core.brain import Brain
from neural_assemblies.programs.markov_coin import CoinFlipModel

BRANCHES = [("q0", "a", "q0", .25), ("q0", "a", "q1", .75)]


def test_implicit_neural_probability_fails_before_brain_mutation():
    with pytest.raises(ValueError, match="not calibrated"):
        PFANetwork(object(), ["q0", "q1"], ["a"], BRANCHES, "q0")
    with pytest.raises(ValueError, match="not calibrated"):
        CoinFlipModel(object(), [("q0", "a", "q1")], "q0")


@pytest.mark.parametrize("override", [
    {"n": 0}, {"k": True}, {"beta": float("nan")}, {"beta": -1},
    {"beta": True}, {"rounds_train": 0}, {"rounds": -1}, {"rounds": True},
    {"fires": 1.5}, {"mode": "probability"},
])
def test_choice_rejects_invalid_settings(override):
    settings = dict(n=100, k=10, beta=.1)
    settings.update(override)
    with pytest.raises(ValueError):
        SeedMixtureChoice(**settings)


def test_choice_is_immutable_and_conflicts_fail_before_construction():
    choice = SeedMixtureChoice(100, 10, .1, rounds=0, fires=0)
    with pytest.raises(FrozenInstanceError):
        choice.beta = 1
    with pytest.raises(ValueError, match="agree"):
        PFANetwork(object(), ["q0", "q1"], ["a"], BRANCHES, "q0",
                   choice=choice, flip_mode="compete")


def test_deterministic_pfa_does_not_construct_a_coin(monkeypatch):
    monkeypatch.setattr(module, "FSMNetwork", lambda *args, **kw: SimpleNamespace())
    def forbidden(*args, **kwargs):
        raise AssertionError("deterministic PFA must not construct a coin")
    monkeypatch.setattr(module, "RandomChoiceArea", forbidden)
    network = PFANetwork(object(), ["q0"], ["a"], [("q0", "a", "q0")], "q0")
    assert network._coin is None


def test_pfa_consumes_independent_training_and_read_schedule(monkeypatch):
    builds, flips = [], []
    monkeypatch.setattr(module, "FSMNetwork", lambda *args, **kw: SimpleNamespace())
    def build(brain, **kwargs):
        builds.append(kwargs)
        def flip(**settings):
            flips.append(settings)
            return 1
        return SimpleNamespace(flip=flip)
    monkeypatch.setattr(module, "RandomChoiceArea", build)
    choice = SeedMixtureChoice(2000, 200, 3., rounds_train=7, fires=4, rounds=6, mode="compete")
    network = PFANetwork(object(), ["q0", "q1"], ["a"], BRANCHES, "q0",
                         n=100, k=10, beta=.1, choice=choice)
    assert builds == [dict(area_name="flip", prefix="_pfa_coin", n=2000, k=200,
                          beta=3., rounds_train=7, fires=4, construction="attractor")]
    assert network.step("a", seed=11) == "q1"
    assert flips == [dict(bias=.25, rounds=6, seed=11, mode="compete")]


def test_explicit_branch_repeats_seeded_selection():
    brain = Brain(p=.05, seed=1, engine="numpy_sparse")
    choice = SeedMixtureChoice(2000, 200, 3., rounds_train=10, rounds=10)
    network = PFANetwork(brain, ["q0", "q1"], ["a"], BRANCHES, "q0",
                         n=500, k=20, beta=.1, rounds=3, choice=choice)
    coin = network._coin
    engine = brain._engine_for(brain.areas[coin.area_name])
    # Replay the same initial seed after an intervening choice. Exact labels and
    # winners establish deterministic read behavior here, not probability calibration.
    outcomes = []
    for seed in (701, 702, 701):
        network._current_state = "q0"
        outcomes.append((network.step("a", seed=seed), brain.areas[coin.area_name].winners.copy()))
    assert outcomes[0][0] == outcomes[2][0]
    np.testing.assert_array_equal(outcomes[0][1], outcomes[2][1])
    assert engine.materialized_count(coin.area_name) == choice.n


def test_markov_wrapper_runs_both_explicit_coins():
    brain = Brain(p=.05, seed=1, engine="numpy_sparse")
    choice = SeedMixtureChoice(2000, 200, 3., rounds_train=10, rounds=0, mode="compete")
    model = CoinFlipModel(brain, [("q0", "a", "q0"), ("q0", "a", "q1")],
                          "q0", n=500, k=20, rounds=3, choice=choice)
    assert model.pfa.choice is choice
    assert model.coin.n == choice.n
    assert model.sample_branch(seed=1) in (0, 1)
    assert model.step_symbol("a", seed=1) in ("q0", "q1")


def test_choice_record_is_json_serializable_with_numpy_controls():
    from dataclasses import asdict
    import json
    choice = SeedMixtureChoice(np.int64(100), np.int64(10), np.float32(.25),
                               rounds_train=np.int64(2), fires=np.int64(0), rounds=np.int64(0))
    assert json.loads(json.dumps(asdict(choice))) == dict(
        n=100, k=10, beta=.25, rounds_train=2, fires=0, rounds=0, mode="k_split")


def test_cascade_consumes_configured_schedule_for_every_branch(monkeypatch):
    flips = []
    monkeypatch.setattr(module, "FSMNetwork", lambda *args, **kw: SimpleNamespace())
    def build(*args, **kwargs):
        def flip(**settings):
            flips.append(settings)
            return 1
        return SimpleNamespace(flip=flip)
    monkeypatch.setattr(module, "RandomChoiceArea", build)
    choice = SeedMixtureChoice(100, 10, .1, rounds=4, mode="compete")
    branches = [("q0", "a", state, weight) for state, weight in
                (("q0", .2), ("q1", .3), ("q2", .5))]
    network = PFANetwork(object(), ["q0", "q1", "q2"], ["a"], branches,
                         "q0", choice=choice)
    assert network.step("a", seed=11) == "q2"
    assert [row["bias"] for row in flips] == pytest.approx([.2, .375])
    assert all(row["rounds"] == 4 and row["mode"] == "compete" for row in flips)

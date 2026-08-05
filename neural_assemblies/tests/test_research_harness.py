"""The harness must FAIL a bad arm, or it is decoration.

Every test here is a real failure mode from 2026-08-05, encoded so the harness
is checked against the mistakes it exists to prevent rather than against its own
happy path:

  * a candidate that INVERTS on one seed while the mean still looks fine
    (a mean hid a seed-42 inversion, and the change was nearly adopted)
  * a candidate that is CONSTANT across seeds (`afferent_energy` read exactly
    0.000 on four seeds, which is a structural artefact, not a clean negative)
  * both arms run in a fixed order, so the second is always measured warm
  * a study that does not record whether it ran on cached or fresh parsers

The most important assertion in the file is
`test_a_deliberately_inverted_arm_fails`: a harness that cannot reject is not a
harness. Its absence is exactly the "guard whose true-negative case was never
constructed" failure this repo keeps hitting.
"""
from __future__ import annotations

import pytest

from research.harness import Criteria, Provenance, study

SEEDS = [11, 12, 13, 42]


def _const(value):
    return lambda seed: {"auc": value}


def _per_seed(mapping):
    return lambda seed: {"auc": mapping[seed]}


class TestItRejectsWhatItMust:

    def test_a_deliberately_inverted_arm_fails(self):
        """THE true-negative. If this passes, the harness asserts nothing."""
        res = study(
            arms={"control": _const(0.9), "candidate": _const(0.1)},
            seeds=SEEDS,
            criteria={"auc": Criteria(above=0.5)},
        )
        assert not res.passed
        assert "0.1000 <= 0.5" in str(res)

    def test_a_single_inverted_seed_fails_even_when_the_mean_passes(self):
        """The seed-42 shape: mean 0.72 looks fine, one seed is below chance.

        Without `on_every_seed` this study PASSES, which is how the last
        structural change got through review.
        """
        vals = {11: 1.0, 12: 0.95, 13: 0.9, 42: 0.05}
        mean_ok = sum(vals.values()) / len(vals) > 0.5
        assert mean_ok, "fixture must have a passing mean, else it proves nothing"

        lenient = study(
            arms={"control": _const(0.9), "candidate": _per_seed(vals)},
            seeds=SEEDS, criteria={"auc": Criteria(above=0.5)},
        )
        assert lenient.passed, "mean-only check should be fooled -- that is the point"

        strict = study(
            arms={"control": _const(0.9), "candidate": _per_seed(vals)},
            seeds=SEEDS,
            criteria={"auc": Criteria(above=0.5, on_every_seed=True)},
        )
        assert not strict.passed
        assert "42" in str(strict)

    def test_a_constant_candidate_fails_must_vary(self):
        """Zero variance across seeds is a structural artefact, not an effect."""
        res = study(
            arms={"control": _per_seed({11: .9, 12: .8, 13: .95, 42: .85}),
                  "candidate": _const(0.7)},
            seeds=SEEDS,
            criteria={"auc": Criteria(above=0.5, must_vary=True)},
        )
        assert not res.passed
        assert "CONSTANT" in str(res)


class TestItAcceptsWhatItShould:

    def test_an_honest_smaller_effect_passes(self):
        """A DROP must be allowed by default.

        Removing a confound should shrink an inflated effect. A bar that treats
        every decrease as failure selects for confounded metrics -- which is the
        opposite of what this repo needs.
        """
        res = study(
            arms={"control": _const(0.95),
                  "candidate": _per_seed({11: .72, 12: .70, 13: .74, 42: .69})},
            seeds=SEEDS,
            criteria={"auc": Criteria(above=0.5, on_every_seed=True,
                                      must_vary=True)},
        )
        assert res.passed
        assert res.metrics["auc"].delta.mean < 0

    def test_unjudged_metrics_are_reported_but_not_failed(self):
        """"Measured" and "committed to" must stay visibly separate."""
        res = study(
            arms={"control": _const(0.9), "candidate": _const(0.1)},
            seeds=SEEDS, criteria={},
        )
        assert res.passed
        assert "auc" in res.metrics


class TestProtocol:

    def test_counterbalancing_alternates_which_arm_runs_first(self):
        """Fixed order is how a candidate got measured only ever WARM."""
        order = []
        arms = {
            "control": lambda s: (order.append(("control", s)), {"auc": 0.9})[1],
            "candidate": lambda s: (order.append(("candidate", s)), {"auc": 0.8})[1],
        }
        study(arms=arms, seeds=SEEDS, order="counterbalance")
        first_per_seed = [order[i][0] for i in range(0, len(order), 2)]
        assert first_per_seed == ["control", "candidate", "control", "candidate"], (
            f"expected alternating first-arm, got {first_per_seed}")

    def test_as_given_preserves_the_old_confounded_order(self):
        """Kept available, but it must be an explicit choice."""
        order = []
        arms = {
            "control": lambda s: (order.append("control"), {"auc": 0.9})[1],
            "candidate": lambda s: (order.append("candidate"), {"auc": 0.8})[1],
        }
        study(arms=arms, seeds=SEEDS, order="as_given")
        assert order[0::2] == ["control"] * len(SEEDS)

    def test_too_few_seeds_is_refused(self):
        with pytest.raises(ValueError, match="cannot support an interval"):
            study(arms={"a": _const(1.0), "b": _const(0.0)}, seeds=[1, 2])

    def test_exactly_two_arms(self):
        with pytest.raises(ValueError, match="exactly two arms"):
            study(arms={"a": _const(1.0)}, seeds=SEEDS)


class TestProvenanceIsRecorded:
    """Phase 0's requirement: an A/B on cached parsers is evidence about
    cached parsers only, so the substrate must be in the artefact."""

    def test_study_records_substrate_provenance(self):
        res = study(arms={"a": _const(0.9), "b": _const(0.8)}, seeds=SEEDS)
        assert isinstance(res.provenance, Provenance)
        assert "disk_hits" in str(res.provenance)
        assert "trained_fresh" in str(res.provenance)

    def test_missing_cache_stats_degrade_to_zero_not_to_a_crash(self):
        """A study must never fail because provenance could not be read."""
        p = Provenance(cache_stats_before={}, cache_stats_after={})
        assert p.cache_disk_hits == 0 and p.trained_fresh == 0

    def test_non_numeric_cache_stats_do_not_crash(self):
        p = Provenance(cache_stats_before={"disk_hits": "?"},
                       cache_stats_after={"disk_hits": "?"})
        assert p.cache_disk_hits == 0

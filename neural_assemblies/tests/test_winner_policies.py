import numpy as np

from neural_assemblies.compute import (
    RelativeThresholdPolicy,
    ThresholdPolicy,
    TopKPolicy,
    WinnerSelector,
)


class TestWinnerPolicies:
    def setup_method(self):
        self.selector = WinnerSelector(np.random.default_rng(2026))

    def test_top_k_policy_matches_existing_selection(self):
        features = np.array([0.1, 0.9, 0.3, 0.8, 0.2])
        winners = self.selector.select_with_policy(features, TopKPolicy(k=3))
        np.testing.assert_array_equal(winners, np.array([1, 3, 2]))

    def test_top_k_policy_honors_tie_policy(self, monkeypatch):
        features = np.array([1.0, 1.0, 0.5])

        def fake_select_top_k_indices(values, k):
            raise AssertionError("TopKPolicy should not bypass its tie-aware selection path")

        monkeypatch.setattr(self.selector, "select_top_k_indices", fake_select_top_k_indices)

        winners = self.selector.select_with_policy(
            features,
            TopKPolicy(k=2, tie_policy="value_then_index"),
        )
        np.testing.assert_array_equal(winners, np.array([0, 1]))

    def test_threshold_policy_caps_selection(self):
        features = np.array([0.1, 0.9, 0.8, 0.75, 0.2])
        winners = self.selector.select_with_policy(
            features,
            ThresholdPolicy(k=2, threshold=0.7),
        )
        np.testing.assert_array_equal(winners, np.array([1, 2]))

    def test_threshold_policy_returns_empty_when_k_is_zero(self):
        features = np.array([1.0, 0.5, 0.2])
        winners = self.selector.select_with_policy(
            features,
            ThresholdPolicy(k=0, threshold=0.4),
        )
        np.testing.assert_array_equal(winners, np.array([], dtype=int))

    def test_threshold_policy_honors_tie_policy(self, monkeypatch):
        features = np.array([1.0, 1.0, 0.5])
        seen = {}

        def fake_select_winners_with_threshold(values, k, threshold=None, tie_policy="value_then_index"):
            seen["tie_policy"] = tie_policy
            return np.array([0, 1]) if tie_policy == "value_then_index" else np.array([1, 0])

        monkeypatch.setattr(
            self.selector,
            "select_winners_with_threshold",
            fake_select_winners_with_threshold,
        )

        winners = self.selector.select_with_policy(
            features,
            ThresholdPolicy(k=2, threshold=0.8, tie_policy="value_then_index"),
        )
        np.testing.assert_array_equal(winners, np.array([0, 1]))
        assert seen["tie_policy"] == "value_then_index"

    def test_relative_threshold_policy_can_return_variable_count(self):
        features = np.array([1.0, 0.92, 0.88, 0.3, 0.2])
        winners = self.selector.select_with_policy(
            features,
            RelativeThresholdPolicy(fraction_of_max=0.85, max_winners=5),
        )
        np.testing.assert_array_equal(winners, np.array([0, 1, 2]))

    def test_relative_threshold_policy_respects_minimum(self):
        features = np.array([1.0, 0.4, 0.3, 0.2])
        winners = self.selector.select_with_policy(
            features,
            RelativeThresholdPolicy(fraction_of_max=1.0, min_winners=2),
        )
        np.testing.assert_array_equal(winners, np.array([0, 1]))

    def test_relative_threshold_policy_validates_bounds(self):
        features = np.array([1.0, 0.5])
        try:
            self.selector.select_with_policy(
                features,
                RelativeThresholdPolicy(fraction_of_max=1.2),
            )
        except ValueError as exc:
            assert "fraction_of_max" in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid fraction_of_max")

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

    def test_threshold_policy_caps_selection(self):
        features = np.array([0.1, 0.9, 0.8, 0.75, 0.2])
        winners = self.selector.select_with_policy(
            features,
            ThresholdPolicy(k=2, threshold=0.7),
        )
        np.testing.assert_array_equal(winners, np.array([1, 2]))

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

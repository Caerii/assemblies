"""Parity tests: package metric kernels match research primitive definitions."""

import numpy as np

from neural_assemblies.assembly_calculus.metrics.instability import (
    cosine_similarity,
    compute_jaccard_instability,
    jaccard_similarity,
    mean_jaccard_instability,
)
from neural_assemblies.assembly_calculus.metrics.prediction import measure_n400
from neural_assemblies.assembly_calculus.assembly import compact_overlap, neuron_overlap, overlap
from neural_assemblies.core.index_spaces import CompactIdx, NeuronIds
from research.experiments.base import measure_overlap


class TestMetricKernels:
    def test_measure_n400_matches_overlap_formula(self):
        a = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
        b = np.array([3, 4, 5, 6, 7], dtype=np.uint32)
        expected = 1.0 - measure_overlap(a, b)
        assert measure_n400(a, b) == expected

    def test_research_overlap_helper_is_the_package_kernel(self):
        a = np.array([1, 2, 3], dtype=np.uint32)
        b = np.array([2, 3, 4], dtype=np.uint32)
        assert measure_overlap(a, b) == overlap(a, b)

    def test_named_overlap_kernels_make_index_space_explicit(self):
        neurons_a = NeuronIds(np.array([1, 2, 3], dtype=np.uint32))
        neurons_b = NeuronIds(np.array([2, 3, 4], dtype=np.uint32))
        compact_a = CompactIdx(np.array([1, 2, 3], dtype=np.uint32))
        compact_b = CompactIdx(np.array([2, 3, 4], dtype=np.uint32))
        assert neuron_overlap(neurons_a, neurons_b) == 2 / 3
        assert compact_overlap(compact_a, compact_b) == 2 / 3

    def test_jaccard_instability_on_known_sequence(self):
        rounds = [{1, 2, 3}, {1, 2, 4}, {5, 6, 7}]
        # pair 0->1: union=4, inter=2, jaccard=0.5, contrib=0.5
        # pair 1->2: union=5, inter=0, jaccard=0, contrib=1.0
        assert compute_jaccard_instability(rounds) == 1.5

    def test_mean_jaccard_normalizes_by_transitions(self):
        rounds = [{1, 2}, {1, 3}, {1, 4}]
        total = compute_jaccard_instability(rounds)
        assert mean_jaccard_instability(rounds) == total / 2.0

    def test_mean_jaccard_empty_or_single_round(self):
        assert mean_jaccard_instability([]) == 0.0
        assert mean_jaccard_instability([{1, 2, 3}]) == 0.0

    def test_jaccard_kernel_has_identity_and_zero_laws(self):
        assert jaccard_similarity([], []) == 1.0
        assert jaccard_similarity([1, 2], [3, 4]) == 0.0
        assert jaccard_similarity([1, 2], [2, 3]) == 1 / 3

    def test_cosine_kernel_has_set_and_empty_laws(self):
        assert cosine_similarity([], [1]) == 0.0
        assert cosine_similarity([1, 1, 2], [2, 3]) == 0.5

    def test_research_jaccard_helper_is_the_package_kernel(self):
        from research.experiments.base import measure_jaccard

        a = np.array([1, 2, 3], dtype=np.uint32)
        b = np.array([2, 3, 4], dtype=np.uint32)
        assert measure_jaccard(a, b) == jaccard_similarity(a, b)

    def test_research_instability_reexports_package_kernel(self):
        from research.experiments.metrics.instability import (
            compute_jaccard_instability as research_jaccard,
        )

        rounds = [{10, 11}, {10, 12}, {10, 13}]
        assert research_jaccard(rounds) == compute_jaccard_instability(rounds)

    def test_research_measurement_reexports_n400(self):
        from research.experiments.lib.measurement import measure_n400 as research_n400

        a = np.array([0, 1, 2], dtype=np.uint32)
        b = np.array([2, 3, 4], dtype=np.uint32)
        assert research_n400(a, b) == measure_n400(a, b)

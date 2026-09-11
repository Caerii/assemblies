"""The DENSE engine must be constructible, and must not swallow what it ignores.

`numpy_explicit` is the ground truth every sampler result is checked against --
it allocates all n neurons up front, so there is no candidate sampler inventing
drive and no lazy materialization to desync. Two defects made it less useful
than that role requires.

ONE: `Brain(engine="numpy_explicit")` raised TypeError on the FIRST `add_area`.
`Brain.add_area` forwards `winner_policy` and `input_noise_std` to the primary
engine unconditionally and this engine's signature had neither, so the dense
engine was unreachable through the public API. It could only be had as an
`explicit=True` area inside a sparse brain, via a different call with a
different kwarg set -- which is why the gap survived: the route people used
worked, and the route the API advertised did not.

TWO: `refractory_period` and `inhibition_strength` WERE in the signature and
went nowhere. `ExplicitAreaState` has no field for either, so LRI was silently
off while the caller's configuration said it was on -- [[silent-no-op-dead-fibers]]
exactly. They now raise instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.compute.winner_policies import EPercentPolicy, TopKPolicy

N, K, P, BETA = 300, 15, 0.05, 0.05


def _brain(**kw):
    b = Brain(p=P, seed=5, engine="numpy_explicit", norm_init=False, **kw)
    b.add_stimulus("S", K)
    return b


class TestReachableThroughTheAPI:

    def test_can_add_an_area_at_all(self):
        b = _brain()
        b.add_area("A", N, K, BETA)
        assert "A" in b.areas

    def test_can_project_and_get_k_winners(self):
        b = _brain()
        b.add_area("A", N, K, BETA)
        b.project({"S": ["A"]}, {})
        assert len(b.areas["A"].winners) == K

    def test_two_areas_and_an_area_to_area_projection(self):
        b = _brain()
        b.add_area("A", N, K, BETA)
        b.add_area("B", N, K, BETA)
        b.project({"S": ["A"]}, {})
        b.project({}, {"A": ["B"]})
        assert len(b.areas["B"].winners) == K

    def test_reported_engine_name(self):
        b = _brain()
        b.add_area("A", N, K, BETA)
        assert b._engine_for(b.areas["A"]).name == "numpy_explicit"

    def test_explicit_flag_reuses_primary_dense_engine(self):
        b = _brain()
        b.add_area("A", N, K, BETA, explicit=True)

        assert b._engine_for(b.areas["A"]) is b._engine
        assert b._explicit_engine is None

    def test_direct_dense_route_matches_legacy_sparse_router(self):
        def run(primary):
            b = Brain(p=P, seed=19, engine=primary, norm_init=False)
            b.add_stimulus("S", K)
            b.add_area("A", N, K, BETA, explicit=True)
            for _ in range(5):
                b.project({"S": ["A"]}, {"A": ["A"]})
            return b

        legacy = run("numpy_sparse")
        direct = run("numpy_explicit")
        assert np.array_equal(legacy.areas["A"].winners,
                              direct.areas["A"].winners)
        assert np.array_equal(
            legacy.connectomes_by_stimulus["S"]["A"].weights,
            direct.connectomes_by_stimulus["S"]["A"].weights,
        )
        assert np.array_equal(legacy.connectomes["A"]["A"].weights,
                              direct.connectomes["A"]["A"].weights)


class TestUnsupportedIsRefusedNotIgnored:

    @pytest.mark.parametrize("kw", [
        {"refractory_period": 3},
        {"inhibition_strength": 0.5},
        {"input_noise_std": 0.1},
    ])
    def test_configuring_an_unimplemented_mechanism_raises(self, kw):
        b = _brain()
        with pytest.raises(NotImplementedError, match="does not implement"):
            b.add_area("A", N, K, BETA, **kw)

    def test_the_message_names_what_was_asked_for(self):
        b = _brain()
        with pytest.raises(NotImplementedError, match="refractory_period=3"):
            b.add_area("A", N, K, BETA, refractory_period=3)

    def test_passing_the_default_is_not_a_request(self):
        """Brain forwards these on EVERY call, so defaults must stay silent or
        nothing could be constructed."""
        b = _brain()
        b.add_area("A", N, K, BETA, refractory_period=0,
                   inhibition_strength=0.0, input_noise_std=0.0)
        assert "A" in b.areas


class TestWinnerPolicies:

    def _drive(self, policy):
        b = _brain()
        b.add_area("A", N, K, BETA, winner_policy=policy)
        b.project({"S": ["A"]}, {})
        for _ in range(4):
            b.project({"S": ["A"]}, {"A": ["A"]})
        return b

    def test_topk_matching_k_is_identical_to_no_policy(self):
        """The fast path must not be a DIFFERENT rule that happens to agree
        approximately -- pin bit-identity."""
        a = self._drive(None)
        b = self._drive(TopKPolicy(k=K))
        assert np.array_equal(np.asarray(a.areas["A"].winners),
                              np.asarray(b.areas["A"].winners))

    def test_epercent_size_is_emergent_and_not_pinned_to_k_or_n(self):
        """The whole point of E%-WTA (Hoff et al. 2026) is that |F| is data
        dependent. On `numpy_sparse` it came out identically equal to the
        recruited pool (#94); on exact drive it settles at a genuine
        selection. This engine has exact drive by construction."""
        b = self._drive(EPercentPolicy(fraction_of_max=0.5))
        size = len(b.areas["A"].winners)
        assert 0 < size < N, f"|F|={size} is degenerate against n={N}"
        assert size != K, (
            "|F| equals k exactly -- the policy is not being applied, it is "
            "falling through to plain k-WTA")

    @pytest.mark.parametrize("seed", [5, 6, 7])
    def test_epercent_does_not_track_a_growing_pool(self, seed):
        """The #94 failure signature, asserted against directly.

        On `numpy_sparse` |F| came out as 50, 100, 150, 200, 250, 300 -- the
        recruited pool exactly, growing by k every round, because the sampler
        partitions by materialization status rather than by drive. Here |F|
        must not be that arithmetic progression.

        DELIBERATELY NOT ASSERTED: that |F| converges. Measured over 14 rounds
        at n=300 it wanders and trends DOWN without a fixed point -- seed 5
        69->19, seed 6 75->11, seed 7 128->61. `numpy_exact` at n=2000/k=50 was
        recorded as flat at 184, but that is a different scale and a different
        norm_init, so the two are not in contradiction and neither is a
        settled fact. Asserting convergence here would pin a realization
        ([[ensemble-not-realization]]).
        """
        b = Brain(p=P, seed=seed, engine="numpy_explicit", norm_init=False)
        b.add_stimulus("S", K)
        b.add_area("A", N, K, BETA,
                   winner_policy=EPercentPolicy(fraction_of_max=0.5))
        b.project({"S": ["A"]}, {})
        sizes = []
        for _ in range(6):
            b.project({"S": ["A"]}, {"A": ["A"]})
            sizes.append(len(b.areas["A"].winners))
        assert sizes != [K * (i + 1) for i in range(len(sizes))], (
            f"|F| is exactly k per round: {sizes} -- this is the recruitment "
            f"signature, not a selection")
        assert not all(b > a for a, b in zip(sizes, sizes[1:])), (
            f"|F| increases every single round: {sizes} -- consistent with "
            f"tracking a growing pool rather than the drive distribution")
        assert all(0 < s < N for s in sizes), f"degenerate sizes: {sizes}"

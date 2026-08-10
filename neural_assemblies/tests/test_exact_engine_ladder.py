"""The acceptance ladder for `numpy_exact` (task #85).

Ordered so each rung is falsifiable on its own, and named so a failure says
which claim broke. L0-L2 are NECESSARY but not sufficient: an engine can pass
all three and still merge disjoint inputs, which is the defect the engine
exists to remove. **L3 is the acceptance test.**

  L0  same substrate   initial weights identical to the explicit engine
  L1  same drive       pre-kWTA drive identical, both norm_init settings
  L2  same dynamics    winners identical over a multi-round protocol
  L3  THE POINT        graded similarity: chance for disjoint inputs, not 0.906
"""
from __future__ import annotations

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.numpy_engine._exact import NumpyExactEngine
from neural_assemblies.core.numpy_engine._explicit import NumpyExplicitEngine
from neural_assemblies.core.numpy_engine._sparse import NumpySparseEngine

N, K, P, BETA, SEED = 600, 30, 0.1, 0.05, 42


def _exact(norm_init=True, n=N, k=K, seed=SEED):
    e = NumpyExactEngine(p=P, seed=seed, norm_init=norm_init)
    e.add_stimulus("s", k)
    e.add_area("A", n, k, BETA)
    e.add_area("B", n, k, BETA)
    return e


def _explicit(norm_init=True, n=N, k=K, seed=SEED):
    b = Brain(p=P, seed=seed, engine="numpy_sparse", norm_init=norm_init)
    b.add_stimulus("s", k)
    b.add_area("A", n, k, BETA)
    b.add_area("B", n, k, BETA)
    b._engine.materialize_area("A")
    b._engine.materialize_area("B")
    return b


# -- L0 -- same substrate ---------------------------------------------------

class TestL0Substrate:
    """Initial weights must be the SAME NUMBERS, not merely the same law."""

    def test_stimulus_fiber_cannot_be_compared_elementwise_and_here_is_why(self):
        """The sparse engine's stim fiber is STREAM-drawn, so parity is not
        available on this path -- and that is a defect there, not here.

        Door 6 ("stim->area init was still drawn from the stream") was reverted
        pending this engine. Measured: declaring three UNUSED stimuli rewrites
        the s->A fiber in `numpy_sparse` (97/600 values survive) while leaving
        this engine bit-identical. So an elementwise assertion against the
        sparse engine would be asserting that we reproduce draw-order
        dependence. Test the property instead.
        """
        def sparse(extra):
            b = Brain(p=P, seed=SEED, engine="numpy_sparse")
            for i in range(extra):
                b.add_stimulus(f"x{i}", K)
            b.add_stimulus("s", K)
            b.add_area("A", N, K, BETA)
            b._engine.materialize_area("A")
            return np.asarray(b._engine._stim_conns["s"]["A"].weights,
                              dtype=np.float64)[:N]

        def exact(extra):
            e = NumpyExactEngine(p=P, seed=SEED)
            for i in range(extra):
                e.add_stimulus(f"x{i}", K)
            e.add_stimulus("s", K)
            e.add_area("A", N, K, BETA)
            return e._stim_base["s"]["A"]

        assert not np.array_equal(sparse(0), sparse(3)), (
            "numpy_sparse became order-independent on the stimulus fiber -- if "
            "door 6 has landed, this test and L1/L2 should compare elementwise")
        assert np.array_equal(exact(0), exact(3)), (
            "declaring an unused stimulus changed this engine's substrate; "
            "content addressing is broken")

    def test_stimulus_afferent_counts_have_the_right_law(self):
        """What CAN be asserted absolutely: it is Binomial(stim_size, p)."""
        e = _exact(n=20_000)
        base = e._stim_base["s"]["A"]
        assert base.mean() == pytest.approx(K * P, rel=0.05)
        assert base.var() == pytest.approx(K * P * (1 - P), rel=0.15)

    @pytest.mark.parametrize("row", [0, 7, N - 1])
    def test_area_fiber_rows_are_identical(self, row):
        b, e = _explicit(), _exact()
        conn = np.asarray(b._engine._area_conns["A"]["B"].weights, dtype=np.float64)
        mine = e._fiber_rows("A", "B", np.array([row]), N)[0]
        assert np.array_equal(conn[row, :N], mine)

    def test_recomputation_is_order_independent(self):
        """Doors 5 and 6 are gone BY CONSTRUCTION, not by patch."""
        e = _exact()
        rows = np.array([5, 2, 9])
        forward = e._fiber_rows("A", "B", rows, N)
        backward = e._fiber_rows("A", "B", rows[::-1], N)[::-1]
        assert np.array_equal(forward, backward)
        # and asking for one row alone gives the same numbers as in a block
        assert np.array_equal(e._fiber_rows("A", "B", np.array([2]), N)[0],
                              forward[1])


# -- L1 -- same drive -------------------------------------------------------

def _seed_source(b, e, pattern):
    """Put the SAME source assembly in both engines, bypassing the stimulus.

    The stimulus path cannot be compared elementwise (see L0), so the source is
    set directly. Legal because the area-fiber substrates ARE identical, and
    the engines index that fiber's rows the same way -- which is exactly what
    `test_area_fiber_rows_are_identical` establishes.
    """
    b._engine.set_winners("A", np.asarray(pattern, dtype=np.uint32))
    e.set_winners("A", np.asarray(pattern, dtype=np.uint32))


def _ab_drive(e, pattern, norm_init):
    """The exact A->B drive vector this engine computes, untouched by k-WTA."""
    block = e._fiber_rows("A", "B", np.asarray(pattern, dtype=np.int64), N)
    d = block.sum(axis=0)
    if norm_init:
        d = d * e._area_norm("A", "B")
    return d


def _assert_same_drive_modulo_ties(got, mine, drive, k, label):
    """The precise claim: the DRIVE agrees; only the tied band may differ.

    A raw overlap threshold cannot express this. With `norm_init=False` the
    drive is an integer afferent count, so the k-th boundary lands inside a
    large exactly-tied band (measured here: 30 neurons tied for 11 slots at
    k*p=3, and raising k*p does NOT help because the tie is intrinsic to a
    discrete drive). Whichever neurons come out of that band is decided by
    convention, not by the model -- so requiring identical winners would be
    requiring identical conventions, which is not what parity means.

    `norm_init=True` divides by a per-neuron 1/d_j, making the drive
    real-valued, and there the winners DO match exactly.
    """
    boundary = np.sort(drive)[-k]
    strictly_above = set(np.flatnonzero(drive > boundary).tolist())

    assert strictly_above <= got, (
        f"{label}: sparse dropped a neuron whose drive strictly exceeds the "
        f"boundary -- that is a drive disagreement, not a tie-break")
    assert strictly_above <= mine, (
        f"{label}: exact dropped a strictly-above-boundary neuron")
    for name, sel in (("sparse", got), ("exact", mine)):
        below = [i for i in sel if drive[i] < boundary]
        assert not below, (
            f"{label}: {name} selected {len(below)} neurons BELOW the boundary "
            f"drive; the two engines are not computing the same drive")


class TestL1Drive:
    """Pre-kWTA drive over the AREA fiber, read through the winners."""

    @pytest.mark.parametrize("norm_init", [False, True])
    def test_area_projection_agrees_on_the_drive(self, norm_init):
        b, e = _explicit(norm_init), _exact(norm_init)
        pattern = np.arange(K, dtype=np.uint32) * 3      # arbitrary, shared
        _seed_source(b, e, pattern)
        drive = _ab_drive(e, pattern, norm_init)

        b.project({}, {"A": ["B"]})
        got = set(np.array(b.areas["B"].winners, dtype=np.int64).tolist())
        mine = set(e.project_into("B", [], ["A"]).winners.astype(np.int64).tolist())

        _assert_same_drive_modulo_ties(got, mine, drive, K,
                                       f"norm_init={norm_init}")

    def test_normalised_drive_gives_EXACT_winner_parity(self):
        """No tied band under norm_init, so nothing is left to convention."""
        b, e = _explicit(True), _exact(True)
        pattern = np.arange(K, dtype=np.uint32) * 3
        _seed_source(b, e, pattern)
        b.project({}, {"A": ["B"]})
        got = np.sort(np.array(b.areas["B"].winners, dtype=np.int64))
        mine = np.sort(e.project_into("B", [], ["A"]).winners.astype(np.int64))
        assert np.array_equal(got, mine)


# -- L2 -- same dynamics ----------------------------------------------------

class TestL2Dynamics:
    """Now with plasticity compounding: beta, w_max and the norm scale."""

    def test_repeated_plastic_projection_stays_identical_under_norm_init(self):
        """Six plastic rounds, exact winner parity throughout.

        This is the rung that would catch a `w_max` clamp applied on the wrong
        scale, or a potentiation exponent that drifts -- both compound, so a
        single-round test has no power against them.
        """
        b, e = _explicit(True), _exact(True)
        pattern = np.arange(K, dtype=np.uint32) * 3
        for rnd in range(6):
            _seed_source(b, e, pattern)          # hold the source fixed
            b.project({}, {"A": ["B"]})
            e.project_into("B", [], ["A"])
            got = np.sort(np.array(b.areas["B"].winners, dtype=np.int64))
            mine = np.sort(np.asarray(e._areas["B"].winners, dtype=np.int64))
            assert np.array_equal(got, mine), (
                f"diverged at plastic round {rnd}: overlap "
                f"{len(set(got.tolist()) & set(mine.tolist()))}/{K}")


# -- L3 -- THE ACCEPTANCE TEST ----------------------------------------------

class TestL3GradedSimilarity:
    """The rung that justifies the engine existing.

    `numpy_sparse` gives 0.906 overlap for FULLY DISJOINT inputs at low area
    load where the exact substrate gives chance (measured 18.0x chance +/-
    0.0231 at 8 seeds, `research/notes/graded_similarity_and_sampler_load.md`).
    L0-L2 can all pass on an engine that still does that.
    """

    @staticmethod
    def _read(e, pattern, n):
        e._areas["A"].winners = np.asarray(pattern, dtype=np.uint32)
        e._areas["A"].fixed_assembly = True
        e.project_into("B", [], ["A"], plasticity_enabled=False)
        return set(np.asarray(e._areas["B"].winners, dtype=np.int64).tolist())

    def test_disjoint_inputs_land_at_chance(self):
        e = _exact()
        e.project_into("B", ["s"], [])          # give B a population
        rng = np.random.default_rng(7)
        pool = rng.permutation(N)
        a = self._read(e, pool[:K], N)
        b = self._read(e, pool[K:2 * K], N)
        ov = len(a & b) / K
        chance = K / N
        assert ov < 4 * chance, (
            f"disjoint inputs overlap {ov:.4f} against chance {chance:.4f} "
            f"({ov/chance:.1f}x). The sparse engine reads 18x here; if this "
            f"engine does too it has not fixed the defect it exists to fix.")

    def test_similarity_is_graded_and_monotone(self):
        """Not just "disjoint is fine" -- the whole curve must be ordered."""
        e = _exact()
        e.project_into("B", ["s"], [])
        rng = np.random.default_rng(7)
        pool = rng.permutation(N)
        ref, disjoint = list(pool[:K]), list(pool[K:])
        base = self._read(e, ref, N)
        curve = []
        for f in (0.0, 0.25, 0.5, 0.75, 1.0):
            shared = int(round(f * K))
            pat = ref[:shared] + disjoint[:K - shared]
            curve.append(len(self._read(e, pat, N) & base) / K)
        assert curve[-1] == 1.0, "identical inputs must give identical output"
        assert all(x <= y + 1e-9 for x, y in zip(curve, curve[1:])), (
            f"similarity is not monotone in shared fraction: {curve}")
        assert curve[-2] > curve[0] + 0.1, (
            f"curve is flat, so nothing is graded: {curve}")


# -- the equivalences the proof sketches claim -----------------------------

class TestSelectionEquivalences:
    """`research/notes/exact_drive_equivalences.md` argues these hold in
    general; these pin the cases a proof cannot catch (an off-by-one in the
    refinement, a tie-break that stops being index-ascending)."""

    @staticmethod
    def _cases():
        rng = np.random.default_rng(0)
        out = []
        for i in range(36):
            n = int(rng.integers(9_000, 40_000))
            k = int(rng.integers(5, 400))
            kind = i % 6
            if kind == 0:
                d = rng.poisson(10, n).astype(np.float32)
            elif kind == 1:                       # ties straddling the boundary
                d = np.zeros(n, dtype=np.float32); d[:k * 3] = 1.0
            elif kind == 2:                       # every value identical
                d = np.ones(n, dtype=np.float32)
            elif kind == 3:                       # mass aligned with the STRIDE
                d = np.zeros(n, dtype=np.float32); d[::64] = 5.0
            elif kind == 4:
                d = rng.exponential(1, n).astype(np.float32)
            else:                                 # mass at the far index end
                d = np.zeros(n, dtype=np.float32); d[-k // 2:] = 9.0
            out.append((d, k))
        return out

    def test_pivot_path_equals_full_selection(self):
        """Claim 1: |C| >= k implies the top-k is inside C, whatever the pivot."""
        e = _exact()
        for d, k in self._cases():
            fast = np.sort(e._select(d, k))
            slow = np.sort(e._exact_topk(d, k))
            assert np.array_equal(fast, slow), (
                f"pivot path diverged at n={d.size}, k={k}")

    def test_tie_break_is_highest_drive_then_lowest_index(self):
        """The half of the total order a value-only argument cannot give."""
        e = _exact()
        d = np.zeros(20_000, dtype=np.float32)
        d[[5, 9, 100]] = 3.0
        d[200:400] = 1.0
        assert e._select(d, 5).tolist() == [5, 9, 100, 200, 201]

    def test_event_application_order_is_deterministic(self):
        """Claim 2's caveat: float multiply is not associative, so the order
        events are applied in is part of the contract, not an accident.

        Asserted on BEHAVIOUR, not on source text -- the previous version
        grepped for `sorted(touched)` and broke the moment the sort moved into
        a helper, while the guarantee was still intact.
        """
        from neural_assemblies.core.numpy_engine._exact import _Potentiation
        pot = _Potentiation()
        rng = np.random.default_rng(3)
        for _ in range(8):
            src = np.sort(rng.choice(500, 40, replace=False)).astype(np.int64)
            tgt = np.sort(rng.choice(500, 40, replace=False)).astype(np.int64)
            pot.bump(src, tgt)
        probe = np.sort(rng.choice(500, 60, replace=False)).astype(np.int64)
        order = pot._touched(probe)
        assert order == sorted(order), (
            "events are applied in an unsorted order; float multiplication is "
            "not associative, so that makes the last ulp depend on set "
            "iteration order")

    def test_positions_handles_unsorted_rows(self):
        """`searchsorted` on an unsorted array fails SILENTLY, and winners are
        not guaranteed sorted -- `set_winners` takes any order."""
        from neural_assemblies.core.numpy_engine._exact import _Potentiation
        rows = np.array([50, 3, 900, 17], dtype=np.int64)
        members = np.array([3, 900], dtype=np.int64)
        pos = _Potentiation._positions(rows, members)
        assert sorted(rows[pos].tolist()) == [3, 900]


# -- L5 -- integrated -------------------------------------------------------

class TestL5Integration:
    """The engine must be REACHABLE, and must refuse what it cannot do.

    L0-L3 test the engine object directly. None of them would notice that it
    is unregistered, that `Brain` cannot construct it, or that it silently
    drops a mechanism `Brain` asked for -- which is the failure mode that has
    cost this repo the most, because the run still completes and still returns
    a number ([[silent-no-op-dead-fibers]]).
    """

    def test_registered_and_constructible_through_brain(self):
        from neural_assemblies.core.engine import create_engine, ensure_engine

        assert ensure_engine("numpy_exact"), "engine not in the registry"
        assert create_engine("numpy_exact", p=P, seed=SEED).name == "numpy_exact"

        b = Brain(p=P, seed=SEED, engine="numpy_exact")
        b.add_stimulus("s", K)
        b.add_area("A", N, K, BETA)
        b.add_area("B", N, K, BETA)
        for _ in range(3):
            b.project({"s": ["A"]}, {})
        b.project({}, {"A": ["B"]})
        assert b.engine_name == "numpy_exact"
        assert len(b.areas["A"].winners) == K
        assert len(b.areas["B"].winners) == K

    @pytest.mark.parametrize("kwargs", [
        {"refractory_period": 3},
        {"inhibition_strength": 0.5},
        {"input_noise_std": 0.1},
    ])
    def test_unimplemented_area_mechanisms_raise_not_ignored(self, kwargs):
        """A mechanism this engine lacks must be an ERROR, never a no-op.

        Silently accepting `refractory_period=3` would produce a run that looks
        like LRI and has none, and no assertion downstream distinguishes that
        from LRI that simply did not help.
        """
        b = Brain(p=P, seed=SEED, engine="numpy_exact")
        with pytest.raises(NotImplementedError, match="does not implement"):
            b.add_area("A", N, K, BETA, **kwargs)

    def test_defaults_are_not_treated_as_requests(self):
        """Passing the default explicitly must NOT raise -- it asks for nothing."""
        b = Brain(p=P, seed=SEED, engine="numpy_exact")
        b.add_area("A", N, K, BETA, refractory_period=0, inhibition_strength=0.0)
        assert "A" in b.areas

    def test_record_activation_returns_the_pre_kwta_drive(self):
        """The ERP components read this hook; an empty result reads as zero.

        Asserted against the winners: every winner's recorded drive must be at
        least the k-th largest value, which is exactly what k-WTA selected on.
        """
        e = _exact()
        for _ in range(3):
            e.project_into("A", ["s"], [], plasticity_enabled=True)
        r = e.project_into("A", ["s"], [], plasticity_enabled=False,
                           record_activation=True)
        assert r.pre_kwta_inputs is not None
        assert r.pre_kwta_inputs.shape == (N,)
        assert r.pre_kwta_total == pytest.approx(float(r.pre_kwta_inputs.sum()),
                                                 rel=1e-6)
        kth = np.sort(r.pre_kwta_inputs)[-K]
        assert (r.pre_kwta_inputs[r.winners] >= kth).all()

    def test_arbiter_exposes_an_exact_arm(self):
        from neural_assemblies import diagnostics as dx

        assert "exact" in dx.ARBITER_ARMS
        assert dx.arm_spec("exact").brain_kwargs["engine"] == "numpy_exact"
        with pytest.raises(ValueError, match="unknown arm"):
            dx.arm_spec("no-such-arm")

    def test_arbitrate_runs_every_arm_through_one_builder(self):
        """The builder must not need to know the arm names.

        That is the whole point of the `Arm` object: `explicit` selects a
        dense engine via `area_kwargs`, `exact` selects a different engine via
        `brain_kwargs`, and this builder branches on neither.
        """
        from neural_assemblies import diagnostics as dx

        def build(arm):
            b = Brain(p=P, seed=SEED, **arm.brain_kwargs)
            b.add_stimulus("s", K)
            b.add_area("A", N, K, BETA, **arm.area_kwargs)
            for _ in range(3):
                b.project({"s": ["A"]}, {})
            return b

        got = dx.arbitrate(build, lambda b: len(b.areas["A"].winners),
                           label="k winners")
        assert set(got.by_arm) == set(dx.ARBITER_ARMS)
        assert all(v == K for v in got.by_arm.values())


def test_engine_norm_init_contract():
    """Every engine accepting `norm_init` must DEFAULT it to False.

    `Brain` forwards the kwarg only when True, so omission is how it says
    False. An engine defaulting to True silently upgrades
    `Brain(norm_init=False)` to the production substrate -- which is precisely
    the runs that pinned it off deliberately (literature parity), and the
    upgrade appears nowhere in any log.

    This is a CONTRACT test, not an exact-engine test: it walks the registry so
    the next engine added inherits the check.
    """
    import inspect

    from neural_assemblies.core.engine import _ENGINE_MODULES, ensure_engine
    from neural_assemblies.core.engine import _ENGINE_REGISTRY

    checked = []
    for engine_name in _ENGINE_MODULES:
        if not ensure_engine(engine_name):
            continue          # optional backend (cupy/torch) not installed
        params = inspect.signature(_ENGINE_REGISTRY[engine_name]).parameters
        if "norm_init" not in params:
            continue
        checked.append(engine_name)
        assert params["norm_init"].default is False, (
            f"{engine_name} defaults norm_init to "
            f"{params['norm_init'].default!r}; Brain omits the kwarg to mean "
            f"False, so this engine ignores Brain(norm_init=False)")

    assert "numpy_sparse" in checked and "numpy_exact" in checked, (
        f"contract checked only {checked} -- if an engine stopped taking "
        f"norm_init, this test has quietly lost its subject")


def test_brain_norm_init_reaches_every_engine():
    """The contract above is about defaults; this one is about the wiring."""
    for engine_name in ("numpy_sparse", "numpy_exact"):
        for want in (True, False):
            b = Brain(p=P, seed=SEED, engine=engine_name, norm_init=want)
            assert b._engine.norm_init is want, (
                f"Brain(norm_init={want}) gave {engine_name} "
                f"norm_init={b._engine.norm_init}")


class TestPerFiberConnectivity:
    """`add_connectivity` is real on this engine, and loud on the others.

    Mitropolsky & Papadimitriou (2025) do not give every fiber the same
    density -- four fibers carry "increased parameters beta AND p", and that
    asymmetry is what makes the noun/verb split emerge without a label. The
    method to express it was already on the engine interface, documented in
    `engine.py` with a worked example, and was `pass` in all three engines.
    """

    @staticmethod
    def _built(fiber_p, n=2000, k=50, seed=3):
        b = Brain(p=P, seed=seed, engine="numpy_exact")
        b.add_area("A", n, k, beta=0.1)
        b.add_stimulus("s", k)
        if fiber_p is not None:
            b._engine.add_connectivity("s", "A", fiber_p)
        for _ in range(5):
            b.project({"s": ["A"]}, {})
        return b

    def test_raising_p_raises_in_degree_proportionally(self):
        """The knob must reach the substrate, not just the bookkeeping."""
        lo = self._built(None)._engine._stim_base["s"]["A"].mean()
        hi = self._built(0.30)._engine._stim_base["s"]["A"].mean()
        assert hi / lo == pytest.approx(0.30 / P, rel=0.05), (
            f"per-fiber p did not reach the in-degree: {lo:.3f} -> {hi:.3f}, "
            f"ratio {hi / lo:.2f} against the requested {0.30 / P:.2f}")

    def test_it_changes_which_neurons_win(self):
        """A density that does not move the assembly is not doing anything."""
        a = set(self._built(None).areas["A"].winners.tolist())
        b = set(self._built(0.30).areas["A"].winners.tolist())
        assert len(a & b) / len(a) < 0.25, (
            f"p=0.05 and p=0.30 on the same fiber elected overlapping "
            f"assemblies ({len(a & b) / len(a):.2f}) -- suspect a no-op")

    def test_default_is_the_global_p(self):
        e = Brain(p=P, seed=SEED, engine="numpy_exact")._engine
        assert e._p_of("anything", "at_all") == P

    def test_refuses_to_change_the_substrate_after_traffic(self):
        """p decides which synapses EXIST; changing it under written weights
        would leave potentiation on synapses that no longer do."""
        b = self._built(None)
        with pytest.raises(RuntimeError, match="structural"):
            b._engine.add_connectivity("s", "A", 0.2)

    def test_restating_the_global_p_after_traffic_is_allowed(self):
        """Not a change, so not an error -- otherwise the guard would fire on
        callers that are merely being explicit."""
        self._built(None)._engine.add_connectivity("s", "A", P)

    def test_rejects_an_impossible_probability(self):
        e = Brain(p=P, seed=SEED, engine="numpy_exact")._engine
        with pytest.raises(ValueError):
            e.add_connectivity("s", "A", 1.5)

    @pytest.mark.parametrize("cls", [NumpyExplicitEngine])
    def test_other_engines_refuse_rather_than_ignore(self, cls):
        """The original bug: `pass` everywhere, so a caller that set a
        per-fiber density silently got the global one.

        `NumpySparseEngine` was on this list until it IMPLEMENTED per-fiber
        connectivity (e97a280); refusing is only the right behaviour for an
        engine that cannot do it. Its own coverage is in
        `test_per_fiber_density.py`, which asserts the density is realized
        rather than merely accepted -- the failure mode this test guards is
        acceptance without effect, and "raises" and "works" are both cures.

        Engines are built DIRECTLY rather than through `Brain`, because
        `numpy_explicit` cannot be reached through `Brain.add_area` at all --
        it does not accept `winner_policy`. That is a separate gap; going
        around it here keeps this test about connectivity.
        """
        e = cls(p=P, seed=SEED)
        with pytest.raises(NotImplementedError, match="per-fiber"):
            e.add_connectivity("s", "A", 0.2)

    @pytest.mark.parametrize("cls", [NumpySparseEngine, NumpyExplicitEngine])
    def test_other_engines_accept_the_global_p(self, cls):
        """Asking for what is already true is not a request."""
        cls(p=P, seed=SEED).add_connectivity("s", "A", P)


class TestFixedTargetPlasticity:
    """A FIXED area still learns. Only its winners are pinned.

    `numpy_exact` used to return from `project_into` on a fixed target having
    applied no plasticity at all. `numpy_sparse` deliberately does the opposite
    and says why at `_fixed_target_plasticity_enabled`: the reference pins the
    winners, skips RECRUITMENT, and still potentiates the afferents. Skipping
    the potentiation silently breaks every protocol that writes INTO a held
    assembly -- `reciprocal_project`, `associate`, and the round trip the
    acquisition paper's Property 2 needs.

    The bug was invisible from outside: the projection returned a well-formed
    winner set, so the only symptom was that training bought nothing.
    """

    @staticmethod
    def _events(engine_name, rounds=10, gate=None):
        b = Brain(p=P, seed=5, engine=engine_name)
        b.add_area("A", 2000, 50, beta=0.2)
        b.add_area("B", 2000, 50, beta=0.2)
        b.add_stimulus("s", 50)
        for _ in range(5):
            b.project({"s": ["A"]}, {})
        for _ in range(5):
            b.project({}, {"A": ["B"]})
        b.areas["B"].fix_assembly()
        b._engine.fix_assembly("B")
        e = b._engine
        key = ("A", "B")

        def total():
            store = e._area_pot.get(key)
            return sum(store._events.values()) if store else 0

        before = total()
        for _ in range(rounds):
            b.project({}, {"A": ["B"]})
        return before, total(), b

    def test_afferents_potentiate_into_a_pinned_area(self):
        """The true negative this needs, constructed rather than assumed.

        Counts POTENTIATION EVENTS, not distinct (row, col) pairs: repeating
        the same (source, target) pairing raises multiplicity and leaves the
        pair count flat, so `len(_Potentiation)` reads unchanged and would pass
        while nothing was learned. That is the shape of the bug being tested.
        """
        before, after, _ = self._events("numpy_exact")
        assert after > before, (
            f"projecting into a fixed area applied no plasticity "
            f"({before} -> {after} events) -- the engine is back to "
            f"short-circuiting on tgt.fixed_assembly")

    def test_the_winners_really_are_pinned(self):
        """Learning must not come at the cost of the thing `fix` is for."""
        _, _, b = self._events("numpy_exact")
        held = set(b.areas["B"].winners.tolist())
        b.project({}, {"A": ["B"]})
        assert set(b.areas["B"].winners.tolist()) == held

    def test_the_env_gate_still_turns_it_off(self):
        """The A/B stays available, and both engines read ONE definition of it."""
        import os
        prev = os.environ.get("ASSEMBLIES_FIXED_TARGET_PLASTICITY")
        os.environ["ASSEMBLIES_FIXED_TARGET_PLASTICITY"] = "0"
        try:
            before, after, _ = self._events("numpy_exact")
            assert after == before, (
                f"gate off but plasticity still applied ({before} -> {after})")
        finally:
            if prev is None:
                os.environ.pop("ASSEMBLIES_FIXED_TARGET_PLASTICITY", None)
            else:
                os.environ["ASSEMBLIES_FIXED_TARGET_PLASTICITY"] = prev

    @pytest.mark.parametrize("engine_name", ["numpy_sparse", "numpy_exact"])
    def test_an_association_written_while_pinned_can_be_retrieved(
            self, engine_name):
        """The FUNCTIONAL consequence, and the one both engines must share.

        THE PROTOCOL MATTERS, and two earlier versions of this test could not
        have failed. Building B from A and then pinning it leaves A -> B
        already trained, so retrieval reads 1.000 whether or not the pinned
        rounds did anything -- a saturated metric whose null is not a null
        (#28's lesson). Here A and B are built from DIFFERENT stimuli, so the
        rounds while B is pinned are the ONLY A -> B training there has ever
        been. If they are discarded, retrieval must fall to chance.

        Measured, and this is the true negative the fix needed:

            gate on    sparse 0.660   exact 0.600
            gate off   sparse 0.000   exact 0.020     (chance k/n = 0.025)

        `exact` read 0.020 BEFORE the fix as well -- retrieval at chance was
        the entire visible symptom of an engine silently discarding the write.
        """
        from neural_assemblies.diagnostics import (assembly_overlap,
                                                   read_assembly)
        b = Brain(p=P, seed=5, engine=engine_name)
        b.add_area("A", 2000, 50, beta=0.2)
        b.add_area("B", 2000, 50, beta=0.2)
        b.add_stimulus("sa", 50)
        b.add_stimulus("sb", 50)
        for _ in range(6):
            b.project({"sa": ["A"]}, {})
        for _ in range(6):
            b.project({"sb": ["B"]}, {})
        pinned = read_assembly(b, "B")

        b.areas["B"].fix_assembly()
        b._engine.fix_assembly("B")
        for _ in range(10):
            b.project({}, {"A": ["B"]})
        b.areas["B"].unfix_assembly()
        b._engine.unfix_assembly("B")

        b.inhibit_areas(["B"])
        b.project({}, {"A": ["B"]})
        got = assembly_overlap(pinned, read_assembly(b, "B"))
        chance = 50 / 2000
        assert got > 20 * chance, (
            f"{engine_name}: an association written while the target was "
            f"PINNED does not retrieve ({got:.3f} against chance "
            f"{chance:.3f}) -- the write was discarded")


class TestWinnerPolicies:
    """E%-WTA on exact drive, and the degeneracy it exposes on the sampler.

    Hoff et al. (2026) replace k-WTA with a window on the drive distribution,
    `F = {j : h_j in [(1-eps) h_max, h_max]}`, so assembly SIZE is emergent.
    That is a claim about the distribution of the drive -- which makes it
    exactly the wrong thing to run on an engine that INVENTS the drive for
    neurons that have not fired.

    `numpy_exact` used to reject winner policies outright, so every E%-WTA
    result in this repository was sampler-side by default: `epwta.py`,
    `programs/markov_coin.py`, `parity/executors.py`, and two literature-golden
    tests.
    """

    POLICY_KW = dict(fraction_of_max=0.5, min_winners=1)

    @staticmethod
    def _sizes(engine_name, rounds=6):
        from neural_assemblies.compute import EPercentPolicy
        b = Brain(p=P, seed=3, engine=engine_name)
        b.add_area("A", 2000, 50, beta=0.1,
                   winner_policy=EPercentPolicy(fraction_of_max=0.5,
                                                min_winners=1))
        b.add_stimulus("s", 50)
        out = []
        for _ in range(rounds):
            b.project({"s": ["A"]}, {})
            out.append((len(b.areas["A"].winners), b.areas["A"].w))
        return out

    def test_exact_accepts_a_winner_policy_at_all(self):
        """It raised NotImplementedError before -- the whole of #94."""
        sizes = self._sizes("numpy_exact")
        assert len(sizes) == 6

    def test_exact_produces_a_genuinely_emergent_size(self):
        """Not k, and not the recruited pool: a property of the drive."""
        sizes = self._sizes("numpy_exact")
        firing = [f for f, _ in sizes]
        assert all(f != 50 for f in firing), (
            f"E%-WTA returned exactly k -- that is k-WTA wearing a hat: {firing}")
        assert len(set(firing)) == 1, (
            f"size should settle once the assembly is stable, got {firing}")

    def test_exact_size_is_a_real_selection(self):
        """Strictly between k and n, and stable: a competition, not a pool.

        NOT phrased as `|F| != area.w`, which was the first attempt and is
        meaningless here: `Brain.Area.w` tracks the last winner count while
        `engine.w` is n (184 against 2000 for the same area). `w` is two
        different quantities depending on which object is asked -- [[same-name-
        two-meanings]] and #83, caught for a second time while writing the test
        that was supposed to catch a different degeneracy.

        On this engine there IS no recruited pool: every neuron exists from
        t=0. So the honest discriminator is that the window selects a
        non-trivial subset and settles.
        """
        firing = [f for f, _ in self._sizes("numpy_exact")]
        assert all(50 < f < 2000 for f in firing), (
            f"E%-WTA selected nothing meaningful -- sizes {firing} against "
            f"k=50 and n=2000")

    def test_sampler_e_percent_is_degenerate_and_stays_pinned(self):
        """DOCUMENTS A DEFECT rather than asserting correctness.

        On `numpy_sparse`, |F| is IDENTICALLY `area.w` at every round: the
        whole recruited pool fires, so the "emergent assembly size" is a
        readout of recruitment, not of the drive distribution. Measured
        50/100/150/200/250/300 against w of 50/100/150/200/250/300.

        Pinned so the identity cannot quietly change without someone noticing
        -- and so that the next reader of an E%-WTA number knows which engine
        produced it. If this ever starts failing, the sampler's candidate drive
        has changed and every E%-WTA result needs re-deriving again.
        """
        for firing, w in self._sizes("numpy_sparse"):
            assert firing == w, (
                f"sampler E%-WTA no longer equals the recruited pool "
                f"({firing} vs {w}) -- re-derive the E%-WTA results and "
                f"update this test")

    def test_a_topk_policy_matching_k_is_the_plain_path(self):
        """The fast path must be exercised, and must agree with no policy."""
        from neural_assemblies.compute import TopKPolicy
        a = Brain(p=P, seed=3, engine="numpy_exact")
        a.add_area("A", 2000, 50, beta=0.1, winner_policy=TopKPolicy(k=50))
        a.add_stimulus("s", 50)
        b = Brain(p=P, seed=3, engine="numpy_exact")
        b.add_area("A", 2000, 50, beta=0.1)
        b.add_stimulus("s", 50)
        for _ in range(3):
            a.project({"s": ["A"]}, {})
            b.project({"s": ["A"]}, {})
        assert (set(a.areas["A"].winners.tolist())
                == set(b.areas["A"].winners.tolist()))

    def test_input_noise_is_still_refused_and_says_why(self):
        """Unsupported ON PURPOSE: it needs an RNG this engine does not have."""
        b = Brain(p=P, seed=3, engine="numpy_exact")
        with pytest.raises(NotImplementedError, match="input_noise_std"):
            b.add_area("A", 500, 20, beta=0.1, input_noise_std=0.5)

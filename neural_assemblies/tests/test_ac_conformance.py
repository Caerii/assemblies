"""Assembly Calculus CONFORMANCE suite -- do the primitives satisfy their definitions?

This is deliberately NOT a regression suite. Regression tests ask "does this
still do what it did yesterday". These ask "does this do what the PAPERS SAY IT
MUST DO". A primitive that runs, returns plausible numbers, and is used
throughout the codebase can still fail every test here -- and several currently
do, which is the point of writing them down.

Every assertion below cites the specific claim it encodes:

    [PNAS20] Papadimitriou, Vempala, Mitropolsky, Collins & Maass (2020),
             "Brain computation by assemblies of neurons", PNAS.
             research/literature/papers/papadimitriou2020_pnas.pdf
    [HOFF26] Hoff et al. (2026), E%-WTA. Eq. 10/11 for assembly density.
             research/literature/papers/hoff2026_epwta.pdf
    [NEMOREF] .reference/mdabagia-nemo/brain.py -- the NEMO reference.
    [ACREF]  .reference/dmitropolsky-assemblies -- the canonical Python AC
             implementation, and the ground truth for the primitives here.
    [COIN24] Dabagia, Papadimitriou & Vempala (2024), arXiv:2406.07715.
             Sec. 2 supplies the definition of an assembly used throughout.

These tags are not free-form: each resolves to a ``cite_tag`` in
``research/literature/index.json``, which holds the full reference and a
local PDF where one is checked in. ``test_literature_index.py`` fails if a
tag cited anywhere in the tree has no entry.

WHY THESE TESTS DID NOT EXIST BEFORE, and why that mattered: the library's
higher-level machinery (parser, roles, FSM, PFA coin, next-token) is built on
these primitives, so a primitive that silently violates its definition
invalidates everything above it rather than one call site. Two violations are
already known -- see the module-level notes on recurrence below -- and they went
undetected for a long time precisely because nothing asserted the definitions.

SCALE. [PNAS20] simulates n=1e7, k=1e4, p=0.001, beta=0.1, which is far beyond a
test suite. The parameters here preserve the properties the proofs rely on --
notably k < sqrt(n), which [PNAS20] assumes when it argues that two unrelated
assemblies share "very few cells" -- while staying fast. Where a test is
sensitive to scale, it says so, because "measured outside the regime where the
phenomenon exists" has already produced one false negative in this project.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.assembly_calculus.assembly import overlap
from neural_assemblies.assembly_calculus.ops import _snap

# k < sqrt(n) so that chance overlap is a few cells, per [PNAS20]'s assumption.
N = 10_000
K = 50            # sqrt(N) = 100, so k < sqrt(n) holds
P = 0.05
BETA = 0.1
SEEDS = (1, 2, 3, 4, 5)
# Association's seed-to-seed spread (sd 0.072 on a mean of 0.122) is close to
# the margin over the paper's 8% figure, so its mean needs more draws than the
# five that suffice elsewhere in this file. 12 puts the standard error of the
# mean at ~0.021, about 2 se below the threshold.
SEEDS_ASSOC = tuple(range(1, 13))
CHANCE = K / N    # 0.005


# These primitives are only expected to hold WITH `norm_init`. Established by
# elimination rather than assumption: the independence collapse is present at
# beta=0 (overlap 0.392 with plasticity OFF), so it is not caused by Hebbian
# learning; it does not track the unmaterialised fraction (flat 0.87-0.94 as n
# goes from 60% to 1.4% materialised), so it is not a lazy-engine artefact; and
# winners sit +1.94 sd above mean recurrent IN-DEGREE, so k-cap is selecting the
# random graph's hubs, which are shared by every stimulus. Normalising each
# postsynaptic neuron's incoming column once at construction removes the degree
# advantage: overlap 0.936 -> 0.000, z +1.94 -> +0.80.
#
# Left as an explicit flag rather than a default because turning it on shifts
# every existing result in the repo; see Brain.__init__.
NORM_INIT = True


def _brain(seed: int, **kw) -> Brain:
    kw.setdefault("norm_init", NORM_INIT)
    return Brain(p=P, seed=seed, **kw)


def _project_recurrent(brain, stim: str, area: str, rounds: int):
    """Project WITH target self-recurrence, per the project() protocol.

    Deliberately does NOT go through ``Brain.project_rounds``: that fast path
    drops target self-recurrence (``a != target``, core/brain.py), which is the
    very divergence several of these tests exist to detect. Driving
    ``brain.project`` directly is the protocol [PNAS20] actually specifies --
    stimulus alone on the first step, then stimulus + recurrence thereafter.
    """
    brain.project({stim: [area]}, {})
    for _ in range(rounds - 1):
        brain.project({stim: [area]}, {area: [area]})
    return _snap(brain, area)


def _mean_sd(vals):
    return float(np.mean(vals)), float(np.std(vals))


# ---------------------------------------------------------------------------
# 1. Projection converges
# ---------------------------------------------------------------------------

def test_projection_converges_to_a_stable_assembly():
    """[PNAS20] "We say that the process has converged when there are no new
    winners." Convergence is exponentially fast for high enough plasticity.

    Encoded as: the winner set stops changing. We use the paper's own criterion
    (no NEW winners) rather than a similarity threshold, and additionally
    require consecutive-round overlap >= 0.95, which is the figure this repo's
    own ``project`` docstring cites.
    """
    finals = []
    for seed in SEEDS:
        b = _brain(seed)
        b.add_area("A", N, K, beta=BETA)
        b.add_stimulus("s", K)
        b.project({"s": ["A"]}, {})
        prev = set(int(x) for x in _snap(b, "A").winners)
        overlaps = []
        for _ in range(14):
            b.project({"s": ["A"]}, {"A": ["A"]})
            cur = set(int(x) for x in _snap(b, "A").winners)
            overlaps.append(len(prev & cur) / max(len(prev), 1))
            prev = cur
        finals.append(overlaps[-1])
    m, sd = _mean_sd(finals)
    assert m >= 0.95, (
        f"projection did not stabilise: consecutive-round overlap "
        f"{m:.3f} +/- {sd:.3f}, expected >= 0.95 [PNAS20]"
    )


# ---------------------------------------------------------------------------
# 2. Independence -- the property that motivated disabling recurrence
# ---------------------------------------------------------------------------

def test_independent_stimuli_form_near_disjoint_assemblies():
    """[PNAS20] two unrelated assemblies in one area share "very few cells"
    when k < sqrt(n).

    KNOWN TO FAIL at the time of writing. Measured with true recurrence:
    overlap 0.240 at 5 rounds and 0.940 at 15 rounds against chance 0.005.
    [NEMOREF] ``RecurrentArea`` holds 0.024 at 15 rounds when ``norm_init=True``
    normalises each postsynaptic neuron's incoming column to sum 1; without it,
    k-cap systematically selects the random graph's high-in-degree hubs
    (winner in-degree z-score +0.52 vs -0.27) and every stimulus converges on
    the same cells. The tolerance below is generous -- 10x chance -- so that it
    passes for any implementation that is merely imperfect rather than broken.
    """
    ovs = []
    for seed in SEEDS:
        b = _brain(seed)
        b.add_area("A", N, K, beta=BETA)
        b.add_stimulus("s0", K)
        b.add_stimulus("s1", K)
        a0 = _project_recurrent(b, "s0", "A", rounds=15)
        a1 = _project_recurrent(b, "s1", "A", rounds=15)
        ovs.append(overlap(a0, a1))
    m, sd = _mean_sd(ovs)
    assert m <= 10 * CHANCE, (
        f"independent stimuli collapsed onto the same neurons: overlap "
        f"{m:.3f} +/- {sd:.3f}, chance {CHANCE:.4f}. Assemblies are not "
        f"distinct, so nothing built on them can be. [PNAS20]"
    )


# ---------------------------------------------------------------------------
# 3. Assemblies are attractors -- pattern completion
# ---------------------------------------------------------------------------

def test_pattern_completion_from_a_partial_assembly():
    """[PNAS20] Fig. 2F2: fire 40% of an assembly's neurons and "the subset
    recovers nearly all of the original assembly".

    REINFORCEMENT MATTERS, and this is [PNAS20]'s own point ("With more
    reinforcement of the original assembly, the subset recovers nearly all").
    The recurrent self-attractor is a BASIN that has to be deep enough. At the
    formation used elsewhere in this file (15 rounds, beta=0.1) the basin is
    shallow -- a 40% cue recovers only ~0.45, and a formed assembly does not
    even self-sustain under pure recurrence (~0.40). That looked like a broken
    primitive, and an earlier version of this test was xfailed on it, but it was
    UNDER-REINFORCEMENT masked by the old degree-hub collapse (which made every
    assembly "self-sustain" trivially onto the hubs). With an adequately
    reinforced assembly -- 30 formation rounds here, or equivalently beta>=0.3 --
    completion reaches ~0.99, matching the paper. The reference NEMO behaves the
    same way. So this forms strongly on purpose.

    HOW THIS IS MEASURED, and why the obvious version lies. ``overlap`` is
    min-normalised (|A n B| / min(|A|,|B|)), so a cue that stays FROZEN as a
    strict subset of the assembly scores a perfect 1.0 while completing nothing
    -- the smaller set is trivially contained in the larger. An earlier version
    of this test (and the library's own ``ops.pattern_complete``, whose
    docstring admits the same floor) reported "success" that was exactly this
    artefact. So here:

      * recovery is |recovered n true| / K  -- a FIXED denominator, so a frozen
        subset scores only its own fraction, not 1.0;
      * the cue is HELD FIRING every round (union'd back before each
        projection), which is what [PNAS20] means by the subset "firing for a
        number of steps" -- releasing it after one round, or pinning it with
        ``_fix`` (a projection INTO a fixed area is short-circuited before
        plasticity, the merge bug), both prevent completion;
      * the bar is EXCESS OVER THE CUE: genuine completion recruits correct
        members the cue did not contain.

    Plasticity is disabled during recall so this reads the stored assembly
    rather than training it (``ops.pattern_complete`` leaves plasticity ON,
    which strengthens the very thing it measures).
    """
    cue_frac = 0.4
    recovered, excess = [], []
    for seed in SEEDS:
        b = _brain(seed)
        b.add_area("A", N, K, beta=BETA)
        b.add_stimulus("s", K)
        # 30 rounds, not the 15 used elsewhere: a deep enough basin for the
        # attractor to complete. See the docstring.
        asm = set(int(x) for x in _project_recurrent(b, "s", "A", rounds=30).winners)

        compact = list(b.areas["A"].winners)
        rng = np.random.default_rng(seed)
        cue = list(rng.choice(compact, size=max(1, int(cue_frac * K)),
                              replace=False))

        saved = b.disable_plasticity
        b.disable_plasticity = True
        try:
            cur = cue[:]
            for _ in range(6):
                arr = np.array(cur, dtype=np.uint32)
                b.areas["A"].winners = arr
                b._engine.set_winners("A", arr)
                b.project({}, {"A": ["A"]})
                cur = list(set(int(x) for x in b.areas["A"].winners) | set(cue))
        finally:
            b.disable_plasticity = saved
        rec = set(int(x) for x in _snap(b, "A").winners)
        frac = len(rec & asm) / K
        recovered.append(frac)
        excess.append(frac - cue_frac)
    m, sd = _mean_sd(recovered)
    ex, _ = _mean_sd(excess)
    assert m >= 0.75, (
        f"pattern completion below [PNAS20 Fig 2F2]: a {int(cue_frac*100)}% cue "
        f"recovered {m:.3f} +/- {sd:.3f} of the assembly (excess over cue "
        f"{ex:+.3f}); the paper's 'nearly all' means >= 0.75. The attractor is "
        f"too weak to recruit the missing members."
    )


# ---------------------------------------------------------------------------
# 4. Assembly density exceeds baseline connectivity
# ---------------------------------------------------------------------------

def test_assembly_density_exceeds_baseline_p():
    """[PNAS20] Fig. 2C: "Synaptic density within the resulting assembly ... is
    always higher than the baseline random synaptic connectivity p".
    [HOFF26] Eq. 10 makes D_A > D_M = p a FORMATION CONDITION, and Eq. 11
    defines D = |S| / (|N|(|N|-1)) on the directed graph.

    Hebb's original conjecture, and the reason assemblies fire synchronously.
    """
    from neural_assemblies.assembly_calculus.epwta import assembly_density

    dens = []
    for seed in SEEDS:
        b = _brain(seed)
        b.add_area("A", N, K, beta=BETA)
        b.add_stimulus("s", K)
        asm = _project_recurrent(b, "s", "A", rounds=15)
        conn = b._engine._area_conns.get("A", {}).get("A")
        if conn is None or np.asarray(conn.weights).size == 0:
            pytest.fail(
                "recurrent fiber A->A was never materialised, so assembly "
                "density is identically zero -- there are no intra-assembly "
                "synapses to count. [PNAS20 Fig 2C]"
            )
        # `asm.winners` holds REAL neuron ids; the connectome is indexed by
        # COMPACT ids. Filtering real ids against the matrix bounds silently
        # discards almost every member and leaves fewer than two, which
        # `assembly_density` correctly reports as 0.0 -- a false failure that
        # looked exactly like "the fiber was never built". Map first.
        adj = np.asarray(conn.weights) > 0
        mapping = b._engine.get_neuron_id_mapping("A")
        real_to_compact = {int(nid): i for i, nid in enumerate(mapping)}
        idx = [real_to_compact[int(w)] for w in asm.winners
               if int(w) in real_to_compact
               and real_to_compact[int(w)] < min(adj.shape)]
        assert len(idx) >= 2, "too few assembly members mapped into the connectome"
        dens.append(assembly_density(adj, idx))
    m, sd = _mean_sd(dens)
    assert m > P, (
        f"assembly is no denser than the ambient graph: D_A = {m:.4f} +/- "
        f"{sd:.4f} vs p = {P}. [PNAS20 Fig 2C; HOFF26 Eq. 10]"
    )


# ---------------------------------------------------------------------------
# 5. Association raises overlap
# ---------------------------------------------------------------------------

def test_association_increases_overlap_substantially():
    """[PNAS20] simultaneous firing of two parents makes their projections'
    overlap "increase substantially"; experimentally 8-10% of assembly size.

    THE CLAIM IS ABOUT RE-CUEING, and this test used to measure something else.
    PNAS20 says that after association, activating source_a ALONE and projecting
    to the target yields an assembly overlapping the one source_b alone yields.
    The old version never re-cued: it compared the phase-3 CO-FIRED target
    assembly (both sources driving at once) against a snapshot taken BEFORE
    association, which association moves away from -- so it read 0.000 once the
    op started working. It had previously read 1.0000, because the A->C and B->C
    blocks were unmaterialised and every phase hit the engine's zero-drive
    branch, which preserves the target's winners.

    Its baseline was wrong too. Comparing against a PRE-association overlap is
    meaningless here: before phase 1 the source->target fiber does not exist, so
    cueing either source just lets the target fall back on its own dominant
    attractor and both cues land there -- that measures 0.678 at k=100 and is
    target-attractor dominance, not association. The CONTROL has to train both
    pathways exactly as phases 1 and 2 do and then omit only the co-firing,
    which is what isolates the step the docstring credits.

    MEASURED over these 12 seeds (chance = k/n = 0.005), as overlap between the
    assembly cued by source_a alone and the one cued by source_b alone:

        pathways trained, NO co-firing   0.0167 +/- 0.0206    3.3x chance
        with co-firing (associate)       0.1217 +/- 0.0721   24.3x chance

    so co-firing raises it ~7x and clears the paper's 8-10%. Asserted on a mean:
    per-seed values run 0.04 to 0.28, so any single-seed threshold near 0.08
    reports the seed rather than the effect.

    The control is written out rather than calling associate() with co-firing
    disabled, deliberately: an independent implementation of the baseline does
    not move when the op is refactored.
    """
    from neural_assemblies.assembly_calculus.ops import (
        associate, _fix, _unfix,
    )

    def _recue(brain, src):
        """Target assembly when only `src` drives it, measured in isolation."""
        probe = copy.deepcopy(brain)
        probe.areas[src].fix_assembly()
        probe.project({}, {src: ["C"]})
        for _ in range(5):
            probe.project({}, {src: ["C"], "C": ["C"]})
        return _snap(probe, "C")

    def _setup(seed):
        b = _brain(seed)
        for area in ("A", "B", "C"):
            b.add_area(area, N, K, beta=BETA)
        b.add_stimulus("sa", K)
        b.add_stimulus("sb", K)
        _project_recurrent(b, "sa", "A", rounds=10)
        _project_recurrent(b, "sb", "B", rounds=10)
        return b

    def _train_pathways_only(b):
        """Phases 1 and 2 with NO co-firing -- the control."""
        _fix(b, "A", "B")
        try:
            for src in ("A", "B"):
                for i in range(10):
                    dsts = {src: ["C"]}
                    if i > 0:
                        dsts["C"] = ["C"]
                    b.project({}, dsts)
        finally:
            _unfix(b, "A", "B")

    control, full = [], []
    for seed in SEEDS_ASSOC:
        b = _setup(seed)
        _train_pathways_only(b)
        control.append(overlap(_recue(b, "A"), _recue(b, "B")))

        b = _setup(seed)
        associate(b, "A", "B", "C", rounds=10)
        full.append(overlap(_recue(b, "A"), _recue(b, "B")))

    mc, _ = _mean_sd(control)
    mf, sdf = _mean_sd(full)
    ch = CHANCE

    assert mf >= 0.08, (
        f"association overlap {mf:.4f} +/- {sdf:.4f} does not clear the paper's "
        f"8% of assembly size (chance {ch:.4f}); per-seed {full}. [PNAS20]"
    )
    assert mf > mc, (
        f"co-firing did not raise overlap above training the pathways alone: "
        f"control {mc:.4f} -> full {mf:.4f}. Phase 3 is the step that "
        f"associates, so this failing means it is not doing anything. [PNAS20]"
    )
    assert mf > ch * 5, (
        f"association overlap {mf:.4f} is not meaningfully above chance {ch:.4f}"
    )


# ---------------------------------------------------------------------------
# 6. Overlap is preserved under projection (RP&C)
# ---------------------------------------------------------------------------

def test_projection_preserves_overlap():
    """[PNAS20] Fig. 2D: "assembly overlap is indeed conserved reasonably well
    under projection".

    This is what lets a similarity signal survive being routed through another
    area -- if it failed, no multi-area architecture could carry affinity, and
    every downstream binding claim in this repo would be unsupportable.
    """
    ins, outs = [], []
    for seed in SEEDS:
        b = _brain(seed)
        b.add_area("A", N, K, beta=BETA)
        b.add_area("B", N, K, beta=BETA)
        b.add_stimulus("sa", K)
        b.add_stimulus("sb", K)
        x = _project_recurrent(b, "sa", "A", rounds=10)
        y = _project_recurrent(b, "sb", "A", rounds=10)
        ins.append(overlap(x, y))

        saved = b.disable_plasticity
        b.disable_plasticity = True
        try:
            b.areas["A"]._winners = np.asarray(x.winners, dtype=np.uint32)
            b._engine.set_winners("A", b.areas["A"]._winners)
            b.project({}, {"A": ["B"]})
            px = _snap(b, "B")
            b.areas["A"]._winners = np.asarray(y.winners, dtype=np.uint32)
            b._engine.set_winners("A", b.areas["A"]._winners)
            b.project({}, {"A": ["B"]})
            py = _snap(b, "B")
        finally:
            b.disable_plasticity = saved
        outs.append(overlap(px, py))
    mi, _ = _mean_sd(ins)
    mo, sdo = _mean_sd(outs)
    assert abs(mo - mi) <= 0.25, (
        f"overlap not preserved under RP&C: {mi:.3f} in -> {mo:.3f} +/- "
        f"{sdo:.3f} out. [PNAS20 Fig 2D]"
    )


# ---------------------------------------------------------------------------
# 7. Merge produces two-way connectivity
# ---------------------------------------------------------------------------

def test_merge_creates_two_way_connectivity_with_bounded_support():
    """[PNAS20] merge(x, y, A, z) yields z with "strong two-way synaptic
    connectivity between x and z, as well as between y and z".

    MEASURED ON THE WEIGHTS, and on support growth -- deliberately NOT by a
    recall probe. An earlier version of this test fired one parent, projected a
    single step into the target with plasticity off, and demanded the result
    overlap z above chance. It reported 0.000 and looked like a hard failure,
    but it was the wrong observable: k-cap selects the top k of n, and a
    3x-potentiated fiber from ONE parent need not win that competition in one
    step without the rest of the circuit driving. Merge was fine.

    The two things actually claimed:

    * ``.reference/dmitropolsky-assemblies/simulations_test.py::test_merge``
      asserts BOUNDED SUPPORT -- w_a, w_b <= 10k and w_c <= 20k, where
      ``saved_w`` is the cumulative count of neurons that have ever fired. That
      is a convergence criterion: the assembly stops recruiting.
    * [PNAS20]'s prose claims strong two-way connectivity, which is a statement
      about SYNAPTIC WEIGHT between the assemblies, not about one-step recall.

    Measured at n=10000, k=100, p=0.01, beta=0.05, 50 rounds: support
    1619-1713 (bound 2000), potentiation 3.19x / 2.47x / 3.31x / 2.53x on
    B->A / A->B / C->A / A->C.

    NOTE the merge protocol here drives the parents with STIMULI rather than
    ``_fix``. The reference keeps ``{"stimA":["A"],"stimB":["B"]}`` firing every
    round; pinning the parents instead is actively wrong in this engine,
    because a projection into a FIXED area is short-circuited before plasticity
    is applied (see ``_fix``'s own docstring), which silently kills the
    C->A / C->B back-projection that defines merge. Fixed sources measure
    0.000 potentiation where stimulus-driven sources measure 3.19x.
    """
    from neural_assemblies.assembly_calculus.ops import merge

    # Reference regime: k ~ sqrt(n), p=0.01, beta=0.05, 50 rounds.
    n, k, p, beta, rounds = 10_000, 100, 0.01, 0.05, 50
    supports, ratios = [], []
    for seed in SEEDS[:3]:
        b = Brain(p=p, seed=seed, norm_init=NORM_INIT)
        for area in ("B", "C", "A"):
            b.add_area(area, n, k, beta=beta)
        b.add_stimulus("sb", k)
        b.add_stimulus("sc", k)

        def _form(stim, area):
            b.project({stim: [area]}, {})
            for _ in range(9):
                b.project({stim: [area]}, {area: [area]})
            return _snap(b, area)

        x = _form("sb", "B")
        y = _form("sc", "C")
        # stimulus-driven, NOT _fix -- see docstring
        z = merge(b, "B", "C", "A", stim_a="sb", stim_b="sc", rounds=rounds)

        eng = b._engine
        supports.append(max(eng._areas[a].w for a in ("A", "B", "C")))

        def _ratio(src, dst, src_asm, dst_asm):
            conn = eng._area_conns.get(src, {}).get(dst)
            if conn is None:
                return None
            W = np.asarray(conn.weights)
            if W.ndim != 2 or W.size == 0:
                return None
            s2c = {int(v): i for i, v in enumerate(eng.get_neuron_id_mapping(src))}
            d2c = {int(v): i for i, v in enumerate(eng.get_neuron_id_mapping(dst))}
            r = [s2c[int(v)] for v in src_asm.winners
                 if int(v) in s2c and s2c[int(v)] < W.shape[0]]
            c = [d2c[int(v)] for v in dst_asm.winners
                 if int(v) in d2c and d2c[int(v)] < W.shape[1]]
            if len(r) < 2 or len(c) < 2:
                return None
            sub = W[np.ix_(r, c)]
            base = W[W > 0].mean() if (W > 0).any() else 1.0
            return float(sub[sub > 0].mean() / base) if (sub > 0).any() else 0.0

        for a, bb, sa, da in (("B", "A", x, z), ("A", "B", z, x),
                              ("C", "A", y, z), ("A", "C", z, y)):
            v = _ratio(a, bb, sa, da)
            if v is not None:
                ratios.append(v)

    max_support = float(np.mean(supports))
    assert max_support <= 20 * k, (
        f"merge did not converge: support {max_support:.0f} exceeds the "
        f"reference bound of 20k = {20 * k}. "
        f"[dmitropolsky-assemblies simulations_test.test_merge]"
    )
    mr, sdr = _mean_sd(ratios)
    assert mr >= 1.5, (
        f"merge produced no two-way potentiation between the merged assembly "
        f"and its parents: mean weight ratio {mr:.2f} +/- {sdr:.2f}, expected "
        f"clearly above 1.0. [PNAS20]"
    )

"""
Parity tests: torch_sparse vs numpy_sparse produce equivalent dynamics.

Both engines should produce the same qualitative behavior (stable assemblies,
high overlap, recovery, separation) even though RNG sequences differ due to
different initialization (hash-based vs random permutation).

We verify qualitative parity: both engines meet the same acceptance thresholds
for each assembly calculus operation, not bit-identical results.
"""

import copy
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.engine import create_engine
from neural_assemblies.core.semantics import (
    ArithmeticMode, CandidateDomain, ConnectomeMode, TieBreakRule,
)
from neural_assemblies.assembly_calculus import (
    chance_overlap,
    project,
    reciprocal_project,
    associate,
    merge,
    pattern_complete,
    separate,
)
from neural_assemblies.assembly_calculus.ops import _snap


# ---------------------------------------------------------------------------
# Skip if torch+CUDA not available
# ---------------------------------------------------------------------------

def _has_torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

pytestmark = pytest.mark.skipif(
    not _has_torch_cuda(),
    reason="torch_sparse requires PyTorch + CUDA")


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N = 10000
K = 100
P = 0.05
BETA = 0.1
ROUNDS = 10
SEED = 42

ENGINES = ["numpy_sparse", "torch_sparse"]


def _make_brain(engine, **kwargs):
    defaults = dict(
        p=P,
        save_winners=True,
        seed=SEED,
        sampled_recurrence_policy="acknowledged",
    )
    defaults.update(kwargs)
    return Brain(engine=engine, **defaults)


def test_torch_profile_exposes_sampled_drive_and_backend_ties():
    semantics = _make_brain("torch_sparse").model_semantics
    assert semantics.connectome is ConnectomeMode.LAZY_CONTENT_ADDRESSED
    assert semantics.candidate_domain is (
        CandidateDomain.MATERIALIZED_PLUS_ORDER_STATISTICS
    )
    assert semantics.default_tie_break is TieBreakRule.BACKEND_TOPK_ORDER
    assert semantics.arithmetic is ArithmeticMode.FLOAT32


def test_torch_dense_drive_is_a_distinct_candidate_domain():
    engine = create_engine(
        "torch_sparse", p=P, seed=SEED, w_max=20.0,
        dense_drive=True, norm_init=True,
    )
    semantics = _make_brain(engine).model_semantics
    assert semantics.candidate_domain is (
        CandidateDomain.ALL_NEURONS_WITH_SAMPLED_DRIVE
    )


# ---------------------------------------------------------------------------
# Projection parity
# ---------------------------------------------------------------------------

class TestProjectParity:
    @pytest.mark.parametrize("engine", ENGINES)
    def test_project_stabilizes(self, engine):
        """Assembly stabilizes after projection rounds on both engines."""
        b = _make_brain(engine)
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)

        asm = project(b, "stim", "A", rounds=ROUNDS)

        # More recurrent rounds shouldn't change the assembly
        for _ in range(5):
            b.project({}, {"A": ["A"]})
        asm_later = _snap(b, "A")

        stability = asm.overlap(asm_later)
        assert stability > 0.9, f"{engine}: stability {stability:.3f} < 0.9"

    @pytest.mark.parametrize("engine", ENGINES)
    def test_project_correct_size(self, engine):
        b = _make_brain(engine)
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)

        asm = project(b, "stim", "A", rounds=ROUNDS)
        assert len(asm) == K, f"{engine}: len={len(asm)} != {K}"


# ---------------------------------------------------------------------------
# Reciprocal projection parity
# ---------------------------------------------------------------------------

class TestReciprocalParity:
    """Reciprocal projection recovers the source -- and the engines AGREE.

    RESOLVED (#96, #98, 2026-08-25). The long-standing 0.20-vs-0.65
    "disagreement" was an artifact of the READOUT: the old protocol measured
    recovery while plasticity was ON, so the readout modified what it read
    ([[probe-isolation-required]]) -- and the two engines punish that
    violation in OPPOSITE directions. Round-by-round at seed 42:

        numpy  B->A then +A->A, plastic : 0.59 0.13 0.29 0.23 0.26 ...
        numpy  same, under read_only    : 0.78 0.78 0.78 0.78 0.78 ...
        torch  B->A then +A->A, plastic : 0.39 0.36 0.68 0.68 0.69 ...
        torch  same, under read_only    : 0.67 0.67 0.67 0.67 0.67 ...

    Plasticity during recovery DESTROYS the recovered assembly on numpy and
    completes it on torch. Isolated, the engines overlap:

        numpy_sparse 0.7760 +/- 0.0368 (n=5, 0.73..0.81)
        torch_sparse 0.7560 +/- 0.0678 (n=5, 0.67..0.82)

    and the level matches the 0.75 the reference implementation restores
    (quoted in `reciprocal_project`'s docstring). So the IDIOM works.

    THE PERTURBATION CHANNEL IS RECRUITMENT, not Hebbian updates. Measured
    with plasticity ON but recruitment OFF (`_no_recruitment`), recovery is
    IDENTICAL to frozen -- same mean, same per-seed range, both engines:

        numpy  frozen 0.776 = norecruit 0.776  >>  plastic 0.232
        torch  frozen 0.756 = norecruit 0.756  >   plastic 0.650

    so weight growth along the recovered trajectory never flips a winner; what
    diverges the engines is how aggressively CANDIDATES outbid the recovered
    incumbents during the mixed rounds -- the same candidate-pricing surface
    documented in research/notes/substrate/FINDING_torch_pricing_exposed.md. That is
    what the plastic-gap assertion below is pinning.
    """

    SEEDS = (42, 1, 2, 3, 4)

    @staticmethod
    def _train(engine, seed):
        b = _make_brain(engine, seed=seed)
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)
        b.add_area("B", N, K, BETA)
        original_a = project(b, "stim", "A", rounds=ROUNDS)
        b.areas["A"].fix_assembly()
        reciprocal_project(b, "A", "B", rounds=ROUNDS)
        b.areas["A"].unfix_assembly()
        return b, original_a

    @classmethod
    def _recover_frozen(cls, seed, engine):
        """Probe-isolated recovery: the readout cannot modify what it reads."""
        b, original_a = cls._train(engine, seed)
        with b.read_only():
            b.project({}, {"B": ["A"]})
            for _ in range(ROUNDS - 1):
                b.project({}, {"B": ["A"], "A": ["A"]})
            return original_a.overlap(_snap(b, "A"))

    @classmethod
    def _recover_plastic(cls, seed, engine):
        """The OLD readout, kept to pin the divergence it manufactures."""
        b, original_a = cls._train(engine, seed)
        b.project({}, {"B": ["A"]})
        for _ in range(ROUNDS - 1):
            b.project({}, {"B": ["A"], "A": ["A"]})
        return original_a.overlap(_snap(b, "A"))

    @pytest.mark.slow
    @pytest.mark.parametrize("engine", ENGINES)
    def test_isolated_recovery_restores_the_source(self, engine):
        """Both engines, one shared band, judged on the confidence bound."""
        from neural_assemblies.diagnostics import ensemble
        e = ensemble(lambda s: self._recover_frozen(s, engine), self.SEEDS,
                     label=f"{engine} isolated reciprocal recovery")
        assert e.beats(0.55), (
            f"{e} -- isolated recovery fell below the shared band; the "
            f"reference restores ~0.75 and both engines measured 0.67-0.82")
        assert e.mean < 0.95, (
            f"{e} -- near-perfect restoration is the signature of a dead "
            f"fiber or a frozen readout artifact "
            f"([[fake-perfect-probe-signatures]]), not of a working idiom")

    @pytest.mark.slow
    def test_engines_agree_under_isolation_and_diverge_without_it(self):
        """Pins BOTH facts: isolation closes the gap, plasticity opens it.

        If the isolated gap widens past 0.15, engine parity on this idiom has
        genuinely regressed. If the plastic gap CLOSES, the open question about
        plastic-readout perturbation has been resolved -- record how, and
        retire the second assertion with the fix that did it.
        """
        from neural_assemblies.diagnostics import ensemble
        frozen = {e: ensemble(lambda s, e=e: self._recover_frozen(s, e),
                              self.SEEDS, label=f"{e} frozen")
                  for e in ENGINES}
        if len(frozen) < 2:
            pytest.skip("needs both engines to compare")
        gap = (max(a.mean for a in frozen.values())
               - min(a.mean for a in frozen.values()))
        assert gap < 0.15, (
            "the engines have stopped agreeing under an isolated readout: "
            + "; ".join(str(a) for a in frozen.values()))

        plastic = {e: ensemble(lambda s, e=e: self._recover_plastic(s, e),
                               self.SEEDS, label=f"{e} plastic")
                   for e in ENGINES}
        pgap = (max(a.mean for a in plastic.values())
                - min(a.mean for a in plastic.values()))
        assert pgap > 0.15, (
            "plastic readout no longer diverges the engines: "
            + "; ".join(str(a) for a in plastic.values())
            + " -- if a fix did this, record it and retire this assertion.")


# ---------------------------------------------------------------------------
# Association parity
# ---------------------------------------------------------------------------

class TestAssociateParity:
    @pytest.mark.parametrize("engine", ENGINES)
    def test_associate_shared_response(self, engine):
        """Association creates shared representation on both engines."""
        b = _make_brain(engine)
        b.add_stimulus("stimA", K)
        b.add_stimulus("stimB", K)
        b.add_area("A", N, K, BETA)
        b.add_area("B", N, K, BETA)
        b.add_area("C", N, K, BETA)

        project(b, "stimA", "A", rounds=ROUNDS)
        project(b, "stimB", "B", rounds=ROUNDS)
        associate(b, "A", "B", "C", stim_a="stimA", stim_b="stimB",
                  rounds=ROUNDS)

        b_copy1 = copy.deepcopy(b)
        b_copy1.project({"stimA": ["A"]}, {"A": ["C"]})
        for _ in range(5):
            b_copy1.project({}, {"A": ["C"], "C": ["C"]})
        c1 = _snap(b_copy1, "C")

        b_copy2 = copy.deepcopy(b)
        b_copy2.project({"stimB": ["B"]}, {"B": ["C"]})
        for _ in range(5):
            b_copy2.project({}, {"B": ["C"], "C": ["C"]})
        c2 = _snap(b_copy2, "C")

        measured = c1.overlap(c2)
        chance = chance_overlap(K, N)
        assert measured > chance * 3, (
            f"{engine}: association overlap {measured:.3f} <= {chance * 3:.3f}")


# ---------------------------------------------------------------------------
# Merge parity
# ---------------------------------------------------------------------------

class TestMergeParity:
    @pytest.mark.parametrize("engine", ENGINES)
    def test_merge_responds_to_either(self, engine):
        """Merge responds to either source on both engines."""
        b = _make_brain(engine)
        b.add_stimulus("stimA", K)
        b.add_stimulus("stimB", K)
        b.add_area("A", N, K, BETA)
        b.add_area("B", N, K, BETA)
        b.add_area("C", N, K, BETA)

        project(b, "stimA", "A", rounds=ROUNDS)
        project(b, "stimB", "B", rounds=ROUNDS)
        merge(b, "A", "B", "C", stim_a="stimA", stim_b="stimB",
              rounds=ROUNDS)

        b_copy1 = copy.deepcopy(b)
        b_copy1.areas["A"].fix_assembly()
        b_copy1.project({}, {"A": ["C"]})
        for _ in range(5):
            b_copy1.project({}, {"A": ["C"], "C": ["C"]})
        c1 = _snap(b_copy1, "C")

        b_copy2 = copy.deepcopy(b)
        b_copy2.areas["B"].fix_assembly()
        b_copy2.project({}, {"B": ["C"]})
        for _ in range(5):
            b_copy2.project({}, {"B": ["C"], "C": ["C"]})
        c2 = _snap(b_copy2, "C")

        measured = c1.overlap(c2)
        chance = chance_overlap(K, N)
        assert measured > chance * 2, (
            f"{engine}: merge overlap {measured:.3f} <= {chance * 2:.3f}")


# ---------------------------------------------------------------------------
# Pattern completion parity
# ---------------------------------------------------------------------------

class TestPatternCompleteParity:
    @pytest.mark.parametrize("engine", ENGINES)
    def test_pattern_completion_recovers(self, engine):
        """Pattern completion recovers assembly on both engines."""
        b = _make_brain(engine)
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)

        project(b, "stim", "A", rounds=ROUNDS)

        recovered, recovery = pattern_complete(
            b, "A", fraction=0.5, rounds=5, seed=42,
            observation_mode="plastic",
        )
        assert recovery > 0.6, (
            f"{engine}: pattern recovery {recovery:.3f} < 0.6")


# ---------------------------------------------------------------------------
# Separation parity
# ---------------------------------------------------------------------------

class TestSeparateParity:
    @pytest.mark.parametrize("engine", ENGINES)
    def test_separate_creates_distinct(self, engine):
        """Two stimuli create distinct assemblies on both engines."""
        b = _make_brain(engine)
        b.add_stimulus("stimA", K)
        b.add_stimulus("stimB", K)
        b.add_area("A", N, K, BETA)

        asm_a, asm_b, measured = separate(
            b, "stimA", "stimB", "A", rounds=ROUNDS)

        assert measured < 0.5, (
            f"{engine}: separation overlap {measured:.3f} >= 0.5")
        assert len(asm_a) == K
        assert len(asm_b) == K


# ---------------------------------------------------------------------------
# CSR expansion after activity reset (pregrown topology preserved)
# ---------------------------------------------------------------------------

class TestCSRShrinkGuard:
    def test_context_reset_preserves_csr_extent(self):
        """CONTEXT w=0 reset must not shrink pregrown CSR below stored rows."""
        import torch

        b = _make_brain("torch_sparse")
        b.add_area("CORE", n=3000, k=30)
        b.add_area("CTX", n=3000, k=30)
        b.add_stimulus("w", 100)
        eng = b._engine
        csr = eng._area_conns["CTX"]["CTX"]
        pat = ({"w": ["CORE"]}, {"CORE": ["CTX"], "CTX": ["CTX"]})

        for _ in range(15):
            b.project(*pat)
            b.project_rounds(
                "CTX",
                {"w": ["CORE"]},
                {"CORE": ["CTX"], "CTX": ["CTX"]},
                rounds=2,
            )

        pregrown = csr._nrows
        assert pregrown > 30

        eng._areas["CTX"].w = 0
        eng._areas["CTX"].winners = torch.zeros(
            0, dtype=torch.int32, device=eng._device)
        b.areas["CTX"].w = 0

        for _ in range(5):
            b.project(*pat)
            b.project_rounds(
                "CTX",
                {"w": ["CORE"]},
                {"CORE": ["CTX"], "CTX": ["CTX"]},
                rounds=2,
            )

        assert csr._nrows >= pregrown
        assert eng._areas["CTX"].w > 0


# ---------------------------------------------------------------------------
# norm_init parity (ported to torch_sparse)
# ---------------------------------------------------------------------------

class TestNormInitParity:
    """norm_init on torch_sparse: the flag is honored (previously swallowed by
    ``**kwargs``), the read-time 1/d_j scale engages, and both engines form
    stable assemblies with or without it."""

    def test_torch_norm_init_flag_is_wired(self):
        # Regression: TorchSparseEngine.__init__ used to accept **kwargs and
        # silently drop norm_init, so Brain(norm_init=True, engine="torch_sparse")
        # ran WITHOUT normalization.
        for ni in (True, False):
            b = _make_brain("torch_sparse", norm_init=ni)
            assert b._engine.norm_init is ni

    @pytest.mark.parametrize("engine", ENGINES)
    @pytest.mark.parametrize("norm_init", [True, False])
    def test_assembly_forms_and_stabilizes(self, engine, norm_init):
        b = _make_brain(engine, norm_init=norm_init)
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)
        asm = project(b, "stim", "A", rounds=ROUNDS)
        assert len(asm) == K, f"{engine} norm_init={norm_init}: size {len(asm)}"
        # Recurrence-only stability (matches TestProjectParity); adding fresh
        # stim drive churns winners and is not what "stable" means here.
        a1 = _snap(b, "A")
        b.project({}, {"A": ["A"]})
        a2 = _snap(b, "A")
        assert a1.overlap(a2) > 0.9, (
            f"{engine} norm_init={norm_init}: stability {a1.overlap(a2):.3f}")

    def test_norm_init_scale_path_engages_on_torch(self):
        # Direct evidence the normalization ran: the stimulus fiber's per-column
        # in-degree snapshot is populated only when norm_init drives _norm_scale.
        b_on = _make_brain("torch_sparse", norm_init=True)
        b_on.add_stimulus("stim", K)
        b_on.add_area("A", N, K, BETA)
        project(b_on, "stim", "A", rounds=ROUNDS)
        conn_on = b_on._engine._stim_conns["stim"]["A"]
        assert getattr(conn_on, "_norm_deg_base", None) is not None

        b_off = _make_brain("torch_sparse", norm_init=False)
        b_off.add_stimulus("stim", K)
        b_off.add_area("A", N, K, BETA)
        project(b_off, "stim", "A", rounds=ROUNDS)
        conn_off = b_off._engine._stim_conns["stim"]["A"]
        assert getattr(conn_off, "_norm_deg_base", None) is None

    def test_recurrent_bridge_parity_both_engines(self):
        # A -> B bridge exercises the area-fiber (CSR) norm_scale path, which
        # must produce a coherent bridged assembly on both engines.
        for engine in ENGINES:
            b = _make_brain(engine, norm_init=True)
            b.add_stimulus("stim", K)
            b.add_area("A", N, K, BETA)
            b.add_area("B", N, K, BETA)
            project(b, "stim", "A", rounds=ROUNDS)
            b.areas["A"].fix_assembly()
            project(b, "stim", "B", rounds=1)
            for _ in range(ROUNDS):
                b.project({}, {"A": ["B"], "B": ["B"]})
            b.areas["A"].unfix_assembly()
            assert len(_snap(b, "B")) == K, f"{engine}: B size wrong"


# ---------------------------------------------------------------------------
# Dense-drive mode (Lever A of docs/gpu_scale_design.md)
# ---------------------------------------------------------------------------

def _torch_engine(**kw):
    from neural_assemblies.core.torch_engine._engine import TorchSparseEngine
    defaults = dict(p=P, seed=SEED)
    defaults.update(kw)
    return TorchSparseEngine(**defaults)


class TestDenseDriveParity:
    """dense_drive scores all n candidates (exact topk) instead of sampling k
    order statistics; it must form stable, separable, recoverable assemblies
    just like the sparse path."""

    def test_dense_flag_wired(self):
        assert _torch_engine(dense_drive=True).dense_drive is True
        assert _torch_engine(dense_drive=False).dense_drive is False

    @pytest.mark.parametrize("norm_init", [True, False])
    def test_dense_assembly_forms_and_stabilizes(self, norm_init):
        eng = _torch_engine(norm_init=norm_init, dense_drive=True)
        b = Brain(engine=eng, p=eng.p, seed=eng.seed, w_max=eng.w_max,
                  norm_init=eng.norm_init, save_winners=True)
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)
        asm = project(b, "stim", "A", rounds=ROUNDS)
        assert len(asm) == K
        a1 = _snap(b, "A")
        b.project({}, {"A": ["A"]})
        a2 = _snap(b, "A")
        assert a1.overlap(a2) > 0.9, f"dense stability {a1.overlap(a2):.3f}"

    def test_dense_separates_and_recovers(self):
        eng = _torch_engine(norm_init=True, dense_drive=True)
        b = Brain(engine=eng, p=eng.p, seed=eng.seed, w_max=eng.w_max,
                  norm_init=eng.norm_init, save_winners=True)
        b.add_stimulus("s1", K)
        b.add_stimulus("s2", K)
        b.add_area("A", N, K, 0.1)
        a1 = project(b, "s1", "A", rounds=ROUNDS)
        b.inhibit_areas(["A"])
        a2 = project(b, "s2", "A", rounds=ROUNDS)
        b.inhibit_areas(["A"])
        assert a1.overlap(a2) < 0.5, f"dense separation {a1.overlap(a2):.3f}"
        a1b = project(b, "s1", "A", rounds=ROUNDS)
        assert a1.overlap(a1b) > 0.6, f"dense recovery {a1.overlap(a1b):.3f}"

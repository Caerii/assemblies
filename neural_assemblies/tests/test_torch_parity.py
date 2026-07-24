"""
Parity tests: torch_sparse vs numpy_sparse produce equivalent dynamics.

Both engines should produce the same qualitative behavior (stable assemblies,
high overlap, recovery, separation) even though RNG sequences differ due to
different initialization (hash-based vs random permutation).

We verify qualitative parity: both engines meet the same acceptance thresholds
for each assembly calculus operation, not bit-identical results.
"""

import copy
import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.assembly_calculus import (
    Assembly,
    overlap,
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
    defaults = dict(p=P, save_winners=True, seed=SEED)
    defaults.update(kwargs)
    return Brain(engine=engine, **defaults)


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
    @pytest.mark.parametrize("engine", ENGINES)
    def test_reciprocal_recovers(self, engine):
        """Reciprocal projection recovers source assembly on both engines."""
        b = _make_brain(engine)
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)
        b.add_area("B", N, K, BETA)

        original_a = project(b, "stim", "A", rounds=ROUNDS)

        b.areas["A"].fix_assembly()
        reciprocal_project(b, "A", "B", rounds=ROUNDS)

        b.areas["A"].unfix_assembly()
        b.project({}, {"B": ["A"]})
        for _ in range(ROUNDS - 1):
            b.project({}, {"B": ["A"], "A": ["A"]})

        recovered_a = _snap(b, "A")
        recovery = original_a.overlap(recovered_a)
        assert recovery > 0.6, f"{engine}: recovery {recovery:.3f} < 0.6"


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
            b, "A", fraction=0.5, rounds=5, seed=42)
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
            asm_b = project(b, "stim", "B", rounds=1)
            for _ in range(ROUNDS):
                b.project({}, {"A": ["B"], "B": ["B"]})
            b.areas["A"].unfix_assembly()
            assert len(_snap(b, "B")) == K, f"{engine}: B size wrong"

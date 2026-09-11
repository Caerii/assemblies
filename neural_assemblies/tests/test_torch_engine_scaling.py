"""torch_sparse: the homeostatic-scaling workload surface.

Every test here pins a defect found while making the S5 word-problem organ
run on the GPU engine -- each was a SILENT gap, not an error: the engine ran
and returned plausible numbers with the feature quietly absent.

  * add_connectivity was `pass`: an organ_p=0.5 organ built p=0.05 fibers.
  * synaptic_scaling was swallowed by **kwargs (as norm_init once was).
  * projection into a FIXED area discarded inputs -- the numpy engine's old
    footgun, reproduced here as a completely EMPTY arc->state fiber after a
    full FSM training run.
  * brain.read_only()/probe() gates recruitment via hasattr(engine,
    "_no_recruitment") and SILENTLY SKIPS engines without the attribute:
    probes recruited, and a trained Z60 machine read back at chance.
  * bfloat16 weight storage: margins are a few units of drive out of ~60,
    and 24 rounds of multiply-then-renormalize randomized them (acc 0.040
    bf16 vs 0.320 f32 vs 0.560 numpy-f32, same seed). Dense fibers are
    float32; the CSR store keeps bf16 for the low-density ambient fibers.

All CUDA-gated: they skip cleanly on CPU-only environments.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="torch CUDA required")

from neural_assemblies.core.brain import Brain  # noqa: E402


def _brain(**kw):
    kw.setdefault("p", 0.05)
    kw.setdefault("seed", 0)
    kw.setdefault("engine", "torch_sparse")
    kw.setdefault("norm_init", False)
    return Brain(**kw)


def test_add_connectivity_is_not_a_silent_noop():
    b = _brain()
    b.add_stimulus("s", 10)
    b.add_area("A", 500, 10, beta=0.1)
    b.add_area("B", 500, 10, beta=0.1)
    b.add_connectivity("A", "B", 0.4)
    eng = b._engine
    assert eng._p_for("A", "B") == 0.4
    assert eng._p_for("B", "A") == eng.p  # untouched fiber keeps global p
    # Structural precedence: after traffic the fiber's density is frozen.
    b.project({"s": ["A"]}, {})
    b.project({"s": ["A"]}, {"A": ["B"]})
    with pytest.raises(RuntimeError, match="carried traffic"):
        b.add_connectivity("A", "B", 0.2)


def test_dense_representation_selected_by_density():
    from neural_assemblies.core.torch_engine._csr import (
        CSRConn, TorchDenseConn,
    )
    b = _brain()
    b.add_area("A", 500, 10, beta=0.1)
    b.add_area("B", 500, 10, beta=0.1)
    b.add_connectivity("A", "B", 0.4)     # >= DENSE_MIN_P -> dense
    b.add_connectivity("B", "A", 0.1)     # < threshold -> stays CSR
    eng = b._engine
    assert isinstance(eng._area_conns["A"]["B"], TorchDenseConn)
    assert isinstance(eng._area_conns["B"]["A"], CSRConn)
    # Dense fibers store float32 (see module docstring: bf16 randomized
    # trained margins); the CSR store keeps bf16.
    assert TorchDenseConn.DTYPE == torch.float32


def test_dense_fiber_layout_follows_shape():
    """Perf guard, the GPU half of the numpy layout rule.

    `index_select(1, cols)` on a row-major (rows, cols) fiber reads each
    column with a rows-sized stride -- uncoalesced, one memory transaction
    per element. Measured on the organ's 20000x4200 fiber: 1195 us row-major
    vs 537 us column-major for the scaling op. Layout preserves logical
    [i, j], so it only moves time. Behavioural, like its numpy sibling.
    """
    from neural_assemblies.core.torch_engine._csr import TorchDenseConn
    b = _brain()
    b.add_area("A", 2000, 50, beta=0.1)   # tall fiber A->B (2000 > 200)
    b.add_area("B", 200, 10, beta=0.1)
    b.add_connectivity("A", "B", 0.4)
    b.add_connectivity("B", "A", 0.4)     # wide fiber B->A
    eng = b._engine
    tall = eng._area_conns["A"]["B"]
    wide = eng._area_conns["B"]["A"]
    assert isinstance(tall, TorchDenseConn) and tall._col_major, "tall fiber"
    assert isinstance(wide, TorchDenseConn) and not wide._col_major, "wide"
    # And the allocated buffer really carries that order: column-major means
    # stepping one ROW is one element (stride[0] == 1), i.e. a column is
    # contiguous.
    empty_i = torch.empty(0, dtype=torch.int32, device="cuda")
    empty_v = torch.empty(0, dtype=TorchDenseConn.DTYPE, device="cuda")
    tall.expand(64, 32, empty_i, empty_i.clone(), empty_v)
    wide.expand(32, 64, empty_i, empty_i.clone(), empty_v)
    assert tall._w.stride()[0] == 1, tall._w.stride()
    assert wide._w.stride()[1] == 1, wide._w.stride()


def test_scaling_setpoint_uses_fiber_p():
    """Aggregate signature, mirroring test_scoped_synaptic_scaling's numpy
    version: touched columns sit at the FIBER's scale (rows * 0.4), nowhere
    near the brain-p scale (rows * 0.05)."""
    b = _brain(synaptic_scaling=True, seed=0)
    b.add_stimulus("s", 10)
    b.add_area("A", 500, 10, beta=0.2)
    b.add_area("B", 500, 10, beta=0.2)
    b.add_connectivity("A", "B", 0.4)
    b.project({"s": ["A"]}, {})
    for _ in range(6):
        b.project({"s": ["A"]}, {"A": ["B"]})
    eng = b._engine
    conn = eng._area_conns["A"]["B"]
    w = conn._w.float()
    rows = min(int(eng._areas["A"].w), w.shape[0])
    touched = eng._areas["B"].winners.long()
    touched = touched[touched < w.shape[1]]
    sums = w[:rows, touched].sum(dim=0)
    fiber_setpoint = rows * 0.4
    brain_setpoint = rows * 0.05
    assert float(sums.min()) > 2.0 * brain_setpoint
    assert float(sums.median()) > 0.5 * fiber_setpoint


def test_deferred_scaling_refuses_loudly():
    with pytest.raises(
        ValueError, match="does not support synaptic_scaling_deferred"
    ):
        _brain(synaptic_scaling=True, synaptic_scaling_deferred=True)


def test_fixed_target_projection_learns():
    """Projection into a FIXED area must potentiate the afferents (and
    materialize the fiber), not silently discard the inputs."""
    b = _brain(seed=3)
    b.add_stimulus("s", 20)
    b.add_area("A", 400, 10, beta=0.1)
    b.add_area("B", 400, 10, beta=0.1)
    b.project({"s": ["A"]}, {})
    b.project({"s": ["A"]}, {"A": ["B"]})   # give B an assembly
    b.areas["B"].fix_assembly()
    eng = b._engine
    conn = eng._area_conns["A"]["B"]
    before = float(conn.accumulate_rows(
        eng._areas["A"].winners.long(), int(eng._areas["B"].w)).sum())
    for _ in range(3):
        b.project({"s": ["A"]}, {"A": ["B"]})
    after = float(conn.accumulate_rows(
        eng._areas["A"].winners.long(), int(eng._areas["B"].w)).sum())
    b.areas["B"].unfix_assembly()
    assert after > before * 1.05, (before, after)


def test_probe_does_not_recruit():
    """brain.read_only()/probe() must stop recruitment on this engine too.
    Without `_no_recruitment` the hasattr gate silently skipped torch and
    probes grew the area ([[probe-isolation-required]])."""
    b = _brain(seed=1)
    b.add_stimulus("s", 20)
    b.add_area("A", 1000, 20, beta=0.1)
    b.add_area("B", 1000, 20, beta=0.1)
    b.project({"s": ["A"]}, {})
    for _ in range(3):
        b.project({"s": ["A"]}, {"A": ["B"]})
    eng = b._engine
    w_before = int(eng._areas["B"].w)
    with b.read_only():
        for _ in range(5):
            b.project({"s": ["A"]}, {"A": ["B"]})
        assert int(eng._areas["B"].w) == w_before  # no growth inside
    assert int(eng._areas["B"].w) == w_before      # nor after restore


def test_materialize_area_full_extent():
    b = _brain(seed=2)
    b.add_stimulus("s", 10)
    b.add_area("A", 300, 10, beta=0.1)
    added = b.materialize_area("A")
    eng = b._engine
    assert added == 300
    assert int(eng._areas["A"].w) == 300
    assert len(eng._areas["A"].compact_to_neuron_id) == 300
    assert b.materialize_area("A") == 0   # idempotent

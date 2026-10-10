"""Packed 4-bit organ counts (DenseOrganFiber(count_dtype="int4"), kernels/06a_organ_packed_kernels.cu)
equal int8 counts wherever the weight clip binds by count 15: the same drives (both kernels, with
and without a brain map), the same stores and replays, the same count edits on the same counts --
and, under unlearning, exactly the int8 model with its counts capped at 15, which is what the
packed layout is (the variant a sleeping study must declare)."""
from __future__ import annotations

import pytest

from neural_assemblies.tests import _devices

torch = _devices.import_torch()
pytestmark = pytest.mark.requires_cuda

DEV = "cuda"
SEEDS = [900, 901, 902]


@pytest.fixture(scope="module")
def mod():
    return _devices.fused_kernels()


def _fibers(n, beta=0.3):
    from neural_assemblies.core.torch_engine._hashed import DenseOrganFiber
    kw = dict(beta=beta, w_max=20.0, norm_init=True, max_rounds=64, device=DEV)
    return (DenseOrganFiber(SEEDS, n, n, 0.5, **kw),
            DenseOrganFiber(SEEDS, n, n, 0.5, count_dtype="int4", **kw))


def _write(fibers, n, k, rounds, pool, gen):
    """random transitions among a small pool of neurons, so counts pass 15; -1 rows in places"""
    for _ in range(rounds):
        prev = torch.stack([pool[torch.randperm(pool.numel(), generator=gen, device=DEV)[:k]] for _ in SEEDS])
        new = torch.stack([pool[torch.randperm(pool.numel(), generator=gen, device=DEV)[:k]] for _ in SEEDS])
        prev[0, :3] = -1
        new[2, -2:] = -1
        for f in fibers:
            f.observe(prev, new)
    for f in fibers:
        f.check()


@pytest.mark.parametrize("n", [2000, 2002])          # the four-column kernel and the scalar one
def test_packed_drive_and_write_equal_int8(mod, n):
    f8, f4 = _fibers(n)
    gen = torch.Generator(device=DEV).manual_seed(1)
    pool = torch.randperm(n, generator=gen, device=DEV)[:120]
    _write((f8, f4), n, 40, 300, pool, gen)
    assert int(f8.C.max()) > 15                          # int8 ran past what 4 bits hold
    assert torch.equal(f4.unpacked(), f8.C.clamp(max=15).to(torch.int16))
    rows = torch.stack([pool[torch.randperm(120, generator=gen, device=DEV)[:40]] for _ in SEEDS])
    a, b = torch.zeros(3, n, device=DEV), torch.zeros(3, n, device=DEV)
    f8.contribute(a, rows)
    f4.contribute(b, rows)
    assert torch.equal(a, b)
    brains = torch.tensor([2, 0, 0, 1, 2], dtype=torch.int32, device=DEV)
    vrows = rows[brains.long()]
    a, b = torch.zeros(5, n, device=DEV), torch.zeros(5, n, device=DEV)
    f8.contribute(a, vrows, brains)
    f4.contribute(b, vrows, brains)
    assert torch.equal(a, b)
    assert f4.C is None and f4.Cp.dtype == torch.uint8 and f4.Cp.numel() * 2 >= f8.C.numel()


def test_packed_count_edits_equal_int8_on_the_same_counts(mod):
    n = 2002
    f8, f4 = _fibers(n)
    gen = torch.Generator(device=DEV).manual_seed(2)
    pool = torch.randperm(n, generator=gen, device=DEV)[:150]
    _write((f8, f4), n, 50, 200, pool, gen)
    f8.clamp_counts(15)
    assert f8.count_sum() == f4.count_sum() and f8.nonzero() == f4.nonzero() == f4.nnz
    B, k = 3, 50
    bi = torch.arange(B, device=DEV).view(B, 1, 1)
    for step in range(20):
        i = torch.stack([pool[torch.randperm(150, generator=gen, device=DEV)[:k]] for _ in SEEDS]).view(B, k, 1)
        j = torch.stack([pool[torch.randperm(150, generator=gen, device=DEV)[:k]] for _ in SEEDS]).view(B, 1, k)
        g = torch.tensor([True, step % 2 == 0, False], device=DEV)
        assert int(f8.decrement(bi, i, j, g)) == int(f4.decrement(bi, i, j, g))
    assert torch.equal(f4.unpacked(), f8.unpacked())
    f8.clamp_counts(10)
    f4.clamp_counts(10)
    assert torch.equal(f4.unpacked(), f8.unpacked())
    g8, g4 = torch.Generator(device=DEV).manual_seed(7), torch.Generator(device=DEV).manual_seed(7)
    assert f8.downscale([0.3, 0.0, 0.7], g8, chunk=300) == f4.downscale([0.3, 0.0, 0.7], g4, chunk=300)
    assert torch.equal(f4.unpacked(), f8.unpacked())
    f8.select([0, 2])
    f4.select([0, 2])
    assert torch.equal(f4.unpacked(), f8.unpacked()) and f4.Cp.shape[0] == 2


def test_packed_counts_refuse_a_clip_past_15(mod):
    from neural_assemblies.core.torch_engine._hashed import DenseOrganFiber
    with pytest.raises(ValueError, match="past the int4 range"):
        DenseOrganFiber(SEEDS, 600, 600, 0.5, beta=0.05, w_max=20.0, device=DEV, count_dtype="int4")


def _store(U, count_dtype, n=2000, k=60, p=0.5, tau=17):
    from research.experiments import memory_sleep as sl
    from research.experiments import memory_reuse_grammar as rg
    from research.experiments import memory_threshold_law as tl
    from research.experiments import memory_fast as mf
    from research.experiments import memory_write_separation as ws
    spec = {"n": n, "k": k, "p": p, "tau": tau, "rho": rg.RHO, "beta": round(tl.theta(n, k, p), 5)}
    original = ws.build                      # memory_sleep.build_store builds through ws.build
    try:
        ws.build = lambda *a: mf.build_memory(*a, count_dtype=count_dtype)
        return sl.build_store(spec, U, SEEDS, DEV)
    finally:
        ws.build = original


def test_build_memory_is_the_registered_build(mod):
    from research.experiments import memory_fast as mf
    from research.experiments import memory_write_separation as ws
    a, b = ws.build(2000, 60, 0.5, 17, SEEDS, DEV), mf.build_memory(2000, 60, 0.5, 17, SEEDS, DEV)
    for name in ("B", "n", "k", "p", "beta", "w_max", "norm_init", "rounds", "strength"):
        assert getattr(a, name) == getattr(b, name), name
    assert a.area.bias_decay == b.area.bias_decay and a.fiber.count_dtype == b.fiber.count_dtype == "int8"


def test_a_packed_memory_stores_and_replays_as_int8(mod):
    from research.experiments import memory_fast as mf
    from research.experiments import memory_sleep as sl
    a, b = _store(40, None), _store(40, "int4")
    assert torch.equal(a["allst"], b["allst"])
    assert torch.equal(b["mem"].fiber.unpacked(), a["mem"].fiber.C.clamp(max=15).to(torch.int16))
    assert sl.reliability(a, DEV) == sl.reliability(b, DEV) == mf.reliability(b, DEV)


def test_packed_sleep_is_int8_sleep_with_counts_capped_at_15(mod):
    from research.experiments import memory_fast as mf
    from research.experiments import memory_sleep as sl
    a, b = _store(40, None), _store(40, "int4")
    a["mem"].fiber.clamp_counts(15)
    ga = torch.Generator(device=DEV).manual_seed(sl.NOISE_SEED)
    gb = torch.Generator(device=DEV).manual_seed(sl.NOISE_SEED)
    ta = mf.sleep(a["mem"], ga, 1.2, 30, DEV)
    tb = mf.sleep(b["mem"], gb, 1.2, 30, DEV)
    assert ta == tb and ta[0] > 0
    assert torch.equal(a["mem"].fiber.unpacked(), b["mem"].fiber.unpacked())

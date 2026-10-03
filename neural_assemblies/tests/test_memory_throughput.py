"""The memory's throughput paths equal its solo paths, bit for bit
(research/notes/substrate/DESIGN_memory_throughput.md).

A learning-rate sweep runs its rates as brains of ONE launch, a checkpoint
reads all its cues in one batched recall, and a sweep drops the rates whose
stores have finished. Each is only a speedup if every brain's trajectory is
the trajectory it has alone: the same stored assemblies, counts, bias and
recalls, and the same readings and stop points in the study loop.
"""
from __future__ import annotations

import pytest

from neural_assemblies.tests import _devices

torch = _devices.import_torch()
pytestmark = pytest.mark.requires_cuda

from neural_assemblies.core.numpy_engine import _seeding               # noqa: E402

N, K, P, T = 600, 30, 0.5, 8
BETAS = (0.05, 0.1, 0.2)
SEEDS = 3


@pytest.fixture(scope="module")
def mod():
    return _devices.fused_kernels()


def _i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def _brains():
    return [_i32(_seeding.fnv1a_pair_seed(42 + b, "A", "A")) for b in range(SEEDS)]


def _stim(a):
    return [_i32(_seeding.fnv1a_pair_seed(42 + b, f"s{a}", "A")) for b in range(SEEDS)]


def _memory(beta, seeds, strength):
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    return AssemblyMemory(seeds, N, K, P, beta=beta, w_max=20.0, norm_init=True,
                          rounds=T, strength=strength, max_items=64)


def _swept(strength, betas=BETAS):
    sd = _brains()
    return _memory([b for b in betas for _ in sd], sd * len(betas), strength)


def _part(g):
    return slice(g * SEEDS, (g + 1) * SEEDS)


@pytest.mark.parametrize("strength", [0.0, 0.5])
def test_a_swept_memory_equals_its_solo_runs(mod, strength):
    swept = _swept(strength)
    solo = [_memory(b, _brains(), strength) for b in BETAS]
    for a in range(20):
        won = swept.store(_stim(a) * len(BETAS))
        for g, m in enumerate(solo):
            assert torch.equal(won[_part(g)], m.store(_stim(a))), (a, g)
    for g, m in enumerate(solo):
        assert torch.equal(swept.fiber.C[_part(g)], m.fiber.C)
        if strength:
            assert torch.equal(swept.bias[_part(g)], m.bias)
        cue = m.area.ever.nonzero()[:K // 2, 1].view(1, -1).expand(SEEDS, -1)
        assert torch.equal(swept.recall(cue.repeat(len(BETAS), 1))[_part(g)], m.recall(cue))


@pytest.mark.parametrize("strength,masked,rounds", [(0.5, None, None), (0.5, False, 3),
                                                     (0.0, None, None)])
def test_recall_many_equals_recall(mod, strength, masked, rounds):
    swept = _swept(strength)
    St = torch.stack([swept.store(_stim(a) * len(BETAS)) for a in range(16)])
    picks = [0, 3, 7, 15]
    cues = St[picks][:, :, : K // 2].permute(1, 0, 2)
    many = swept.recall_many(cues, masked=masked, rounds=rounds)
    for s, i in enumerate(picks):
        alone = swept.recall(St[i][:, : K // 2], masked=masked, rounds=rounds)
        assert torch.equal(many[:, s], alone), i


def test_recall_many_in_several_passes_equals_one(mod, monkeypatch):
    from neural_assemblies.core.torch_engine import _memory
    swept = _swept(0.5)
    St = torch.stack([swept.store(_stim(a) * len(BETAS)) for a in range(8)])
    cues = St[:, :, : K // 2].permute(1, 0, 2)
    one = swept.recall_many(cues)
    monkeypatch.setattr(_memory, "RECALL_BYTES", 8 * N * swept.B * 3)   # 3 cues a pass
    assert torch.equal(swept.recall_many(cues), one)


def test_dropping_rates_leaves_the_rest_on_their_solo_runs(mod):
    swept = _swept(0.5)
    solo = [_memory(b, _brains(), 0.5) for b in BETAS]
    for a in range(10):
        swept.store(_stim(a) * len(BETAS))
        for m in solo:
            m.store(_stim(a))
    kept = (0, 2)
    swept.select([g * SEEDS + b for g in kept for b in range(SEEDS)])
    assert swept.beta == tuple(BETAS[g] for g in kept for _ in range(SEEDS))
    for a in range(10, 20):
        won = swept.store(_stim(a) * len(kept))
        for j, g in enumerate(kept):
            assert torch.equal(won[_part(j)], solo[g].store(_stim(a))), (a, g)
    for j, g in enumerate(kept):
        assert torch.equal(swept.fiber.C[_part(j)], solo[g].fiber.C)
        assert torch.equal(swept.bias[_part(j)], solo[g].bias)
        assert torch.equal(swept.area.ever[_part(j)], solo[g].area.ever)


def test_a_swept_study_reads_and_stops_as_its_solo_runs(mod):
    """The study loop: every rate's readings at every checkpoint, its first-
    item count and its stop point equal the one-rate launch's -- including
    rates that stop at different checkpoints and are dropped mid-launch."""
    from research.experiments import memory_learning_rate as lr
    betas = (0.03, 0.1, 0.3)
    seeds = list(range(SEEDS))
    profiles = {b: lr.profile(b) for b in betas}
    options = dict(stop_on=("complete_distinct",), grid_start=2, give_up=48, stop_from=8)
    swept = lr.run_betas(N, K, betas, seeds, 96, "cuda", profiles, **options)
    for b, (c, cache) in zip(betas, swept):
        c_alone, cache_alone = lr.run_beta(N, K, b, seeds, 96, "cuda", profiles[b], **options)
        assert c == c_alone and cache == cache_alone, b
    assert len({max(cache) for _, cache in swept}) > 1, "the rates should stop apart"


def test_a_sweep_too_large_for_one_launch_is_split(mod, monkeypatch):
    from research.experiments import memory_learning_rate as lr
    seeds = list(range(SEEDS))
    betas = (0.05, 0.1, 0.2)
    profiles = {b: lr.profile(b) for b in betas}
    options = dict(stop_on=("complete_distinct",), grid_start=2, give_up=16)
    whole = lr.run_betas(N, K, betas, seeds, 32, "cuda", profiles, **options)
    monkeypatch.setattr(lr, "launch_rates", lambda n, B, **_: 2)
    assert lr.run_betas(N, K, betas, seeds, 32, "cuda", profiles, **options) == whole


@pytest.mark.parametrize("strength", [0.0, 0.5])
def test_a_graphed_store_equals_the_eager_store(mod, strength):
    """The write replayed as a CUDA graph: the same items, counts, bias,
    ever-fired record and round count as the eager write, through a drop of
    rates (which re-captures at the new shape)."""
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    sd = _brains() * len(BETAS)
    betas = [b for b in BETAS for _ in range(SEEDS)]

    def build(graphs):
        return AssemblyMemory(sd, N, K, P, beta=betas, w_max=20.0, norm_init=True,
                              rounds=T, strength=strength, max_items=64, graphs=graphs)
    eager, graphed = build(False), build(True)
    for a in range(24):
        if a == 12:
            keep = [g * SEEDS + b for g in (0, 2) for b in range(SEEDS)]
            eager.select(keep)
            graphed.select(keep)
        ss = _stim(a) * (eager.B // SEEDS)
        assert torch.equal(graphed.store(ss), eager.store(ss)), a
    assert torch.equal(graphed.fiber.C, eager.fiber.C)
    assert torch.equal(graphed.area.ever, eager.area.ever)
    if strength:
        assert torch.equal(graphed.bias, eager.bias)
    assert (graphed.items, graphed.area.rounds_seen) == (eager.items, eager.area.rounds_seen)
    graphed.check()

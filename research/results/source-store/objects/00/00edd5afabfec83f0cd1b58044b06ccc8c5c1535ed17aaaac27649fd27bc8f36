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


def test_int16_counts_equal_int8_where_both_are_exact(mod):
    """The count width changes storage, not arithmetic: at a rate whose clip
    binds by count 127, int16 counts give the int8 run bit for bit."""
    from neural_assemblies.core.torch_engine._hashed import DenseOrganFiber
    runs = []
    for dtype in ("int8", "int16"):
        mem = _memory(0.1, _brains(), 0.5)
        mem.fiber = DenseOrganFiber(mem.seeds, N, N, P, beta=0.1, w_max=20.0, norm_init=True,
                                    max_rounds=256, device="cuda", count_dtype=dtype)
        assert mem.fiber.C.dtype == getattr(torch, dtype)
        won = [mem.store(_stim(a)) for a in range(16)]
        runs.append((won, mem.fiber.C.to(torch.int32), mem.bias.clone(),
                     mem.recall(won[3][:, : K // 2])))
    (w8, c8, b8, r8), (w16, c16, b16, r16) = runs
    assert all(torch.equal(a, b) for a, b in zip(w8, w16))
    assert torch.equal(c8, c16) and torch.equal(b8, b16) and torch.equal(r8, r16)


def test_a_weak_write_counts_past_127_on_a_table_that_reaches_its_clip(mod):
    """Below rate ~0.024 the clip binds past count 127: the fiber widens to
    int16 and its chain table reaches the clip, so no count is mispriced."""
    from neural_assemblies.core.torch_engine._hashed import clip_count
    # the Hebbian control: its hubs fire every round, so their counts pass 127
    mem = _memory(0.005, _brains(), 0.0)
    assert mem.fiber.count_dtype == "int16" and mem.fiber.MAX_COUNT == 32767
    assert mem.fiber.tab.shape[-1] - 1 >= clip_count(0.005, 20.0) == 601
    tab = mem.fiber.tab.cpu()
    assert tab[-1] == tab[-2] == torch.tensor(20.0)
    for a in range(8):
        mem.store(_stim(a))
    mem.check()
    # counts far past 127 (planted), priced by the kernel as the table says
    from research.experiments import memory_pattern_efficiency as pe
    fiber = mem.fiber
    gen = torch.Generator(device="cuda").manual_seed(7)
    fiber.C.copy_(torch.randint(0, 1000, fiber.C.shape, generator=gen, device="cuda",
                                dtype=torch.int16))
    rows = torch.randint(0, N, (SEEDS, K), generator=gen, device="cuda")
    drive = torch.zeros(SEEDS, N, device="cuda")
    fiber.contribute(drive, rows)
    ref = torch.zeros(SEEDS, N, device="cuda", dtype=torch.float64)
    table = fiber.tab.double()
    for b in range(SEEDS):
        present = pe.presence_of(fiber.pres, b, N)[rows[b]]            # [K, N]
        counts = fiber.C[b][rows[b]].long().clamp_max(table.shape[-1] - 1)
        ref[b] = (table[counts] * present).sum(0) * fiber.invdj[b].double()
    assert torch.allclose(drive.double(), ref, rtol=1e-5, atol=1e-6)


def test_settling_rounds_ride_along_without_changing_the_read(mod):
    """``recall_many(settle=True)`` returns the same winners as the plain read,
    and each cue's settling round (period <= 2) lies in [3, rounds + 1]."""
    swept = _swept(0.5)
    St = torch.stack([swept.store(_stim(a) * len(BETAS)) for a in range(16)])
    cues = St[:, :, : K // 2].permute(1, 0, 2)
    plain = swept.recall_many(cues, rounds=12)
    winners, settled = swept.recall_many(cues, rounds=12, settle=True)
    assert torch.equal(plain, winners)
    assert settled.shape == cues.shape[:2]
    assert int(settled.min()) >= 3 and int(settled.max()) <= 13


def test_the_burst_write_counts_exactly_the_bursting_pairs(mod):
    """After one item under the burst write, a synapse holds one count iff it
    is present and both its ends fired in at least `burst_min` rounds -- in
    both directions, the order of their firing ignored."""
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    from research.experiments import memory_pattern_efficiency as pe
    mem = AssemblyMemory(_brains(), N, K, P, beta=0.1, w_max=20.0, norm_init=True, rounds=T,
                         strength=0.5, max_items=64, write_rule="burst", burst_min=2)
    mem.store(_stim(0))
    burst = mem.last_burst
    assert bool(burst.any()) and not bool(burst.all())
    for b in range(SEEDS):
        present = pe.presence_of(mem.fiber.pres, b, N)
        expected = present & burst[b].view(-1, 1) & burst[b].view(1, -1)
        assert torch.equal(mem.fiber.C[b] > 0, expected)
        assert int(mem.fiber.C[b].max()) == 1


def test_the_round_write_is_untouched_by_the_burst_option(mod):
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    a = AssemblyMemory(_brains(), N, K, P, beta=0.1, w_max=20.0, norm_init=True, rounds=T,
                       strength=0.5, max_items=64)
    b = AssemblyMemory(_brains(), N, K, P, beta=0.1, w_max=20.0, norm_init=True, rounds=T,
                       strength=0.5, max_items=64, write_rule="round", burst_min=5)
    for i in range(8):
        assert torch.equal(a.store(_stim(i)), b.store(_stim(i)))
    assert torch.equal(a.fiber.C, b.fiber.C)


def test_online_burst_at_one_is_the_round_write(mod):
    """Gating the round write on `burst_min` = 1 gates nothing: every pre
    fired the round before and every post fires now."""
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    a = AssemblyMemory(_brains(), N, K, P, beta=0.1, w_max=20.0, norm_init=True, rounds=T,
                       strength=0.5, max_items=64)
    b = AssemblyMemory(_brains(), N, K, P, beta=0.1, w_max=20.0, norm_init=True, rounds=T,
                       strength=0.5, max_items=64, write_rule="online_burst", burst_min=1)
    for i in range(8):
        assert torch.equal(a.store(_stim(i)), b.store(_stim(i)))
    assert torch.equal(a.fiber.C, b.fiber.C)


def test_online_burst_writes_only_between_neurons_that_burst(mod):
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    from research.experiments import memory_pattern_efficiency as pe
    mem = AssemblyMemory(_brains(), N, K, P, beta=0.1, w_max=20.0, norm_init=True, rounds=T,
                         strength=0.5, max_items=64, write_rule="online_burst", burst_min=2)
    mem.store(_stim(0))
    burst = mem.last_fired >= 2
    for b in range(SEEDS):
        written = mem.fiber.C[b] > 0
        assert bool(written.any())
        allowed = pe.presence_of(mem.fiber.pres, b, N) & burst[b].view(-1, 1) & burst[b].view(1, -1)
        assert not bool((written & ~allowed).any())


def test_the_deferred_write_counts_the_items_own_transitions(mod):
    """After one item under the deferred write, a present synapse holds the
    number of rounds whose pre fired the round before and whose post fired
    in it -- the round write's counts, of rounds that ran without writing."""
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    from research.experiments import memory_pattern_efficiency as pe
    mem = AssemblyMemory(_brains(), N, K, P, beta=0.1, w_max=20.0, norm_init=True, rounds=T,
                         strength=0.5, max_items=64, write_rule="deferred")
    seen, project = [], mem.area.project

    def recording(*args, **kwargs):
        out = project(*args, **kwargs)
        seen.extend(kwargs["record"])
        return out
    mem.area.project = recording
    mem.store(_stim(0))
    assert len(seen) == T
    for b in range(SEEDS):
        x = [torch.zeros(N, dtype=torch.float32, device=seen[0].device).index_fill_(0, r[b], 1.0)
             for r in seen]
        count = sum(torch.outer(x[t - 1], x[t]) for t in range(1, T))
        present = pe.presence_of(mem.fiber.pres, b, N)
        assert torch.equal(mem.fiber.C[b].to(torch.float32), count * present)


def test_a_chosen_sequence_writes_its_element_transitions(mod):
    """store_sequence at one round per element: the area is inhibited once,
    and a present synapse holds the number of consecutive rounds whose pre
    fired in one and whose post fired in the next -- element to element."""
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    from research.experiments import memory_pattern_efficiency as pe
    mem = AssemblyMemory(_brains(), N, K, P, beta=0.1, w_max=20.0, norm_init=True, rounds=1,
                         strength=0.5, max_items=64)
    elements = [_stim(e) for e in range(5)]
    states, rounds = mem.store_sequence(elements, rounds_per_element=1)
    assert states.shape[:2] == (5, SEEDS) and rounds.shape[0] == 5
    assert torch.equal(states, rounds)
    for b in range(SEEDS):
        x = [torch.zeros(N, dtype=torch.float32, device=rounds.device).index_fill_(0, r[b], 1.0)
             for r in rounds]
        count = sum(torch.outer(x[t - 1], x[t]) for t in range(1, 5))
        present = pe.presence_of(mem.fiber.pres, b, N)
        assert torch.equal(mem.fiber.C[b].to(torch.float32), count * present)

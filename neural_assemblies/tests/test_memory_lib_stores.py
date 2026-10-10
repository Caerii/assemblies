"""memory_lib's Plan and Store write and read what the registered loops wrote and read, bit for
bit: a plain Store is memory_sleep.build_store, a comparator Store memory_lifecycle.comparator_store,
its replay memory_sleep.reliability, its gates Amendment 54's and 55's, its lifetime plan the
lifetime probe's words."""
from __future__ import annotations

import pytest

from neural_assemblies.tests import _devices

torch = _devices.import_torch()
pytestmark = pytest.mark.requires_cuda

DEV = "cuda"
SEEDS = [900, 901, 902, 903]
N, K, P, TAU = 2000, 60, 0.5, 17


@pytest.fixture(scope="module")
def mod():
    return _devices.fused_kernels()


def _cell():
    from research.experiments import memory_lib as lib
    return lib.Cell.of(N, K, P, tau=TAU)


def _spec():
    return _cell().spec(0.05)


def test_a_plain_store_is_build_store_and_reads_as_reliability(mod):
    from research.experiments import memory_sleep as sl
    from research.experiments.memory_lib.stores import Plan, Store
    ref = sl.build_store(_spec(), 40, SEEDS, DEV)
    st = Store(Plan.of(_cell(), SEEDS, 40), DEV).write()
    rec = st.record()
    assert (rec["M"], rec["L"]) == (ref["M"], ref["L"])
    assert torch.equal(rec["allst"], ref["allst"]) and torch.equal(rec["wordof"], ref["wordof"])
    assert torch.equal(st.mem.fiber.C, ref["mem"].fiber.C)
    assert st.replay() == sl.reliability(ref, DEV)


def test_a_store_written_in_parts_is_the_store_written_at_once(mod):
    from research.experiments.memory_lib.stores import Plan, Store
    a = Store(Plan.of(_cell(), SEEDS, 40), DEV).write()
    b = Store(Plan.of(_cell(), SEEDS, 40), DEV)
    while b.written < b.plan.M:
        b.write(3)
    assert torch.equal(a.record()["allst"], b.record()["allst"]) and a.replay() == b.replay()


def test_a_comparator_store_is_comparator_store(mod):
    from research.experiments import memory_lifecycle as lc
    from research.experiments.memory_lib.stores import Plan, Store
    ref, stats = lc.comparator_store(_spec(), 40, SEEDS, DEV)
    st = Store(Plan.of(_cell(), SEEDS, 40), DEV, compare=True).write()
    assert torch.equal(st.record()["allst"], ref["allst"]) and torch.equal(st.mem.fiber.C, ref["mem"].fiber.C)
    assert int(st.writer.flags) == stats["flag"] > 0


def test_the_gates_are_amendment_54s_and_55s(mod):
    from research.experiments import memory_setpoint_sleep as sp
    from research.experiments import memory_sleep as sl
    from research.experiments import memory_write_separation as ws
    from research.experiments.memory_lib.stores import Plan, Store, birth_setpoint, reference_median
    refs = [904, 905, 906, 907]
    ref = sl.build_store(_spec(), 10, refs, DEV)
    expected = float(sp.brain_maxima(ref["mem"], len(refs), DEV).median()) * sl.MARGIN
    assert reference_median(_cell(), refs, DEV) == expected
    st = Store(Plan.of(_cell(), SEEDS, 40), DEV)
    mine = birth_setpoint(st, DEV)
    empty = ws.build(N, K, P, TAU, SEEDS, DEV)
    assert torch.equal(mine, (sp.brain_maxima(empty, len(SEEDS), DEV) * sl.MARGIN).to(mine.device))
    st.write(2)
    with pytest.raises(ValueError, match="at birth"):
        birth_setpoint(st, DEV)


def test_the_lifetime_plan_is_the_lifetime_probes(mod):
    from research.experiments import memory_lib as lib
    from research.experiments import memory_reuse_grammar as rg
    from research.experiments.memory_lib.stores import Plan
    days, per_day, uses_end = 24, 43, 100
    M = days * per_day
    plan = Plan.lifetime(lib.Cell.of(10000, 75, 0.48), SEEDS, days, per_day, uses_end, 777_000 + M)
    V = max(8, round(M * rg.LENGTH / uses_end))
    assert plan.V == V == 165
    for sd, w in zip(SEEDS, plan.words):
        assert (w == rg.walks(sd, M, V, None, 777_000 + M)).all()

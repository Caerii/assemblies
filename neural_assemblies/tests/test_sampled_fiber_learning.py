"""Sampled masks isolate weight updates, not recruitment or area-level state."""
import numpy as np
import pytest

from neural_assemblies import Brain


def make_brain(scaling=False, deferred=False):
    b = Brain(engine="numpy_sparse", p=.3, seed=101, norm_init=False, w_max=100,
              synaptic_scaling=scaling, synaptic_scaling_deferred=deferred)
    for name in ("A", "B", "T"):
        b.add_area(name, 8, 2, .5)
        b.materialize_area(name)
    for source in ("A", "B"):
        b.areas[source].winners = [0, 1]
        b.connectomes[source]["T"].weights[:] = 0
    b.connectomes["A"]["T"].weights[0, [6, 7]] = [10, 9]
    b.connectomes["B"]["T"].weights[0] = 1
    return b


@pytest.mark.parametrize("scaling", [False, True])
@pytest.mark.parametrize("mode", ["ordinary", "fixed", "compiled"])
def test_mask_blocks_hebbian_and_triggered_scaling(scaling, mode):
    b = make_brain(scaling)
    if mode == "compiled":
        b._engine._areas["T"]._freeze_connectome_growth = True
        b._engine._areas["T"]._plasticity_only_mode = True
        assert b._engine._use_compiled_projection(b._engine._areas["T"])
    if mode == "fixed":
        b.areas["T"].winners = [6, 7]
        b.areas["T"].fix_assembly()
    a = b.connectomes["A"]["T"].weights.copy()
    before_b = b.connectomes["B"]["T"].weights.copy()
    b.set_fiber_plasticity("A", "T", False)
    b.project({}, {"A": ["T"], "B": ["T"]})
    assert list(b.areas["T"].winners) == [6, 7]
    np.testing.assert_array_equal(b.connectomes["A"]["T"].weights, a)
    assert np.any(b.connectomes["B"]["T"].weights != before_b)


def test_deferred_scaling_retains_masked_work_until_scope_exit():
    b = make_brain(scaling=True, deferred=True)
    b.project({}, {"A": ["T"], "B": ["T"]})
    engine = b._engine
    before = b.connectomes["A"]["T"].weights.copy()
    pending = set(engine._pending_scaling[("A", "T")])
    with engine.suppress_fiber_learning([("A", "T")]):
        assert engine.flush_synaptic_scaling() == 1
        np.testing.assert_array_equal(b.connectomes["A"]["T"].weights, before)
        assert engine._pending_scaling[("A", "T")] == pending
        assert ("B", "T") not in engine._pending_scaling
    assert engine.flush_synaptic_scaling() == 1
    assert not engine._pending_scaling
    assert np.any(b.connectomes["A"]["T"].weights != before)


def test_mask_does_not_disable_recruitment():
    b = Brain(engine="numpy_sparse", p=.3, seed=101, norm_init=False)
    b.add_stimulus("s", 20)
    b.add_area("T", 100, 5, .5)
    b.set_fiber_plasticity("s", "T", False)
    b.project({"s": ["T"]}, {})
    assert b._engine.materialized_count("T") >= 5
    before = b._engine._stim_conns["s"]["T"].weights.copy()
    b.areas["T"].fix_assembly()
    b.project({"s": ["T"]}, {})
    np.testing.assert_array_equal(b._engine._stim_conns["s"]["T"].weights, before)


def test_global_engine_disable_blocks_fixed_target_learning():
    b = make_brain()
    b.areas["T"].winners = [6, 7]
    b.areas["T"].fix_assembly()
    before = b.connectomes["A"]["T"].weights.copy()
    b._engine._plasticity_enabled_global = False
    b.project({}, {"A": ["T"]})
    np.testing.assert_array_equal(b.connectomes["A"]["T"].weights, before)


def test_explicit_source_bootstrap_keeps_masked_weights():
    from neural_assemblies.diagnostics import read_assembly
    b = Brain(engine="numpy_sparse", p=.3, seed=101, norm_init=False)
    b.add_area("A", 8, 2, .5, explicit=True)
    b.add_area("T", 8, 2, .5)
    b.areas["A"].winners = [0, 1]
    weights = b.connectomes["A"]["T"].weights
    weights[:] = 0
    weights[0, [6, 7]] = [10, 9]
    before = weights.copy()
    reference = b.clone()
    reference.project({}, {"A": ["T"]})
    b.set_fiber_plasticity("A", "T", False)
    b.project({}, {"A": ["T"]})
    assert set(read_assembly(b, "T")) == {6, 7}
    np.testing.assert_array_equal(weights, before)
    assert np.any(reference.connectomes["A"]["T"].weights > before)


def test_fixed_target_uses_public_cap_instead_of_stale_backend_cap():
    b = make_brain()
    b._engine.set_winners("T", [0, 1])
    b.areas["T"].winners = [6, 7]
    b.areas["T"].fix_assembly()
    b.project({}, {"A": ["T"]})
    np.testing.assert_array_equal(b._engine.get_winners("T"), [6, 7])
    np.testing.assert_array_equal(b.areas["T"].winners, [6, 7])

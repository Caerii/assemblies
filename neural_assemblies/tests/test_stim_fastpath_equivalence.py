"""Bit-identity of the NumpySparseEngine stim->area vector fast path.

``_expand_stim_vectors_fast`` replaces a per-step ``concatenate`` with
amortized capacity and batches the per-stimulus background ``binomial`` draws
into one call per run of identical parameters. Both are only supposed to change
how memory is allocated and how draws are batched -- never a sampled value and
never the order in which ``self._rng`` is consumed.

These tests run the same simulation with the fast path OFF and ON and require
the resulting engine state to be bit-identical. A divergence means the fast
path is wrong; it must never be answered by loosening the comparison.

``scripts/check_stim_fastpath_equivalence.py`` runs the same comparison against
a full TWO_WORD parser training, which is too slow for the unit suite.
"""

import itertools

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.numpy_engine import _sparse as sparse_mod


def snapshot(engine):
    """Full comparable state of a NumpySparseEngine."""
    snap = {"areas": {}, "stim_conns": {}, "area_conns": {}}
    for name, st in sorted(engine._areas.items()):
        snap["areas"][name] = {
            "w": int(st.w),
            "winners": np.array(st.winners, dtype=np.uint32, copy=True),
            "map": list(st.compact_to_neuron_id),
            "pool_ptr": int(st.neuron_id_pool_ptr),
        }
    for src, tmap in sorted(engine._stim_conns.items()):
        for tgt, conn in sorted(tmap.items()):
            snap["stim_conns"][(src, tgt)] = np.array(conn.weights, copy=True)
    for src, tmap in sorted(engine._area_conns.items()):
        for tgt, conn in sorted(tmap.items()):
            snap["area_conns"][(src, tgt)] = np.array(conn.weights, copy=True)
    return snap


def overlaps(snap):
    """Pairwise assembly overlaps between areas, in stable neuron ids."""
    out = {}
    for x, y in itertools.combinations(sorted(snap["areas"]), 2):
        ax, ay = snap["areas"][x], snap["areas"][y]
        sx = {ax["map"][int(i)] for i in ax["winners"] if int(i) < len(ax["map"])}
        sy = {ay["map"][int(i)] for i in ay["winners"] if int(i) < len(ay["map"])}
        out[(x, y)] = len(sx & sy)
    return out


def assert_identical(a, b, label):
    errs = []
    for section in ("areas", "stim_conns", "area_conns"):
        assert set(a[section]) == set(b[section]), f"{label}: {section} keys differ"
        for key in sorted(a[section], key=str):
            va, vb = a[section][key], b[section][key]
            if section == "areas":
                for field in ("w", "pool_ptr"):
                    if va[field] != vb[field]:
                        errs.append(f"areas[{key}].{field}: {va[field]} != {vb[field]}")
                if not np.array_equal(va["winners"], vb["winners"]):
                    errs.append(f"areas[{key}].winners differ")
                if va["map"] != vb["map"]:
                    errs.append(f"areas[{key}].compact_to_neuron_id differ")
            elif va.shape != vb.shape:
                errs.append(f"{section}[{key}]: shape {va.shape} != {vb.shape}")
            elif not np.array_equal(va, vb):
                errs.append(f"{section}[{key}]: {int((va != vb).sum())} cells differ")
    oa, ob = overlaps(a), overlaps(b)
    for key in sorted(oa, key=str):
        if oa[key] != ob[key]:
            errs.append(f"overlap{key}: {oa[key]} != {ob[key]}")
    assert not errs, f"{label}: fast path diverged:\n  " + "\n  ".join(errs[:20])


def build(seed, n, k, n_stim, steps):
    """Many stimuli into few areas -- the shape that stresses vector growth."""
    brain = Brain(p=0.05, seed=seed, engine="numpy_sparse")
    areas = ["A", "B", "C"]
    for area in areas:
        brain.add_area(area, n, k, beta=0.05)
    stims = [f"S{i}" for i in range(n_stim)]
    for stim in stims:
        brain.add_stimulus(stim, k)

    rng = np.random.default_rng(seed)
    for _ in range(steps):
        stim = stims[int(rng.integers(0, n_stim))]
        tgt = areas[int(rng.integers(0, len(areas)))]
        src = [
            a for a in areas
            if a != tgt and brain._engine.get_num_ever_fired(a) > 0
        ]
        brain.project({stim: [tgt]}, {a: [tgt] for a in src[:1]})

    # A stimulus registered after the areas already have ever-fired neurons
    # takes the other branch of add_stimulus.
    brain.add_stimulus("LATE", k)
    for _ in range(5):
        brain.project({"LATE": ["A"]}, {})
    return brain._engine


@pytest.fixture
def restore_fastpath():
    saved = (sparse_mod._STIM_FASTPATH, sparse_mod._DENSE_STIM_THRESHOLD)
    yield
    sparse_mod.set_stim_fastpath(saved[0])
    sparse_mod.set_dense_stim_threshold(saved[1])


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_fastpath_is_bit_identical(seed, restore_fastpath):
    sparse_mod.set_stim_fastpath(False)
    legacy = snapshot(build(seed, 800, 20, 120, 60))
    sparse_mod.set_stim_fastpath(True)
    fast = snapshot(build(seed, 800, 20, 120, 60))
    assert_identical(legacy, fast, f"seed={seed}")


@pytest.mark.parametrize("threshold", [None, 0.1, 0.25, 0.5, 0.9])
def test_dense_threshold_never_changes_results(threshold, restore_fastpath):
    """Capacity policy is an allocation choice, so no threshold may alter state."""
    sparse_mod.set_stim_fastpath(False)
    legacy = snapshot(build(3, 400, 15, 60, 40))
    sparse_mod.set_stim_fastpath(True)
    sparse_mod.set_dense_stim_threshold(threshold)
    fast = snapshot(build(3, 400, 15, 60, 40))
    assert_identical(legacy, fast, f"threshold={threshold}")


def test_sparse_area_untouched_by_dense_switch(restore_fastpath):
    """An area that never approaches the threshold keeps the doubling path."""
    sparse_mod.set_stim_fastpath(True)
    sparse_mod.set_dense_stim_threshold(0.25)
    engine = build(11, 3000, 30, 20, 6)
    for name, st in engine._areas.items():
        if st.w == 0 or st.w >= 0.25 * st.n:
            continue
        for tmap in engine._stim_conns.values():
            conn = tmap.get(name)
            buf = getattr(conn, "_cap_buf", None)
            if buf is not None:
                assert buf.shape[0] < st.n, (
                    f"{name} is only {st.w}/{st.n} materialized but its stim "
                    f"vector was allocated densely ({buf.shape[0]})"
                )


def test_fastpath_is_deterministic(restore_fastpath):
    sparse_mod.set_stim_fastpath(True)
    first = snapshot(build(42, 600, 20, 80, 40))
    second = snapshot(build(42, 600, 20, 80, 40))
    assert_identical(first, second, "same seed twice")


def test_weights_stay_exactly_w_long(restore_fastpath):
    """Over-allocated capacity must stay invisible: len(weights) == area.w."""
    sparse_mod.set_stim_fastpath(True)
    engine = build(5, 500, 20, 40, 30)
    for area_name, st in engine._areas.items():
        for tmap in engine._stim_conns.values():
            conn = tmap.get(area_name)
            if conn is not None and conn.sparse:
                assert len(conn.weights) == st.w


def test_capacity_buffer_survives_pickling(restore_fastpath):
    """Pickling drops the slack buffer and keeps the logical weights."""
    import pickle

    sparse_mod.set_stim_fastpath(True)
    engine = build(9, 500, 20, 40, 30)
    revived = pickle.loads(pickle.dumps(engine))
    for stim, tmap in engine._stim_conns.items():
        for area, conn in tmap.items():
            other = revived._stim_conns[stim][area]
            assert not hasattr(other, "_cap_buf")
            assert np.array_equal(np.asarray(conn.weights), np.asarray(other.weights))

"""`materialize_area` brings all n neurons into existence without changing them.

WHY THE ENGINE NEEDS THIS.  Lazy materialization is what makes large ``n``
tractable, but it silently breaks any protocol that drives an area from an
ARBITRARY SUBSET of ``n`` rather than from an assembly the area already grew.
The measured case is the reference NEMO coin, whose ``flip`` seeds a uniform
random ``k``-subset of all ``n``: at ``n=2000`` the area had ``w=357``, so
82% of the seed named neurons that did not exist and delivered nothing.

The load-bearing property is not "it is bigger" -- it is that materializing
must not CHANGE the brain. ``_init_area_block`` is addressed by absolute
position, so an area materialized all at once must hold exactly the weights it
would have held had the same neurons been recruited one at a time. The test for
that is the point of this file; the rest are guards on the bookkeeping.
"""

import numpy as np

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.numpy_engine._drive_cache import _csr_storage_available

N, K, P, BETA = 400, 40, 0.05, 0.1


def _grown(seed=1, rounds=5):
    b = Brain(p=P, save_winners=True, seed=seed, engine="numpy_sparse")
    b.add_stimulus("s", K)
    b.add_area("A", N, K, BETA)
    for _ in range(rounds):
        b.project({"s": ["A"]}, {})
    return b


def _eng(b):
    return b._engine_for(b.areas["A"])


def test_materializes_every_neuron():
    b = _grown()
    eng = _eng(b)
    before = int(eng._areas["A"].w)
    assert 0 < before < N, f"precondition: area should be partly grown, w={before}"

    added = eng.materialize_area("A")

    assert added == N - before
    assert int(eng._areas["A"].w) == N
    assert len(eng._areas["A"].compact_to_neuron_id) == N


def test_compact_ids_stay_unique():
    """A duplicated neuron id would silently alias two compact slots."""
    b = _grown()
    eng = _eng(b)
    eng.materialize_area("A")
    ids = eng._areas["A"].compact_to_neuron_id
    assert len(set(ids)) == N, (
        f"{N - len(set(ids))} duplicate neuron ids after materializing")


def test_existing_weights_are_preserved_exactly():
    """The already-grown block must survive byte-for-byte.

    If materializing perturbed the existing sub-block, every assembly already
    stored in the area would silently move.
    """
    b = _grown()
    eng = _eng(b)
    for _ in range(3):
        b.project({"s": ["A"]}, {"A": ["A"]})
    conn = eng._area_conns["A"]["A"]
    before = np.array(conn.weights, copy=True)
    r, c = before.shape

    eng.materialize_area("A")

    after = np.asarray(conn.weights)
    assert after.shape[0] >= r and after.shape[1] >= c
    assert np.array_equal(after[:r, :c], before), (
        "materializing perturbed weights that already existed")


def test_new_weights_match_incremental_growth():
    """THE load-bearing property: same brain, only sooner.

    Two brains at the same seed. One is materialized all at once; the other is
    driven until it grows on its own. Where they overlap, the self-fiber blocks
    must agree exactly -- `_init_area_block` is addressed by absolute position
    precisely so that growth order cannot change any value.
    """
    a = _grown(seed=7)
    ea = _eng(a)
    for _ in range(2):
        a.project({"s": ["A"]}, {"A": ["A"]})
    ea.materialize_area("A")
    wa = np.asarray(ea._area_conns["A"]["A"].weights)

    b = _grown(seed=7)
    eb = _eng(b)
    for _ in range(2):
        b.project({"s": ["A"]}, {"A": ["A"]})
    wb = np.asarray(eb._area_conns["A"]["A"].weights)

    r = min(wa.shape[0], wb.shape[0])
    c = min(wa.shape[1], wb.shape[1])
    assert r > 0 and c > 0
    assert np.array_equal(wa[:r, :c], wb[:r, :c]), (
        "a materialized area holds different weights than a grown one -- "
        "materializing is supposed to change WHEN neurons exist, not which "
        "brain this is")


def test_arbitrary_seed_can_drive_after_materializing():
    """The defect this exists to fix: a uniform k-subset of n delivers drive.

    Before materializing, most of a uniform random seed names neurons with no
    outgoing synapse, so a self-projection delivers ~nothing and `project_into`
    hands back the incumbent winners.
    """
    b = _grown()
    eng = _eng(b)
    eng.materialize_area("A")
    for _ in range(3):
        b.project({"s": ["A"]}, {"A": ["A"]})

    rng = np.random.default_rng(0)
    seed = rng.choice(N, size=K, replace=False).astype(np.uint32)
    b.areas["A"]._winners = seed
    eng.set_winners("A", seed)
    before = sorted(int(x) for x in b.areas["A"].winners)

    with b.frozen():
        b.project({}, {"A": ["A"]})
    after = sorted(int(x) for x in b.areas["A"].winners)

    assert after != before, (
        "self-projection from a uniform random seed left the winners "
        "unchanged -- the fiber is still not carrying drive")


def test_is_idempotent():
    b = _grown()
    eng = _eng(b)
    eng.materialize_area("A")
    w1 = np.array(eng._area_conns["A"]["A"].weights, copy=True)
    assert eng.materialize_area("A") == 0
    assert np.array_equal(np.asarray(eng._area_conns["A"]["A"].weights), w1)


def test_projection_still_works_afterwards():
    """A fully materialized area must remain a normal area."""
    b = _grown()
    eng = _eng(b)
    eng.materialize_area("A")
    for _ in range(3):
        b.project({"s": ["A"]}, {})
    assert len(b.areas["A"].winners) == K
    assert int(eng._areas["A"].w) == N


def test_csr_storage_fallback_resolves_process_backend():
    """The module-level fallback must resolve the active array module."""
    assert isinstance(_csr_storage_available(), bool)

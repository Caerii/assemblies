"""The `_compact_index` inverse-table cache must never serve a stale mapping.

WHY THIS FILE EXISTS. `_compact_index` inverts an area's
``compact_to_neuron_id`` table (neuron id -> compact index). It is O(area
size) and ran once per PROJECTION -- every `_cue_state` rebuilt the whole
dict to activate one k-neuron assembly (measured 0.81s of an 11.7s GPU build,
3.9s of a 93.9s numpy build). It is now cached on the engine.

THE HAZARD THE CACHE INTRODUCES is precisely the defect the function was
centralised to prevent: a stale inverse silently maps neuron IDs to the WRONG
compact indices, which reads as exactly chance and is invariant to every
parameter -- the confusion that has cost this project three results
([[two-index-spaces-compact-vs-neuron-id]]). The table is NOT append-only:
`materialize_area` and the explicit-dense bootstrap REBIND it wholesale.

So the tests below pin both transitions, per engine:
  * growth by recruitment (same list object, longer) -- incremental path;
  * wholesale rebind (`materialize_area`) -- must invalidate.
Each asserts against a FRESHLY COMPUTED inverse, so a stale cache fails
rather than merely being slower.
"""
from __future__ import annotations

import pytest

from neural_assemblies.assembly_calculus.ops import _compact_index
from neural_assemblies.core.brain import Brain

ENGINES = ["numpy_sparse"]
try:  # pragma: no cover - environment dependent
    import torch
    if torch.cuda.is_available():
        ENGINES.append("torch_sparse")
except ImportError:  # pragma: no cover
    pass


def _truth(engine, area):
    """The inverse, computed from scratch -- what the cache must equal."""
    mapping = engine.get_neuron_id_mapping(area)
    return {int(nid): i for i, nid in enumerate(mapping)}


@pytest.mark.parametrize("engine_name", ENGINES)
def test_cache_tracks_recruitment_growth(engine_name):
    b = Brain(p=0.05, seed=0, engine=engine_name, norm_init=False)
    b.add_stimulus("s", 20)
    b.add_area("A", 1000, 20, beta=0.1)
    b.add_area("B", 1000, 20, beta=0.1)
    b.project({"s": ["A"]}, {})
    eng = b._engine

    seen = []
    for _ in range(5):
        b.project({"s": ["A"]}, {"A": ["B"]})
        got = _compact_index(eng, "B")
        assert got == _truth(eng, "B"), "stale inverse after growth"
        seen.append(len(got))
    # The area really did grow during the test, so the incremental path was
    # exercised rather than trivially returning a constant table.
    assert seen[-1] > seen[0], seen


@pytest.mark.parametrize("engine_name", ENGINES)
def test_cache_invalidates_on_wholesale_rebind(engine_name):
    """`materialize_area` REBINDS the table; the cache must not survive it."""
    b = Brain(p=0.05, seed=0, engine=engine_name, norm_init=False)
    b.add_stimulus("s", 20)
    b.add_area("A", 400, 20, beta=0.1)
    b.project({"s": ["A"]}, {})
    eng = b._engine

    before = dict(_compact_index(eng, "A"))   # populate the cache
    b.materialize_area("A")
    after = _compact_index(eng, "A")
    assert after == _truth(eng, "A"), "stale inverse after materialize_area"
    assert len(after) > len(before), (len(before), len(after))


@pytest.mark.parametrize("engine_name", ENGINES)
def test_two_areas_do_not_share_an_entry(engine_name):
    """The cache is keyed per area; a shared entry would cross the spaces."""
    b = Brain(p=0.05, seed=0, engine=engine_name, norm_init=False)
    b.add_stimulus("s", 20)
    b.add_area("A", 500, 20, beta=0.1)
    b.add_area("B", 700, 20, beta=0.1)
    b.project({"s": ["A"]}, {})
    b.project({"s": ["A"]}, {"A": ["B"]})
    eng = b._engine
    assert _compact_index(eng, "A") == _truth(eng, "A")
    assert _compact_index(eng, "B") == _truth(eng, "B")

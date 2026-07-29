"""Regression tests for the silent-no-op failure class (#50).

Three bugs in this repo shared one signature: a projection that delivers no
drive still returns k winners, so a dead mechanism is indistinguishable from a
working one at the call site. In results it reads as "mechanism X turns out to
have surprisingly little effect", which looks exactly like a real negative
result.

These tests pin the two engine-level behaviours that make that class detectable
rather than silent, and they are written to FAIL if either regresses:

  1. A source area projecting into an already-grown target must get its weight
     block sized, so it actually contributes. The deferred-init path used to sit
     after the zero-signal early return in `project_into`, making it unreachable
     in precisely the case it existed for -- when the new source was the only
     source.
  2. `ASSEMBLIES_STRICT_DRIVE=1` must warn when a projection delivers nothing.
"""

from __future__ import annotations

import warnings

import pytest

from neural_assemblies.core.brain import Brain

N, K, P, ROUNDS = 500, 25, 0.2, 5


def _brain():
    b = Brain(p=P, save_winners=True, seed=7, engine="numpy_sparse")
    for name in ("SRC_A", "SRC_B", "TGT"):
        b.add_area(name, N, K, 0.1)
    b.add_stimulus("stim_a", K)
    b.add_stimulus("stim_b", K)
    return b


def _drive(brain, stim, src, n=ROUNDS):
    """Build an assembly in *src*, then project src -> TGT and snapshot."""
    from neural_assemblies.assembly_calculus.ops import _snap
    for _ in range(n):
        brain.project({stim: [src]}, {src: [src]})
    brain.areas[src].fix_assembly()
    for _ in range(n):
        brain.project({}, {src: ["TGT"]})
    out = _snap(brain, "TGT")
    brain.areas[src].unfix_assembly()
    return out


def test_second_source_into_grown_target_contributes():
    """SRC_B must not be silently ignored because SRC_A got there first.

    The bug produced a STALE assembly, not an empty one: the zero-signal branch
    preserves the target's winners, so SRC_B's "stored" assembly was whatever
    SRC_A had last written. Asserting inequality is therefore the test -- and it
    must hold in BOTH orders, since the original symptom flipped with training
    order (whichever source ran first was the only one that worked).
    """
    brain = _brain()
    a = _drive(brain, "stim_a", "SRC_A")
    b = _drive(brain, "stim_b", "SRC_B")

    conn = brain._engine._area_conns["SRC_B"]["TGT"]
    assert conn.weights.shape[1] > 0, (
        "SRC_B->TGT weight block was never sized, so SRC_B delivers zero "
        "drive into TGT while the projection still returns k winners"
    )

    shared = set(int(x) for x in a.winners) & set(int(x) for x in b.winners)
    assert len(shared) < len(a.winners), (
        f"SRC_B's stored assembly is identical to SRC_A's "
        f"({len(shared)}/{len(a.winners)} shared) -- the projection from SRC_B "
        f"delivered no drive and TGT kept SRC_A's assembly"
    )


def test_reversed_source_order_also_contributes():
    """The mirror image, because the original bug was order-dependent."""
    brain = _brain()
    b = _drive(brain, "stim_b", "SRC_B")
    a = _drive(brain, "stim_a", "SRC_A")
    assert brain._engine._area_conns["SRC_A"]["TGT"].weights.shape[1] > 0
    shared = set(int(x) for x in a.winners) & set(int(x) for x in b.winners)
    assert len(shared) < len(a.winners)


def test_strict_drive_warns_on_zeroed_connectome(monkeypatch):
    """A reset connectome must be loud under ASSEMBLIES_STRICT_DRIVE.

    This is the configuration that returned bit-identical assemblies for every
    input via the k-WTA index tie-break, and read as retrieval at exactly chance
    with a unit margin, invariant to beta.
    """
    monkeypatch.setenv("ASSEMBLIES_STRICT_DRIVE", "1")
    brain = _brain()
    _drive(brain, "stim_a", "SRC_A")
    brain._engine.reset_area_connections("TGT")
    brain.areas["SRC_A"].fix_assembly()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        brain.project({}, {"SRC_A": ["TGT"]})
    brain.areas["SRC_A"].unfix_assembly()
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, RuntimeWarning)]
    assert any("ZERO drive" in m for m in msgs), (
        f"a projection into a zeroed connectome was silent; warnings: {msgs}"
    )


def test_strict_drive_silent_when_drive_is_real():
    """No false positives: a healthy projection must not warn."""
    import os
    os.environ["ASSEMBLIES_STRICT_DRIVE"] = "1"
    try:
        brain = _brain()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _drive(brain, "stim_a", "SRC_A")
        msgs = [str(w.message) for w in caught
                if issubclass(w.category, RuntimeWarning)
                and "ZERO drive" in str(w.message)]
        assert not msgs, f"healthy projection warned: {msgs}"
    finally:
        os.environ.pop("ASSEMBLIES_STRICT_DRIVE", None)


def test_train_roles_leaves_a_live_role_connectome():
    """The shipped parser must not destroy what it just learned.

    Measured before the fix: `reset_area_connections` fired 138 times on a
    40-sentence corpus and all 138 zeroed a pathway carrying weight, leaving
    SEQ->SEQ as the only live area->area pathway in the brain.
    """
    from neural_assemblies.assembly_calculus.parser import NemoParser
    from neural_assemblies.diagnostics import FiberState, fiber_census

    brain = Brain(p=0.2, save_winners=True, seed=42, engine="numpy_sparse")
    parser = NemoParser(brain, n=1000, k=50, beta=0.0718, rounds=10)
    parser.setup_areas()
    for w in ("dog", "cat", "bird"):
        parser.register_word(w, "noun", f"vis_{w}")
    for w in ("chases", "sees"):
        parser.register_word(w, "verb", f"mot_{w}")
    parser.train_lexicon()
    parser.train_roles([["dog", "chases", "cat"], ["cat", "sees", "bird"]])

    rows = [f for f in fiber_census(brain) if isinstance(f, FiberState)]
    live = {(f.src, f.dst) for f in rows if not f.dead}
    for pair in (("LEX_NOUN", "ROLE_AGENT"), ("LEX_NOUN", "ROLE_PATIENT"),
                 ("LEX_VERB", "ROLE_ACTION")):
        assert pair in live, (
            f"{pair[0]}->{pair[1]} carries no weight after train_roles, so "
            f"that role area receives zero drive at parse time"
        )


def test_role_assemblies_are_distinct_after_training():
    """The consequence that matters: stored role assemblies must differ.

    With a zeroed connectome every candidate has equal input and the
    deterministic index tie-break returns the same k winners for every word, so
    all stored assemblies were literally the identical set.
    """
    from neural_assemblies.assembly_calculus.parser import NemoParser

    brain = Brain(p=0.2, save_winners=True, seed=42, engine="numpy_sparse")
    parser = NemoParser(brain, n=1000, k=50, beta=0.0718, rounds=10)
    parser.setup_areas()
    for w in ("dog", "cat", "bird"):
        parser.register_word(w, "noun", f"vis_{w}")
    for w in ("chases", "sees"):
        parser.register_word(w, "verb", f"mot_{w}")
    parser.train_lexicon()
    parser.train_roles([["dog", "chases", "cat"], ["cat", "sees", "bird"],
                        ["bird", "chases", "dog"]])

    agents = parser.role_lexicons.get("ROLE_AGENT", {})
    assert len(agents) >= 2
    sets = [frozenset(int(x) for x in a.winners) for a in agents.values()]
    assert len(set(sets)) == len(sets), (
        "stored ROLE_AGENT assemblies are not all distinct -- the index "
        "tie-break signature of a zeroed or dead connectome"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

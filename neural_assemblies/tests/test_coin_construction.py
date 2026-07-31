"""``RandomChoiceArea(construction="attractor")`` builds a coin that decides.

WHY THIS FILE EXISTS.  The shipped ``legacy`` construction produces numbers
that look like a coin and are not one: the settled state is barely above chance
overlap with either stored attractor, so the reported 0/1 is decided by the
seed RNG rather than by the dynamics.  The failure is invisible to a fairness
test, which is why ``test_pfa.py``'s existing coverage passed throughout.

THE LOAD-BEARING ASSERTION IS ``decisive``, NOT ``heads``.  A coin returning
pure noise scores ~0.5 on fairness; so does a working one.  Fairness alone
cannot separate them -- measured, the untrained control is *fairer* than the
trained coin, because with nothing stored there is no basin to be lopsided.
Every test here therefore checks how close the settled state lands to an
attractor, against the ``k/n`` chance floor.

Full analysis: ``research/notes/neural_coin_fairness.md``.
"""

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.assembly_calculus.assembly import overlap
from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.assembly_calculus.pfa import RandomChoiceArea

N, K, BETA, P = 2000, 200, 3.0, 0.05
CHANCE = K / N          # 0.10 -- overlap of two independent k-subsets of n
SETTLE = 20


def _coin(construction, seed=1, **kw):
    b = Brain(p=P, save_winners=True, seed=seed, engine="numpy_sparse")
    coin = RandomChoiceArea(b, n=N, k=K, beta=BETA, rounds_train=10,
                            construction=construction, **kw)
    return b, coin


def _decisiveness(b, coin, flips=40, rounds=SETTLE):
    """Mean overlap between the settled state and whichever attractor won."""
    out = []
    for i in range(flips):
        coin.flip(seed=700 + i, rounds=rounds, mode="compete")
        r = _snap(b, coin.area_name)
        out.append(max(overlap(r, coin.asm0), overlap(r, coin.asm1)))
    return float(np.mean(out))


def test_attractor_construction_settles_onto_a_stored_assembly():
    """THE point of the coin: the settled state IS one of the attractors."""
    b, coin = _coin("attractor")
    dec = _decisiveness(b, coin)
    assert dec > 0.8, (
        f"settled state overlaps the winning attractor only {dec:.3f} "
        f"(chance is {CHANCE:.3f}) -- the area is not completing an assembly, "
        f"so the 0/1 it returns comes from the seed, not the dynamics")


def test_legacy_construction_is_near_chance():
    """Pin the defect, so a fix cannot land without this file noticing.

    If this starts failing because ``legacy`` got better, that is good news --
    but the recorded ``coin2024_*`` goldens were produced against this
    behaviour and must be re-recorded in the same change.
    """
    b, coin = _coin("legacy")
    dec = _decisiveness(b, coin)
    assert dec < 0.4, (
        f"legacy decisiveness is {dec:.3f}, expected near the {CHANCE:.3f} "
        f"chance floor -- has the construction been fixed?")


def test_fairness_cannot_tell_the_two_apart():
    """The trap, asserted: ``heads`` does NOT separate working from broken.

    This is the test that justifies every other assertion in the file being
    written against ``decisive``. If someone later "simplifies" those to a
    fairness check, this documents why that is not a test.
    """
    heads = {}
    for construction in ("legacy", "attractor"):
        b, coin = _coin(construction)
        r = [coin.flip(seed=700 + i, rounds=SETTLE, mode="compete")
             for i in range(40)]
        heads[construction] = sum(x == 0 for x in r) / len(r)

    assert abs(heads["legacy"] - 0.5) < 0.25
    assert abs(heads["attractor"] - 0.5) < 0.25, (
        f"both constructions look fair: {heads} -- which is the point. "
        f"Fairness is not evidence that the coin works.")


def test_attractor_construction_materialises_the_area():
    """The seed spans ``n``, so every neuron must exist before the first flip.

    Without this the recurrent fiber spans only the ``w`` neurons that have won
    something, and a uniform ``k``-subset of ``n`` mostly names neurons with no
    outgoing synapse -- measured at 82% of the seed at ``n=2000``.
    """
    b, coin = _coin("attractor")
    eng = b._engine_for(b.areas[coin.area_name])
    assert int(eng._areas[coin.area_name].w) == N


def test_stored_assemblies_survive_the_index_remap():
    """``Assembly.winners`` are neuron ids; ``set_winners`` wants compact ones.

    Feeding one to the other is silently accepted by the engine. Under the
    attractor construction the area is fully materialised, so every stored id
    must map to a compact slot and none may be dropped.
    """
    from neural_assemblies.assembly_calculus.ops import _compact_index
    b, coin = _coin("attractor")
    inv = _compact_index(b._engine_for(b.areas[coin.area_name]),
                         coin.area_name) or {}
    for name, asm in (("asm0", coin.asm0), ("asm1", coin.asm1)):
        missing = [int(x) for x in asm.winners if int(x) not in inv]
        assert not missing, (
            f"{name}: {len(missing)} of {len(asm.winners)} stored neuron ids "
            f"have no compact slot -- they would be dropped from every seed")


def test_settling_is_what_produces_the_decision():
    """rounds=0 must NOT already have the answer.

    The original defect returned identical results at rounds 0 / 1 / 10 across
    200 flips, because the fiber delivered zero drive and the engine handed
    back the seeded winners. A settle loop that changes nothing is the exact
    signature to guard against.
    """
    b, coin = _coin("attractor")
    dec0 = _decisiveness(b, coin, flips=20, rounds=0)
    dec_n = _decisiveness(b, coin, flips=20, rounds=SETTLE)
    assert dec_n > dec0 + 0.3, (
        f"settling bought almost nothing: {dec0:.3f} -> {dec_n:.3f}. The "
        f"recurrent fiber is probably delivering no drive.")


@pytest.mark.slow
def test_untrained_null_stays_at_chance():
    """beta=0: nothing stored, so nothing to settle into.

    Without this arm, "the coin is fair" is equally consistent with the area
    having learned nothing at all.
    """
    b = Brain(p=P, save_winners=True, seed=1, engine="numpy_sparse")
    coin = RandomChoiceArea(b, n=N, k=K, beta=0.0, rounds_train=10,
                            construction="attractor")
    dec = _decisiveness(b, coin, flips=30)
    assert dec < 3 * CHANCE, (
        f"untrained coin reads {dec:.3f} against a {CHANCE:.3f} chance floor "
        f"-- if an area with no plasticity 'decides', the metric is measuring "
        f"something other than a stored attractor")

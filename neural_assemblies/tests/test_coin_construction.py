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

Full analysis: ``research/notes/coin/neural_coin_fairness.md``.
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


@pytest.mark.parametrize("mode", ["k_split", "compete"])
def test_legacy_construction_is_rejected_before_flipping(mode):
    """The historical broken instrument cannot manufacture new fairness data."""
    b, coin = _coin("legacy")
    area = b.areas[coin.area_name]
    prior = area.winners.copy()
    with pytest.raises(ValueError, match="Legacy coin construction"):
        coin.flip(seed=700, mode=mode)
    np.testing.assert_array_equal(area.winners, prior)


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

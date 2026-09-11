"""Reference-faithful word-order learner: single-mood floor and multi-mood.

Port under test: ``neural_assemblies/reference/word_order_learner.py``, a
translation of ``.reference/dmitropolsky-assemblies/word_order_int.py`` (the
author's own implementation of Mitropolsky & Papadimitriou 2025 Sec. 2.4-2.5)
onto this repository's Brain API.

Why this exists alongside ConstituentOrderMixin: the mixin scores the generation
competition by overlap against a stored "role code" assembly, while the paper
and the reference select on TOTAL SYNAPTIC INPUT, through an intermediate helper
layer. Grafting just the helper areas onto the mixin destroyed its order signal;
the chain and the scoring have to move together, so they were ported together.

Unlike the mixin, this port has NO default word order -- an untrained model does
not emit SVO -- so "generated the trained order" is evidence here without
needing the permutation control the mixin required.
"""
import pytest

from neural_assemblies.reference.word_order_learner import WordOrderLearner

pytestmark = pytest.mark.slow

ORDERS = {
    "SVO": ("S", "V", "O"),
    "SOV": ("S", "O", "V"),
    "VSO": ("V", "S", "O"),
    "OVS": ("O", "V", "S"),
}


def _learner(mood_orders, seed=1):
    return WordOrderLearner(
        num_nouns=4, num_verbs=2, mood_orders=mood_orders,
        n=1000, k=50, p=0.05, beta=0.1, seed=seed,
    )


class TestSingleMood:
    """The floor: one mood, one order, generated correctly."""

    @pytest.mark.parametrize("name", sorted(ORDERS))
    def test_generates_trained_order(self, name):
        m = _learner({0: ORDERS[name]})
        m.train(20, mood_index=0)
        got = "".join(m.generate(0))
        assert got == name, f"trained {name}, generated {got}"


class TestMultiMood:
    """Two moods with DIFFERENT orders on ONE brain -- the whole point of the
    helper layer, and what the paper's second sweep axis requires."""

    def test_moods_differing_at_first_constituent(self):
        # Settled by MOOD -> helper, which is outside the circularity below.
        orders = {0: ORDERS["SOV"], 1: ORDERS["OVS"]}
        m = _learner(orders)
        m.train(60)
        for idx, want in ((0, "SOV"), (1, "OVS")):
            got = "".join(m.generate(idx))
            assert got == want, f"mood{idx}: wanted {want}, got {got}"

    @pytest.mark.xfail(
        strict=True,
        reason="SVO+VSO sits ON the decision boundary -- it flips between pass "
               "and fail across historical runs. An unexpected pass now fails "
               "CI so the apparent repair must be reviewed. It passed "
               "consistently only BEFORE the w_max clamp on the dense "
               "explicit->sparse bridge, i.e. while riding unbounded weights "
               "that overflow float past ~120 sentences, so that earlier "
               "result was resting on a numerical bug rather than on learning. "
               "The underlying cause is the same mood-chain collapse: MOOD's "
               "distinct chains form at init (SYNTAX overlap 0.04) and merge "
               "within ~20 training sentences (-> ~1.0).",
    )
    def test_moods_differing_at_first_constituent_svo_vso(self):
        orders = {0: ORDERS["SVO"], 1: ORDERS["VSO"]}
        m = _learner(orders)
        m.train(60)
        for idx, want in ((0, "SVO"), (1, "VSO")):
            got = "".join(m.generate(idx))
            assert got == want, f"mood{idx}: wanted {want}, got {got}"

    @pytest.mark.xfail(
        strict=True,
        reason="moods sharing an OPENING constituent (SVO/SOV both start with "
               "S) must diverge at the second word, which is cued by the "
               "syntactic area -- and that stays mood-blind: SYNTAX_subject "
               "overlaps 1.000 between the two moods. MOOD -> SYNTAX cannot "
               "move the winners because helper -> SYNTAX is reinforced by "
               "every sentence of every mood, so MOOD merely learns to predict "
               "the same mood-independent assembly. Raising MOOD's plasticity "
               "onto the syntactic areas attacks this in the right direction "
               "but is an order of magnitude short: beta 0.3/0.6/0.9 moves the "
               "overlap 0.978/0.956/0.816 and never flips the outcome.",
    )
    def test_moods_sharing_first_constituent(self):
        orders = {0: ORDERS["SVO"], 1: ORDERS["SOV"]}
        m = _learner(orders)
        m.train(60)
        for idx, want in ((0, "SVO"), (1, "SOV")):
            got = "".join(m.generate(idx))
            assert got == want, f"mood{idx}: wanted {want}, got {got}"


class TestPerMoodSyntax:
    """`per_mood_syntax=True`: give each mood its own syntactic areas.

    A deliberate deviation from the paper, which keeps one SUBJ/VERB/OBJ and
    expects the per-mood "distinct chain of assemblies" to be emergent. Measured:
    the emergent version forms those chains correctly at init (SYNTAX overlap
    0.04) but loses them to training (-> 1.00 within ~20 sentences), and that
    collapse survives 100x capacity (n=1e5), the paper's beta=0.06, norm_init,
    and raised MOOD plasticity. Making the distinctness structural removes the
    shared target that was collapsing.
    """

    @pytest.mark.parametrize("a,b", [("SVO", "SOV"), ("SVO", "VSO"),
                                     ("SOV", "OVS"), ("VSO", "OVS")])
    def test_two_moods_different_orders(self, a, b):
        m = WordOrderLearner(
            num_nouns=4, num_verbs=2, mood_orders={0: ORDERS[a], 1: ORDERS[b]},
            n=1000, k=50, p=0.05, beta=0.06, seed=1, per_mood_syntax=True,
        )
        m.train(60)
        for idx, want in ((0, a), (1, b)):
            got = "".join(m.generate(idx))
            assert got == want, f"mood{idx}: wanted {want}, got {got}"

    def test_default_is_off(self):
        """The class stays a faithful reproduction unless asked otherwise."""
        assert WordOrderLearner(mood_orders={0: ORDERS["SVO"]}).per_mood_syntax \
            is False


def test_no_default_word_order():
    """An untrained model must not already prefer SVO.

    The mixin does have such a default, which is why its results need a
    permutation control to mean anything. This port does not, so a matching
    order is evidence on its own.
    """
    seen = set()
    for seed in range(6):
        m = _learner({0: ORDERS["SVO"]}, seed=seed)   # built, NOT trained
        seen.add("".join(m.generate(0)))
    assert seen != {"SVO"}, (
        "untrained model always emits SVO -- there is a default order and "
        "trained-order results would need a permutation control")

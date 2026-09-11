"""Multi-mood word order: what works, and the architectural gap that blocks it.

The paper sweeps NUMBER OF MOODS as one of its two axes: each mood of a language
may impose a different constituent order, and the model must CONDITION on mood.
This matters because with a single mood the task is degenerate -- there is
exactly one correct order, so a learner can ignore everything else and be right.

WHAT WORKS (tested below): mood selection (`set_mood`), a distinct MOOD assembly
per (mood, clause frame), and mood-conditioned firing of the syntactic areas.

WHAT DOES NOT (xfail below): two moods with DIFFERENT word orders on one brain.
The later-trained mood wins both. Root cause is measured, not guessed: the
syntactic assembly is supposed to differ per mood so that SYN[i] -> ROLE[i+1]
lands in disjoint synapses, but SUBJ under two moods still overlaps ~0.34-0.40
where near-zero is needed. `ROLE -> SYN` is reinforced by every sentence of
every mood, so it dominates the k-WTA and MOOD cannot move the winners enough.

Two principled fixes were tried and measured:
  * adding MOOD to the role-competition cue -- destroys ordering outright
    (MOOD carries the "which role OPENS" pairing, so it swamps the weaker
    syntactic drive; SVO and SOV both produced 'OSV');
  * priming SYN from MOOD before the role code arrives -- no better (0.34 ->
    0.40).

The reference implementation is titled "word order learner with INTERMEDIATE
helper TPJ areas" and carries `TPJ_*_helper` areas between the role and syntax
layers, which this repo does not have. That helper layer is where mood-specific
chains can live without competing with the strong role->syntax drive, and
implementing it is the prerequisite for real multi-mood support.
"""
import pytest

from neural_assemblies.assembly_calculus.emergent.parser import EmergentParser
from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
    create_training_sentences,
)
from neural_assemblies.assembly_calculus.emergent.core.areas import (
    ROLE_ACTION, ROLE_AGENT, ROLE_PATIENT, SUBJ,
)
from neural_assemblies.assembly_calculus.ops import _snap
from research.experiments.word_order_generation import (
    ORDERS, reorder, transitive_sentences,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def parser():
    p = EmergentParser(n=1000, k=50, p=0.05, beta=0.1, seed=42, rounds=10)
    p.train(create_training_sentences())
    p._bootstrap_order_paths()
    return p


class TestMoodSelection:
    def test_default_mood(self, parser):
        assert parser.mood == parser.DEFAULT_MOOD

    def test_set_mood(self, parser):
        parser.set_mood("interrogative")
        assert parser.mood == "interrogative"
        parser.set_mood(parser.DEFAULT_MOOD)

    def test_distinct_assembly_per_mood(self, parser):
        parser.set_mood("declarative")
        a = parser._frame_assembly(3)
        parser.set_mood("interrogative")
        b = parser._frame_assembly(3)
        parser.set_mood(parser.DEFAULT_MOOD)
        assert a is not None and b is not None
        assert a.overlap(b) < 0.2, (
            f"mood assemblies not distinct (overlap {a.overlap(b):.3f})")

    def test_mood_conditions_syntactic_assembly(self, parser):
        """Mood does move the syntactic assembly -- just not far enough (see
        the module docstring); this pins that it has an effect at all."""
        parser.set_mood("declarative")
        ma = parser._frame_assembly(3)
        parser.set_mood("interrogative")
        mb = parser._frame_assembly(3)
        parser.set_mood(parser.DEFAULT_MOOD)
        parser._fire_constituent("dog", ROLE_AGENT, mood_assembly=ma)
        sa = _snap(parser.brain, SUBJ)
        parser._fire_constituent("dog", ROLE_AGENT, mood_assembly=mb)
        sb = _snap(parser.brain, SUBJ)
        assert sa.overlap(sb) < 0.95, "mood has no effect on the syntactic area"


@pytest.mark.xfail(
    strict=True,
    reason="two moods with different orders on one brain: the later-trained "
           "mood wins both. SUBJ separates only to ~0.34-0.40 across moods "
           "because ROLE->SYN dominates the k-WTA. Needs the reference's "
           "intermediate TPJ helper areas (see module docstring).",
)
def test_two_moods_different_orders():
    moods = {"declarative": "SVO", "interrogative": "SOV"}
    p = EmergentParser(n=1000, k=50, p=0.05, beta=0.1, seed=42, rounds=10)
    p.train(create_training_sentences())
    held = transitive_sentences()[2]
    hw = {r: w for w, r in zip(held.words, held.roles) if r}

    for mood, order_name in moods.items():
        corpus = [x for x in
                  (reorder(s, ORDERS[order_name]) for s in transitive_sentences())
                  if x]
        withheld_words = [hw[r] for r in ORDERS[order_name]]
        p.set_mood(mood)
        p.train_constituent_order(
            [s for s in corpus if s.words != withheld_words], repetitions=4)

    sym = {hw["agent"]: "S", hw["action"]: "V", hw["patient"]: "O"}
    for mood, want in moods.items():
        p.set_mood(mood)
        p.prepare_scene({ROLE_AGENT: hw["agent"], ROLE_ACTION: hw["action"],
                         ROLE_PATIENT: hw["patient"]})
        got = "".join(sym.get(w, "?")
                      for w in p.generate_from_roles(max_len=6))
        assert got.startswith(want), f"mood={mood}: wanted {want}, got '{got}'"

"""Faithful port of the reference word-order learner (helper-area architecture).

Source: ``.reference/dmitropolsky-assemblies/word_order_int.py`` -- "word order
learner with 'intermediate' helper TPJ areas" -- the author's own implementation
of Mitropolsky & Papadimitriou (2025) Sec. 2.4-2.5.

WHY A SEPARATE PORT RATHER THAN AN EDIT TO ConstituentOrderMixin.  The two
designs differ in more than one area:

* the mixin drives the syntactic area from a filler-INDEPENDENT "role code"
  (``_role_identity``), an abstraction the reference does not have; the
  reference drives ``PHON -> TPJ -> helper -> SYNTAX`` from real word
  assemblies;
* the mixin scores the generation competition with ``bind_strength`` -- overlap
  against a stored identity assembly -- while the reference scores TOTAL
  SYNAPTIC INPUT between live assemblies (``get_total_input``), which is also
  what the paper specifies: "the role area with the most synaptic input will be
  selected".

Grafting only the helper areas onto the mixin was tried and destroyed the order
signal outright (trained SVO produced VSO, SOV produced VSO -- outputs constant
and uncorrelated with training). The chain and the scoring have to move
together, so they are ported together here, standalone and independently
testable, leaving the working mixin untouched.

Architecture (identical to the reference)::

    PHON  ->  TPJ_x  <->  TPJ_x_helper  ->  SYNTAX_x
    MOOD  ->  SYNTAX_x                       (every step)
    MOOD  ->  TPJ_x_helper                   (first word only: which role opens)
    SYNTAX_i -> TPJ_x_helper                 (what follows constituent i)

Word order lives in the ``SYNTAX_i -> helper`` synapses, and the mood
conditioning rides in because ``MOOD -> SYNTAX`` fires at every step, making the
syntactic assemblies themselves mood-specific.

One deliberate improvement over the reference: cross-area drive comparison uses
``input_drive(..., metric="pre_kwta")``, which normalizes per candidate neuron.
The reference sums raw weights over winners, which biases the comparison toward
whichever area happens to have recruited more neurons (this repository measured
two role areas differing 391 vs 449, enough to reverse a ranking on size alone).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

from neural_assemblies.assembly_calculus.binding import input_drive
from neural_assemblies.core.brain import Brain

PHON = "PHON"
MOOD = "MOOD"

TPJ = {"S": "TPJ_agent", "V": "TPJ_action", "O": "TPJ_patient"}
HELPER = {
    "S": "TPJ_agent_helper",
    "V": "TPJ_action_helper",
    "O": "TPJ_patient_helper",
}
SYNTAX = {"S": "SYNTAX_subject", "V": "SYNTAX_verb", "O": "SYNTAX_object"}

CONSTITUENTS = ("S", "V", "O")


class WordOrderLearner:
    """Reference word-order learner: scene -> sentence with learned order."""

    def __init__(
        self,
        *,
        num_nouns: int = 4,
        num_verbs: int = 2,
        mood_orders: Optional[Dict[int, Sequence[str]]] = None,
        n: int = 1000,
        k: int = 50,
        p: float = 0.05,
        beta: float = 0.1,
        seed: int = 0,
        training_fire_rounds: int = 10,
        previous_constituent_fire_rounds: int = 2,
        norm_init: bool = False,
    ):
        self.num_nouns = num_nouns
        self.num_verbs = num_verbs
        self.num_words = num_nouns + num_verbs
        self.mood_orders = dict(mood_orders or {0: ("S", "V", "O")})
        self.num_moods = len(self.mood_orders)
        self.k = k
        self.training_fire_rounds = training_fire_rounds
        self.previous_constituent_fire_rounds = previous_constituent_fire_rounds
        self._rng = random.Random(seed)

        # norm_init is off by default: this is a literature reproduction, and
        # the reference substrate is un-normalized (see the repo's
        # norm_init-substrate-vs-reference convention).
        self.brain = Brain(p=p, seed=seed, norm_init=norm_init)
        # PHON and MOOD are EXPLICIT: each word / mood is a fixed, addressable
        # assembly, activated by index exactly as the reference does.
        self.brain.add_explicit_area(PHON, self.num_words * k, k, beta)
        self.brain.add_explicit_area(MOOD, max(self.num_moods, 1) * k, k, beta)
        for c in CONSTITUENTS:
            self.brain.add_area(TPJ[c], n, k, beta)
            self.brain.add_area(HELPER[c], n, k, beta)
            self.brain.add_area(SYNTAX[c], n, k, beta)

    # -- training ---------------------------------------------------------

    def _activate_role(self, phon_index: int, c: str, firings: int = 10) -> None:
        """Drive a word into its role area and its helper (reference
        ``activate_role``)."""
        tpj, helper = TPJ[c], HELPER[c]
        self.brain.activate(PHON, phon_index)
        self.brain.project({}, {PHON: [tpj]})
        self.brain.project({}, {PHON: [tpj], tpj: [helper, tpj]})
        for _ in range(firings):
            self.brain.project({}, {
                PHON: [tpj],
                tpj: [helper, tpj],
                helper: [tpj, helper],
            })

    def _project_training(
        self, c: str, t: int, *, first_word: bool, previous: Optional[str],
    ) -> None:
        """One training step for constituent `c` (reference
        ``project_training``)."""
        tpj, helper, syn = TPJ[c], HELPER[c], SYNTAX[c]
        # NOTE: an earlier version primed SYN from MOOD alone at t == 0, on the
        # theory that MOOD should establish the syntactic frame before the
        # constituent filled it. It is removed: it deviates from the reference
        # for no measured benefit (mood separation in SYNTAX moved 0.955 ->
        # 0.952), and the one multi-mood pair it appeared to fix (SVO+VSO) was
        # in fact passing on unbounded weights, which the w_max clamp on the
        # dense bridge later exposed. MOOD co-fires with the helper below, as
        # in the reference.
        pmap: Dict[str, List[str]] = {
            PHON: [tpj],
            tpj: [helper, tpj],
            helper: [helper, tpj, syn],
            MOOD: [syn],          # tonic: keeps the syntactic assembly
        }                         # mood-specific, which is what keys the order
        if t > 0:
            pmap[syn] = [syn]
        if first_word:
            # Which constituent OPENS the clause is learned from MOOD.
            pmap[MOOD] = pmap[MOOD] + [helper]
        if t <= self.previous_constituent_fire_rounds and previous is not None:
            # The order synapse: the PREVIOUS constituent's syntactic area
            # fires into THIS constituent's helper, recording "c follows
            # previous".
            pmap[SYNTAX[previous]] = [helper]
        self.brain.project({}, pmap)

    def train_sentence(self, mood_index: int = 0) -> None:
        """Present one random transitive sentence in `mood_index`'s order."""
        subj = self._rng.randrange(self.num_nouns)
        obj = self._rng.randrange(self.num_nouns)
        verb = self._rng.randrange(self.num_nouns, self.num_words)
        self.brain.activate(MOOD, mood_index)
        for c, idx in (("S", subj), ("O", obj), ("V", verb)):
            self._activate_role(idx, c)

        order = list(self.mood_orders[mood_index])
        previous: Optional[str] = None
        for pos, c in enumerate(order):
            for t in range(self.training_fire_rounds):
                self._project_training(
                    c, t, first_word=(pos == 0), previous=previous)
            previous = c

    def train(self, num_sentences: int, mood_index: Optional[int] = None) -> None:
        for _ in range(num_sentences):
            m = (mood_index if mood_index is not None
                 else self._rng.randrange(self.num_moods))
            self.train_sentence(m)

    # -- generation -------------------------------------------------------

    def _strongest(self, source: str, candidates: Sequence[str]) -> str:
        """The candidate helper receiving the most synaptic input from `source`.

        This is the paper's selection rule ("the role area with the most
        synaptic input will be selected") and the reference's
        ``get_biggest_input_TPJ_from_*``, but normalized per candidate neuron so
        the comparison is not decided by which area recruited more neurons.
        """
        drives = input_drive(
            self.brain,
            sources=[source],
            target_areas=[HELPER[c] for c in candidates],
            metric="pre_kwta",
        )
        best = max(candidates, key=lambda c: drives.get(HELPER[c], 0.0))
        return best

    def generate(self, mood_index: int = 0, firings: int = 3) -> List[str]:
        """Generate the constituent order for a scene (reference
        ``generate_random_sentence``). Returns e.g. ``['S', 'V', 'O']``."""
        subj = self._rng.randrange(self.num_nouns)
        obj = self._rng.randrange(self.num_nouns)
        verb = self._rng.randrange(self.num_nouns, self.num_words)

        with self.brain.frozen():          # reference: no_plasticity = True
            self.brain.activate(MOOD, mood_index)
            for c, idx in (("S", subj), ("O", obj), ("V", verb)):
                self._activate_role(idx, c, firings=firings)

            # First constituent: whichever helper MOOD drives hardest.
            current = self._strongest(MOOD, CONSTITUENTS)
            self.brain.project({}, {MOOD: [HELPER[current]]})
            order = [current]

            for _ in range(len(CONSTITUENTS) - 1):
                syn = SYNTAX[current]
                self.brain.project({}, {HELPER[current]: [syn], MOOD: [syn]})
                remaining = [c for c in CONSTITUENTS if c not in order]
                # Refresh every role area's helper, then let the current
                # syntactic area pick the next constituent.
                self.brain.project({}, {
                    syn: [HELPER[c] for c in remaining],
                    **{TPJ[c]: [HELPER[c]] for c in CONSTITUENTS},
                })
                current = self._strongest(syn, remaining)
                order.append(current)
        return order

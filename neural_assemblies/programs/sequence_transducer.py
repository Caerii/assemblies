"""A transducer whose state is INDUCED rather than assigned.

    stim[w]            -> LEX                 the current word
    gstim[w]           -> OUT                 grounding signature (as in #14)
    LEX + SEQ_STATE    -> SEQ_ARC             the conjunction (history, word)
    SEQ_ARC            -> SEQ_STATE           state update, FEED-FORWARD
    SEQ_ARC            -> OUT                 prediction from the conjunction

WHAT THIS IS FOR. #14 gave a helper area a self fiber and asked it to
accumulate a prefix; it collapsed to one attractor (cross-prefix overlap
0.7566) and HALVED next-token MRR against a no-context control. The plan of
record says context lives in TRANSITIONS, not accumulation
([[SEQ-TIME-IN-WEIGHTS]]), so the accumulator is replaced by the organ that A1
and A2 validated. Three differences, each a separately documented failure
channel:

  * no self fiber on the state ([[recurrence-is-the-collapse-channel]]);
  * the update passes through a REFRACTED conjunction, so the arc cannot
    collapse onto whichever conjunct is exposed more
    ([[ARC-CONJUNCT-EXPOSURE]], [[REFRACTION-PROPORTIONAL]]);
  * the organ carries its own density so it clears kp >= 3 ln n inside a brain
    whose ambient p is far lower ([[SEQ-REGIME]], [[SEQ-ORGAN-EMBEDS]]).

Firing OUT together with the state update is what makes this a transducer
rather than an acceptor ([[SEQ-TRANSDUCER]]).

WHAT IS DIFFERENT FROM ``NemoArcFSM``, AND WHY IT IS NOT THAT CLASS. There the
state alphabet is GIVEN: states are disjoint neuron-ID blocks, transitions come
from a table, and each write is teacher-forced onto a named target. Here
nothing names a state. The state area is written only by the arc, so whatever
code it settles into is the code the dynamics produced -- which is exactly the
open question [[SEQ-STATE-CODE-EMERGENT]] records as unproven, and exactly what
this class exists to measure. A class whose states are induced cannot share the
constructor of one whose states are assigned.

The two DO share an arc-and-state core (refracted conjunction, per-fiber
density, one conjunction tick per input symbol), and since 2026-09-09 they
share it in code: `programs/arc_core.add_arc_state_core` here, and
`HashedArcCore` on the hashed substrate ([[one-canonical-way]]). The
refactor was gated on the drive-replay parity of both organs.

THE CLOCK. One tick == one input word. ``tick`` recomputes the arc from
(LEX, SEQ_STATE); everything after it reads a FIXED arc, so repeating the write
strengthens the fiber without advancing the machine. That is what lets
``rounds`` mean "how hard to train this pair" -- #14's meaning -- instead of
silently running the state forward once per round.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from neural_assemblies.assembly_calculus.assembly import Assembly, overlap
from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.programs.arc_core import add_arc_state_core


class SequenceTransducer:
    """Refracted-arc transducer over a word vocabulary. State is induced.

    Areas are named ``{prefix}_lex`` / ``_arc`` / ``_state`` / ``_out``.
    """

    def __init__(
        self,
        brain,
        vocab: Sequence[str],
        *,
        n: int = 10000,
        n_arc: int | None = None,
        n_state: int | None = None,
        k: int = 200,
        beta: float = 0.10,
        organ_p: float | None = None,
        refracted_strength: float = 0.1,
        state_refracted_strength: float = 0.0,
        prefix: str = "_seq",
    ):
        self.brain = brain
        self.vocab = list(vocab)
        self.k = k
        self.n_arc = n_arc or n
        # Sized separately from LEX and OUT so a sweep over the STATE's load
        # does not also resize the areas either side of it and confound itself.
        self.n_state = n_state or n

        self.lex_area = f"{prefix}_lex"
        self.arc_area = f"{prefix}_arc"
        self.state_area = f"{prefix}_state"
        self.out_area = f"{prefix}_out"

        brain.add_area(self.lex_area, n, k, beta)
        # The refracted arc-and-state core, shared with NemoArcFSM. THE STATE
        # MAY BE REFRACTED TOO, and by default is not -- the configuration A3
        # measured collapsing (state overlap 0.985 +/- 0.028 beside an arc at
        # 0.035); turning it on is a RE-MEASUREMENT, PREREG_state_refraction.md.
        add_arc_state_core(brain, prefix, n_arc=self.n_arc, n_state=self.n_state,
                           k=k, beta=beta, refracted_strength=refracted_strength,
                           state_refracted_strength=state_refracted_strength,
                           organ_p=organ_p)
        brain.add_area(self.out_area, n, k, beta)

        # Stimuli keep #14's parameters EXACTLY, including their ambient
        # density. `organ_p` is applied to the four fibers the organ itself
        # drives and to nothing else, so word grounding is bit-for-bit the
        # construction #14 measured and the comparison stays about the organ.
        # The cost is that LEX sits at the ambient kp; `regime_audit` reports
        # it rather than hiding it.
        self._s_stim: Dict[str, str] = {}
        self._g_stim: Dict[str, str] = {}
        for w in self.vocab:
            s, g = f"{prefix}_s_{w}", f"{prefix}_g_{w}"
            brain.add_stimulus(s, k)
            brain.add_stimulus(g, k)
            self._s_stim[w] = s
            self._g_stim[w] = g

        # Connectivity is STRUCTURAL: set before any traffic, as in NemoArcFSM.
        self.organ_p = organ_p
        if organ_p is not None:
            brain.add_connectivity(self.lex_area, self.arc_area, organ_p)
            brain.add_connectivity(self.arc_area, self.out_area, organ_p)

        self.out_signature: Dict[str, Assembly] = {}

    # -- grounding ----------------------------------------------------------

    def ground(self, rounds: int = 5) -> None:
        """Form each word's LEX assembly and its OUT signature.

        The OUT signature is stored in NEURON IDS. #14's harness stored compact
        engine indices instead; that happens to be safe here because sparse
        growth appends and the stored prefix stays valid (measured), but it is
        safe by accident of the growth path rather than by construction, and a
        readout should not depend on that. See [[two-index-spaces-compact-vs-neuron-id]].
        """
        b = self.brain
        for w in self.vocab:
            b.inhibit_areas([self.lex_area, self.out_area, self.arc_area,
                             self.state_area])
            for _ in range(rounds):
                b.project({self._s_stim[w]: [self.lex_area]}, {})
            b.inhibit_areas([self.out_area])
            for _ in range(rounds):
                b.project({self._g_stim[w]: [self.out_area]}, {})
            self.out_signature[w] = _snap(b, self.out_area)

    # -- the clock ----------------------------------------------------------

    def reset(self) -> None:
        """Sentence boundary: no state, no arc, no residual output."""
        self.brain.inhibit_areas([self.lex_area, self.arc_area,
                                  self.state_area, self.out_area])

    def tick(self, word: str, rounds: int = 3) -> None:
        """Advance one input word: settle LEX, then recompute the arc.

        This is the ONLY call that moves the machine forward.
        """
        b = self.brain
        for _ in range(rounds):
            b.project({self._s_stim[word]: [self.lex_area]}, {})
        b.project({}, {self.lex_area: [self.arc_area],
                       self.state_area: [self.arc_area]})

    def write(self, target_word: str, rounds: int = 3) -> None:
        """Teacher-force the OUT signature of *target_word* and update state.

        The arc is fixed here, so the state settles once and every extra round
        potentiates arc -> state and arc -> OUT onto that same settled pair.
        """
        b = self.brain
        for _ in range(rounds):
            b.project({self._g_stim[target_word]: [self.out_area]},
                      {self.arc_area: [self.state_area, self.out_area]})

    def emit(self) -> Assembly:
        """Update the state and read OUT with NO teacher signal."""
        b = self.brain
        b.project({}, {self.arc_area: [self.state_area, self.out_area]})
        return _snap(b, self.out_area)

    def state(self) -> Assembly:
        return _snap(self.brain, self.state_area)

    # -- readout ------------------------------------------------------------

    def rank(self, emitted: Assembly, rng) -> List[str]:
        """Vocabulary ranked by overlap with *emitted*, TIES BROKEN RANDOMLY.

        Vocabulary-order tie-breaks are not neutral here: overlap is quantised
        to multiples of 1/k and is frequently 0 for every candidate, and the
        vocabulary is grouped by word class, so `sorted` alone scored a beta=0
        null at the unigram baseline with nothing learned (`study4/ntp.py`).
        """
        keyed = [(-overlap(emitted, self.out_signature[w]), rng.random(), w)
                 for w in self.vocab]
        keyed.sort()
        return [w for _o, _t, w in keyed]

    # -- training -----------------------------------------------------------

    def train_sentence(self, sentence: Sequence[str], rounds: int = 3) -> None:
        self.reset()
        for a, nxt in zip(sentence, sentence[1:]):
            self.tick(a, rounds=rounds)
            self.write(nxt, rounds=rounds)

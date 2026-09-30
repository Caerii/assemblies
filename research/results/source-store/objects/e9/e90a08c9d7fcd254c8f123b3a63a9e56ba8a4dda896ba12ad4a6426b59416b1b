"""StatePredictionMixin -- next-token prediction from a bounded brain state.

Motivation
----------
The original prediction path accumulates every word of a prefix into a single
running assembly in a ``CONTEXT`` area, then bridges ``CONTEXT -> PREDICTION``.
That design is not part of the published NEMO model and it has a structural
defect: one area of ``k`` winners cannot encode an unbounded prefix, so the
area saturates. Measured on a trained parser, ``CONTEXT`` recruited only ~108
distinct neurons at ``k=100``, which makes the first several prefixes of a
sentence map to *identical* assemblies -- the running context stops
distinguishing anything.

What the literature actually specifies
-------------------------------------
Mitropolsky & Papadimitriou (2025), "Simulated Language Acquisition", has no
context-buffer area. Sequence information lives in three places:

1. Tonic drive -- the scene/semantic assemblies and MOOD "keep firing at every
   step throughout generation".
2. The currently active constituent, held by its own recurrent firing.
3. Learned synapses between the syntactic areas and the role areas. The paper:
   "neurons from SUBJ (from the previous word) also fire into ROLE_action, and
   this is how the fact that verb comes after subject is recorded in the
   synapses between SUBJ and ROLE_action."

So the sequence is carried by *weights between bounded states*, not by an
activity buffer that grows with prefix length.

Design
------
``PREDICTION`` is driven by a bounded state:

* the current word's core (category) assembly -- lexical identity,
* the active syntactic area (SUBJ / OBJ / IOBJ) -- structural position,
* MOOD -- tonic sentence-level drive.

Each source area holds exactly one ``k``-assembly, so nothing accumulates and
no area has to grow with prefix length. The number of distinguishable states is
combinatorial in the *synapses* (core x syntactic position x mood), which is
where the published model puts it.

This mixin is additive: it does not modify or remove the existing
``CONTEXT``-based path, so the two can be measured against each other.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from neural_assemblies.assembly_calculus.ops import (
    activate_assembly, project, _snap,
)
from neural_assemblies.assembly_calculus.readout import readout_all

from ..core.areas import CORE_AREAS, MOOD, OBJ, PREDICTION, SUBJ
from neural_assemblies.core.brain import Brain

if TYPE_CHECKING:
    from ..curriculum.data import GroundedSentence
    from ..core.grounding import GroundingContext
    from ..acquisition.pos_inference import BootstrapScores


class StatePredictionMixin:
    """Next-token prediction from a bounded syntactic + lexical state."""

    brain: Brain
    stim_map: Dict[str, str]
    core_lexicons: Dict[str, Dict]
    inference_rounds: int
    _state_pred_bootstrapped: bool = False

    if TYPE_CHECKING:
        def _word_core_area(self, word: str) -> str: ...
        def classify_word_cached(self, word: str, grounding: Optional[GroundingContext] = None) -> Tuple[str, "BootstrapScores"]: ...
        def _bootstrap_prediction_connectivity(self) -> None: ...
        def _ensure_prediction_lexicon(self) -> None: ...
        def _clear_prediction_activity(self) -> None: ...

    def _bootstrap_state_paths(self) -> None:
        """Materialize the state -> PREDICTION connectomes once, plasticity off.

        Specification: neural_assemblies/ir/VERIFICATION.md#contract-state-path-active-sources

        In sparse mode a projection can only strengthen synapses that exist.
        ``_bootstrap_prediction_connectivity`` materializes phon -> PREDICTION
        and CONTEXT -> PREDICTION only, so without this the core, syntactic and
        MOOD pathways have no columns into PREDICTION and the Hebbian pairing
        has nothing to write to -- retrieval overlap stays at exactly zero.

        WHY THE STIMULUS IS CO-FIRED.  An area->area connectome starts empty
        (shape ``(0, 0)``).  The sparse engine initialises it LAZILY: an empty
        connectome is registered for deferred init during input accumulation
        (``_sparse.py`` ~line 655) and the weights are actually sampled at the
        END of ``project_into`` (~line 908).  But an empty connectome
        contributes NO drive, so a projection whose only source is that empty
        fiber hits the "zero signal -> preserve current assembly" early return
        (~line 682) and never reaches the initialisation block.  The fiber is
        therefore stuck empty forever: it cannot deliver drive until it is
        initialised, and it is not initialised unless something delivers drive.

        Co-firing an arbitrary phonological stimulus into PREDICTION breaks that
        deadlock -- the projection has nonzero drive, skips the early return,
        and the deferred init runs.  Measured on a trained parser: projecting
        ``{CORE: [PREDICTION]}`` alone leaves the connectome at ``(0, 0)``;
        the same projection co-driven by a stimulus yields ``(876, 999)`` with
        43,736 synapses.  This is exactly the pattern
        ``_bootstrap_prediction_connectivity`` already uses for the CONTEXT
        fiber (``project({phon: [PREDICTION]}, {CONTEXT: [PREDICTION]})``).

        Two ordering requirements follow from the deferred-init guard
        ``src.w > 0 and tgt.w > 0``:
        * PREDICTION must already have materialised neurons, so
          ``_ensure_prediction_lexicon`` runs BEFORE this.
        * SUBJ/OBJ are empty until something drives them, so a core area is
          projected into them first.
        """
        if self._state_pred_bootstrapped:
            return
        brain = self.brain
        if not self.stim_map:
            return
        arb_phon = next(iter(self.stim_map.values()))

        with brain.frozen():
            # SUBJ/OBJ hold no assembly until a constituent is placed in them;
            # deferred init needs a non-empty source, so seed them from a core.
            seed_core = next(
                (a for a in CORE_AREAS
                 if a in brain.areas and brain.areas[a].active_count > 0),
                None,
            )
            if seed_core is not None:
                for syn in (SUBJ, OBJ):
                    if syn in brain.areas and brain.areas[syn].active_count == 0:
                        brain.project({}, {seed_core: [syn]})

            sources = [
                a for a in (*CORE_AREAS, SUBJ, OBJ, MOOD)
                if a in brain.areas and brain.areas[a].active_count > 0
            ]
            for area in sources:
                # Co-fire the stimulus so the projection carries drive; see the
                # deadlock explanation above.
                brain.project({arb_phon: [PREDICTION]}, {area: [PREDICTION]})

        brain.inhibit_areas([PREDICTION])
        self._state_pred_bootstrapped = True

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def _syntactic_state_area(self, verb_seen: bool) -> str:
        """Syntactic slot the *next* constituent occupies.

        Before the verb the open slot is the subject; after it, the object.
        This mirrors the paper's SUBJ -> ROLE_action -> OBJ chain rather than
        encoding absolute word position, so it does not grow with length.
        """
        return OBJ if verb_seen else SUBJ

    def _activate_core(self, word: str) -> Optional[str]:
        """Drive `word`'s core area, preferring the stabilized lexicon entry.

        Re-projecting phon -> core applies plasticity and drifts the
        representation between training and inference, so the converged
        snapshot is replayed when one exists.
        """
        if word not in self.stim_map:
            return None
        core_area = self._word_core_area(word)
        stored = self.core_lexicons.get(core_area, {}).get(word)
        if stored is not None:
            activate_assembly(self.brain, stored)
        else:
            project(self.brain, self.stim_map[word],
                    core_area, rounds=self.inference_rounds)
        return core_area

    def _set_state(self, words: List[str], upto: int) -> Optional[Dict[str, List[str]]]:
        """Establish the bounded state for the prefix ``words[:upto]``.

        Returns the ``{source_area: [PREDICTION]}`` map to project from, or
        None when the prefix has no usable content word.
        """
        # Establishing the state must be deterministic. With plasticity on,
        # the core -> syntactic projection is itself a learning event, so the
        # syntactic assembly drifts between the moment a state is bound to a
        # continuation and the moment it is read back -- and retrieval then
        # misses entirely. The state is a readout of where we are in the
        # sentence, not something to learn, so it is built with plasticity off.
        with self.brain.frozen():
            verb_seen = False
            last_core: Optional[str] = None
            for w in words[:upto]:
                cat = self.classify_word_cached(w)[0]
                core = self._activate_core(w)
                if core is not None:
                    last_core = core
                if cat == "VERB":
                    verb_seen = True

            if last_core is None:
                return None

            syn_area = self._syntactic_state_area(verb_seen)
            # Set the syntactic slot from the current constituent, so SUBJ/OBJ
            # hold an assembly rather than stale winners.
            self.brain.project({}, {last_core: [syn_area]})

        sources: Dict[str, List[str]] = {
            last_core: [PREDICTION],
            syn_area: [PREDICTION],
        }
        if MOOD in self.brain.areas and len(self.brain.areas[MOOD].winners):
            sources[MOOD] = [PREDICTION]
        return sources

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_next_token_state(
        self,
        sentences: List["GroundedSentence"],
        *,
        repetitions: int = 1,
    ) -> None:
        """Hebbian bind state -> PREDICTION with the next word as teacher.

        The target is *pinned* to the continuation's own PREDICTION assembly --
        the one the word drives when it actually arrives -- and the state is
        then fired into it so plasticity strengthens state -> target synapses.

        Co-firing the teacher stimulus alongside the state instead would let
        both drive the competition, so the winners are a blend of the two. At
        inference only the state fires, producing a different assembly than the
        one stored in the lexicon, and readout misses. Pinning makes the
        training and inference targets identical by construction. It is also
        the biologically natural reading: the next word arrives and drives
        PREDICTION while the state that anticipated it is still active, which
        is exactly the pairing Hebbian plasticity needs.
        """
        # Order matters: the lexicon pass is what first materializes neurons in
        # PREDICTION, and the deferred-init guard requires a non-empty target
        # (see _bootstrap_state_paths). Bootstrapping the state fibers before it
        # leaves every state -> PREDICTION connectome empty.
        self._bootstrap_prediction_connectivity()
        self._ensure_prediction_lexicon()
        self._bootstrap_state_paths()
        lexicon = getattr(self, "prediction_lexicon", {}) or {}

        for _ in range(max(1, repetitions)):
            for sent in sentences:
                words = [w for w in sent.words if w in self.stim_map]
                for i in range(len(words) - 1):
                    target = lexicon.get(words[i + 1])
                    if target is None:
                        continue
                    sources = self._set_state(words, i + 1)
                    if sources is None:
                        continue
                    self._clear_prediction_activity()
                    activate_assembly(self.brain, target)
                    self.brain.areas[PREDICTION].fix_assembly()
                    try:
                        self.brain.project({}, sources)
                    finally:
                        self.brain.areas[PREDICTION].unfix_assembly()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_next_state(
        self,
        words: List[str],
        *,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Rank continuations of ``words`` from the bounded state."""
        if not words:
            return []
        self._bootstrap_prediction_connectivity()
        self._ensure_prediction_lexicon()   # must precede the state bootstrap
        self._bootstrap_state_paths()
        lexicon = getattr(self, "prediction_lexicon", None)
        if not lexicon:
            return []

        sources = self._set_state(words, len(words))
        if sources is None:
            return []

        self._clear_prediction_activity()
        with self.brain.frozen():
            self.brain.project({}, sources)
            asm = _snap(self.brain, PREDICTION)

        ranked = readout_all(asm, lexicon)
        return ranked[:top_k]

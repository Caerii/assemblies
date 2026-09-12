"""LexiconTrainingMixin -- Word forms into category areas: the first training stage.

Split out of the CoreParserMixin monolith. Bodies are unchanged;
only their address is.
"""


from typing import TYPE_CHECKING, Dict, List, Optional, Set, Sequence, cast
from neural_assemblies.core.brain import Brain
from neural_assemblies.assembly_calculus.assembly import Assembly
from neural_assemblies.assembly_calculus.ops import _snap

from ..core.areas import (
    CORE_AREAS,
    CORE_TO_CATEGORY,
    GROUNDING_TO_CORE,
    DET_CORE,
)
from ..core.grounding import GroundingContext
from ..curriculum.data import GroundedSentence
from ._shared import _MODALITY_FIELDS

if TYPE_CHECKING:
    from ..parser import EmergentParser
    from ..training.compiled import CompiledTopologyParser


class LexiconTrainingMixin:
    """Word forms into category areas: the first training stage."""

    brain: Brain
    k: int
    rounds: int
    fast_training: bool
    stim_map: Dict[str, str]
    word_grounding: Dict[str, GroundingContext]
    core_lexicons: Dict[str, Dict[str, Assembly]]
    _grounding_stim_names_set: Set[str]
    _category_cache: Dict[str, str]

    if TYPE_CHECKING:
        def add_phon_stimulus(self, word: str) -> str: ...

    def _register_vocabulary(self, vocab: Dict[str, GroundingContext]):
        """Register all vocabulary words and their grounding stimuli."""
        for word, ctx in vocab.items():
            # Phonological stimulus -- `add_phon_stimulus` is the only route,
            # so `phon_weight` reaches every registration path.
            self.add_phon_stimulus(word)
            self.word_grounding[word] = ctx

            # Grounding feature stimuli
            for mod in _MODALITY_FIELDS:
                for feat in getattr(ctx, mod):
                    stim_name = f"{mod}_{feat}"
                    if stim_name not in self._grounding_stim_names_set:
                        self.brain.add_stimulus(stim_name, self.k)
                        self._grounding_stim_names_set.add(stim_name)

    def _register_corpus_vocabulary(
        self, sentences: List[GroundedSentence],
    ) -> Set[str]:
        """Register corpus words the static vocabulary table does not cover.

        Returns the set of newly registered words (empty when the corpus adds
        nothing), so callers and tests can assert on coverage rather than infer
        it.

        A ``GroundedSentence`` carries a ``GroundingContext`` per word in
        ``contexts``, which is the same thing ``VOCABULARY`` supplies, so a
        corpus word needs nothing extra to be trainable. Words already
        registered keep their EXISTING context: the static table is
        hand-authored and is treated as authoritative where the two disagree,
        and re-registering would also re-add stimuli.

        Sentences whose ``contexts`` are missing or the wrong length are
        skipped rather than guessed at -- a word with no grounding cannot be
        trained into a core lexicon, and silently inventing one is how the bug
        this fixes stayed hidden.
        """
        added: Set[str] = set()
        for sent in sentences:
            contexts = getattr(sent, "contexts", None)
            if not contexts or len(contexts) != len(sent.words):
                continue
            for word, ctx in zip(sent.words, contexts, strict=True):
                if ctx is None or word in self.word_grounding:
                    continue
                self._register_vocabulary({word: ctx})
                added.add(word)
        return added

    def _grounding_stim_names(self, ctx: GroundingContext) -> List[str]:
        """Return stimulus names for all grounding features in a context."""
        names = []
        for mod in _MODALITY_FIELDS:
            for feat in getattr(ctx, mod):
                names.append(f"{mod}_{feat}")
        return names

    def _word_core_area(self, word: str) -> str:
        """Return the core area name for a word based on its grounding."""
        ctx = self.word_grounding.get(word)
        if ctx is None:
            return DET_CORE
        return GROUNDING_TO_CORE[ctx.dominant_modality]

    # ==================================================================
    # Training
    # ==================================================================

    def _clear_core_activity(self, core_area: str) -> None:
        """Clear core winners without wiping learned connectomes or ID mapping."""
        self.brain.inhibit_areas([core_area])
        if self.brain.is_fixed(core_area):
            self.brain.unfix_assembly(core_area)

    def train_lexicon(
        self,
        holdout_words: Optional[set] = None,
        *,
        skip_known: bool = True,
        words: Optional[Set[str]] = None,
    ):
        """Phase 1: Grounded word learning.

        For each word, project phon + grounding features simultaneously
        into the appropriate core area with recurrence. The assembly that
        forms represents the word in its grammatical category area.

        SIMULTANEITY IS THE MECHANISM.  The word form and its grounding are
        projected in the SAME time step, so both drive the winner selection
        and the resulting assembly is potentiated from both.  That single
        assembly is what later lets the phon stimulus alone retrieve the word
        (the grounded half of the input is no longer needed once the synapses
        exist) and lets grounding alone generalise to a word never heard.
        Presenting them in separate steps would build two assemblies and no
        link.

        Pattern follows readout.py:build_lexicon() â€” clear core winners
        between words without wiping learned connectomes.  The distinction
        matters: winners must be cleared so the next word does not inherit
        this one's activity, but the connectome must NOT be reset, because
        every word in a category shares one core area and the whole point is
        that they accumulate there.

        Holdout words have their grounding stimuli registered but never
        trained.  That is what makes the generalisation test meaningful: the
        features exist and can drive the area, but nothing about the specific
        word was ever written into the connectome.

        Args:
            holdout_words: Optional set of words to skip during training.
                These words' grounding stimuli are still registered in the
                brain, so their features can drive generalization.
            skip_known: Skip words already present in ``core_lexicons``.
            words: When set, only train words in this subset (e.g. corpus vocab).
        """
        from ..training.batch import BatchProjector
        from ..training.compiled import compiled_topology, lexicon_topology_spec
        from ..training.compiler import compile_lexicon_plan
        from ..training.linker import link_lexicon_topology

        holdout = holdout_words or set()
        batch = BatchProjector(cast("EmergentParser", self))

        for core_area in CORE_AREAS:
            if core_area not in self.core_lexicons:
                self.core_lexicons[core_area] = {}

        compiled_enabled = getattr(self, "_compiled_training_enabled", True)
        plan = compile_lexicon_plan(
            cast("EmergentParser", self),
            holdout_words=holdout,
            skip_known=skip_known,
            words=cast(Optional[Sequence[str]], words),
        )
        if not plan.lexicon_ops:
            return

        if compiled_enabled:
            link_lexicon_topology(cast("EmergentParser", self), plan)
            plan.topology = lexicon_topology_spec(
                cast("CompiledTopologyParser", self), plan.core_areas
            )
            use_compiled = plan.topology.ready
            prev_fidelity = self.brain.projection_fidelity
            if self.fast_training and use_compiled:
                self.brain.projection_fidelity = "compiled"
            try:
                with compiled_topology(
                    cast("CompiledTopologyParser", self), plan.topology
                ):
                    for op in plan.lexicon_ops:
                        ctx = self.word_grounding[op.word]
                        batch.apply_lexicon_word(
                            op.word, ctx, op.core_area, rounds=op.rounds,
                        )
                        self.core_lexicons[op.core_area][op.word] = _snap(
                            self.brain, op.core_area,
                        )
                        self._category_cache[op.word] = CORE_TO_CATEGORY[
                            op.core_area
                        ]
            finally:
                self.brain.projection_fidelity = prev_fidelity
            return

        pending: Dict[str, List[tuple]] = {c: [] for c in CORE_AREAS}
        for op in plan.lexicon_ops:
            ctx = self.word_grounding[op.word]
            pending[op.core_area].append((op.word, ctx))

        for core_area in CORE_AREAS:
            for word, ctx in pending[core_area]:
                batch.apply_lexicon_word(
                    word, ctx, core_area, rounds=self.rounds,
                )
                self.core_lexicons[core_area][word] = _snap(self.brain, core_area)
                self._category_cache[word] = CORE_TO_CATEGORY[core_area]


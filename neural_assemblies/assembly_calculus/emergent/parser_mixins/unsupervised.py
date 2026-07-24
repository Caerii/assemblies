"""UnsupervisedMixin -- unsupervised thematic role learning from raw exposure.

WHAT IS AND IS NOT UNSUPERVISED HERE.  "Unsupervised" means no per-sentence
role ANNOTATION: the training sentences arrive as word strings, and nothing
tells the model that "dog" is the agent of this sentence.  Roles are derived
instead from position relative to the verb, using the word order the parser
has inferred for the language -- see ``core.corpus_index._assign_noun_roles``,
which for SVO reads nouns before the verb as agent and nouns after it as
patient.  So the supervision that remains is STRUCTURAL (a word-order
typology, itself detected from the corpus) rather than semantic.

That is a real but bounded claim, and worth stating precisely because it is
easy to over- or under-sell.  The model is not discovering thematic roles from
nothing; it is discovering which words fill which role, given a positional
regularity it has extracted from the same corpus.  Sentences that violate the
typology (passives, scrambling) will be assigned the wrong role and trained on
it, silently.

The neural content is what happens after the assignment: each word's core
assembly is projected into its role area, so a role area accumulates the
superposition of every filler ever assigned to it, and individual bindings
survive only as the synapses from each word.  This is why
``_extract_role_lexicons_from_weights`` reads the lexicon back OUT of the
connectome rather than off a Python dict -- the stored representation is the
weights, and a snapshot taken during training would be contaminated by
whichever filler was most recently trained.
"""

from typing import Dict, List, Optional, TYPE_CHECKING

from neural_assemblies.assembly_calculus.ops import project

from ..core.areas import (
    ROLE_AGENT,
    ROLE_PATIENT,
    THEMATIC_AREAS,
)
from ..training.batch import BatchProjector
from ..training.compiled import compiled_topology, role_topology_spec
from ..training.compiler import compile_role_plan
from ..training.linker import link_role_topology

if TYPE_CHECKING:
    from ..core.corpus_index import CorpusIndex
    from ..curriculum.data import GroundedSentence

_ROLE_TRAINING_AREAS = (ROLE_AGENT, ROLE_PATIENT)


class UnsupervisedMixin:
    """Unsupervised thematic role learning from raw exposure."""

    def _clear_role_activity(self, role_area: str) -> None:
        """Clear role winners without wiping learned connectomes.

        Rewinding ``w`` to 0 is a ring-reuse device: under compiled training the
        role area holds a bounded pre-grown column pool and each binding replays
        into ring slots ``0..k`` without consuming new neuron IDs, so resetting
        ``w`` is how the next binding lands back at the start of the ring.  Under
        exact training there is no ring -- ``w`` is the count of neurons that
        have ever fired and backs both ``compact_to_neuron_id`` and the
        ``neuron_id_pool`` pointer.  Zeroing it there desyncs those three (the
        engine then treats every subsequent winner as first-time, re-recruiting
        fresh pool IDs on an area that is already full and eventually exhausting
        the pool), and it also throws away the learned role columns instead of
        reusing them.  So only rewind ``w`` when the ring path is actually in
        use; otherwise behave like ``_clear_prediction_activity`` -- inhibit and
        unfix, leaving the history intact for the next projection to reuse.
        """
        self.brain.inhibit_areas([role_area])
        if self.brain._engine.is_fixed(role_area):
            self.brain._engine.unfix_assembly(role_area)
        if getattr(self, "_compiled_training_enabled", True):
            area = self.brain.areas[role_area]
            area.w = 0
            engine = self.brain._engine
            if hasattr(engine, "_areas") and role_area in engine._areas:
                engine._areas[role_area].w = 0

    def _pregrow_role_pathways(self, corpus_index: "CorpusIndex") -> None:
        """Pre-expand core→role connectomes before Hebbian role training."""
        if getattr(self, "_role_paths_bootstrapped", False):
            return

        samples: Dict[tuple, str] = {}
        for update in corpus_index.role_updates:
            if update.word not in self.stim_map:
                continue
            core = self._word_core_area(update.word)
            key = (core, update.role_area)
            if key not in samples:
                samples[key] = update.word

        if not samples:
            return

        with self.brain.frozen():
            for (core_area, role_area), word in samples.items():
                phon = self.stim_map[word]
                project(self.brain, phon, core_area, rounds=self.rounds)
                self.brain.areas[core_area].fix_assembly()
                self.brain.project(
                    {},
                    {core_area: [role_area], role_area: [role_area]},
                )
                if self.rounds > 1:
                    self.brain.project_rounds(
                        target=role_area,
                        areas_by_stim={},
                        dst_areas_by_src_area={
                            core_area: [role_area],
                            role_area: [role_area],
                        },
                        rounds=self.rounds - 1,
                    )
                self.brain.areas[core_area].unfix_assembly()
                self._clear_role_activity(role_area)

        engine = self.brain._engine
        caps: Dict[str, int] = {}
        for role_area in _ROLE_TRAINING_AREAS:
            if hasattr(engine, "_areas") and role_area in engine._areas:
                caps[role_area] = max(int(engine._areas[role_area].w), self.k)
        self._role_ring_capacity_cols = caps
        self._role_paths_bootstrapped = True

    def train_unsupervised(
        self,
        sentences: List["GroundedSentence"],
        repetitions: int = 3,
        *,
        corpus_index: Optional["CorpusIndex"] = None,
        force_link: bool = False,
    ):
        """Learn role assignments from raw sentence exposure (no role labels).

        When ``corpus_index`` is provided, uses precompiled role updates
        instead of re-classifying every token each pass.
        """
        from ..core.corpus_index import compile_corpus
        from ..training.perf import adaptive_rounds

        for role_area in THEMATIC_AREAS:
            if role_area not in self.role_lexicons:
                self.role_lexicons[role_area] = {}

        if corpus_index is None:
            corpus_index = compile_corpus(self, sentences)

        compiled_enabled = getattr(self, "_compiled_training_enabled", True)
        batch = BatchProjector(self)

        if compiled_enabled:
            link_role_topology(
                self, corpus_index, force=force_link,
            )
            plan = compile_role_plan(self, corpus_index, repetitions)
            with compiled_topology(self, plan.topology):
                for op in plan.role_ops:
                    batch.apply_role_update(
                        op.word, op.role_area, rounds=op.rounds,
                    )
        else:
            word_freq = corpus_index.word_freq
            for _rep in range(repetitions):
                for update in corpus_index.role_updates:
                    if update.word not in self.stim_map:
                        continue
                    rounds = adaptive_rounds(
                        self.rounds, word_freq.get(update.word, 1),
                    )
                    batch.apply_role_update(
                        update.word, update.role_area, rounds=rounds,
                    )

        self._extract_role_lexicons_from_weights(
            corpus_vocab=corpus_index.content_words,
        )

    def _extract_role_lexicons_from_weights(
        self,
        corpus_vocab: Optional[set] = None,
    ):
        """Extract role lexicons for corpus content words (not full vocab)."""
        for role_area in [ROLE_AGENT, ROLE_PATIENT]:
            if role_area not in self.role_lexicons:
                self.role_lexicons[role_area] = {}

        candidates = corpus_vocab
        if candidates is None:
            candidates = {
                w for w, ctx in self.word_grounding.items()
                if ctx.dominant_modality in ("visual", "social")
                and w in self.stim_map
            }

        missing = [
            (word, role_area)
            for word in candidates
            for role_area in [ROLE_AGENT, ROLE_PATIENT]
            if (
                word in self.stim_map
                and word not in self.role_lexicons.get(role_area, {})
                and (ctx := self.word_grounding.get(word)) is not None
                and ctx.dominant_modality in ("visual", "social")
            )
        ]
        if not missing:
            return

        role_spec = role_topology_spec(self)
        batch = BatchProjector(self)
        compiled_enabled = getattr(self, "_compiled_training_enabled", True)

        def _extract_one(word: str, role_area: str) -> None:
            batch.apply_role_update(
                word, role_area, rounds=self.rounds, clear_role=True,
            )

        if compiled_enabled and role_spec.ready:
            with compiled_topology(self, role_spec):
                for word, role_area in missing:
                    _extract_one(word, role_area)
        else:
            for word, role_area in missing:
                _extract_one(word, role_area)

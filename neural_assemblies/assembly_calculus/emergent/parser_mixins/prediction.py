"""PredictionMixin -- structural next-token prediction using context assemblies.

The pipeline is three areas and two learned mappings::

    words --(one step per word)--> CONTEXT --(bridge)--> PREDICTION --> readout

CONTEXT accumulates the prefix: each word's core assembly is projected in with
recurrence, so the area's state is a function of everything seen so far.  The
CONTEXT -> PREDICTION bridge is trained by pairing each prefix's context
assembly with the assembly of the word that actually followed it, which is a
Hebbian analogue of a language model's next-token objective -- co-fire the
context and the answer, and the synapses between them carry the prediction.
At inference the answer half is absent, so driving PREDICTION from CONTEXT
alone reconstructs whatever that context has most often been followed by.

WHY THE LONGEST-COMMON-PREFIX BOOKKEEPING EXISTS.  Building the context for
"the dog chased the" and then for "the dog chased the cat" would repeat four
words of work.  Because CONTEXT is a running state, a prefix that EXTENDS the
current one can be reached by feeding only the new words
(``_build_prefix_context_incremental``).  A prefix that diverges cannot -- the
state has to be torn down and rebuilt from scratch, which is what the
``shared < len(current)`` branch does.  This is an optimisation, not a model
claim, but it is load-bearing for correctness: skipping the reset on a
divergent prefix would silently predict from a context that mixes two
different sentences.

KNOWN STRUCTURAL LIMIT.  One area of ``k`` winners cannot encode an unbounded
prefix, so CONTEXT saturates on longer inputs and successive prefixes start
mapping to near-identical assemblies.  That limit -- with the measured neuron
counts -- is documented in ``state_prediction.py``, which implements the
paper-faithful alternative that carries sequence in weights between bounded
states rather than in an accumulating buffer.  Both paths are kept; prefer
``StatePredictionMixin`` when prefix length matters.

Inference forces ``projection_fidelity = "exact"`` and restores the previous
setting afterwards.  "Compiled" fidelity is a topology-quantization shortcut
that skips candidate sampling and connectome expansion (see
``core.projection_fidelity``); it is accurate enough to TRAIN through, since
it is gated on readout overlap rather than step-by-step identity, but
prediction quality is the quantity being measured here and must not be read
through the approximation.
"""

from contextlib import nullcontext
from typing import List, Optional, Tuple, TYPE_CHECKING

from neural_assemblies.assembly_calculus.ops import project, _snap
from neural_assemblies.assembly_calculus.readout import readout_all

from ..core.areas import (
    CONTEXT,
    PREDICTION,
)
from ..training.compiled import compiled_topology
from ..training.compiler import (
    compile_training_plan,
    group_bridge_ops_by_prefix,
    lcp_length,
)
from ..training.linker import link_bridge_topology

if TYPE_CHECKING:
    from ..core.corpus_index import CorpusIndex, TransitionCache
    from ..curriculum.data import GroundedSentence


class PredictionMixin:
    """Structural next-token prediction using context assemblies."""

    _prediction_paths_bootstrapped: bool = False

    def _clear_prediction_activity(self) -> None:
        """Clear PREDICTION winners without wiping learned connectomes."""
        if PREDICTION in self.brain.areas:
            self.brain.inhibit_areas([PREDICTION])
            if self.brain.is_fixed(PREDICTION):
                self.brain.unfix_assembly(PREDICTION)

    def _bootstrap_prediction_connectivity(self) -> None:
        """Materialize baseline CONTEXT/PREDICTION pathways once (plasticity off)."""
        if self._prediction_paths_bootstrapped or not self.stim_map:
            return

        brain = self.brain
        arb_phon = next(iter(self.stim_map.values()))
        with brain.frozen():
            brain.project({arb_phon: [PREDICTION]}, {})
            brain.project({arb_phon: [CONTEXT]}, {})
            brain.project(
                {arb_phon: [PREDICTION]},
                {CONTEXT: [PREDICTION]},
            )
            brain.project({}, {PREDICTION: [PREDICTION]})

        brain.inhibit_areas([PREDICTION])
        self._reset_context_state()
        self._prediction_paths_bootstrapped = True

    def _ensure_prediction_lexicon(
        self,
        words: Optional[List[str]] = None,
    ) -> None:
        """Build prediction lexicon entries only for missing words."""
        targets = words if words is not None else list(self.stim_map.keys())
        if not getattr(self, "_compiled_training_enabled", True):
            if not hasattr(self, "prediction_lexicon"):
                self.prediction_lexicon = {}
            self._bootstrap_prediction_connectivity()
            for word in targets:
                if word in self.prediction_lexicon:
                    continue
                phon = self.stim_map.get(word)
                if phon is None:
                    continue
                self._clear_prediction_activity()
                project(self.brain, phon, PREDICTION, rounds=self.rounds)
                self.prediction_lexicon[word] = _snap(self.brain, PREDICTION)
            return

        from ..training.linker import link_prediction_lexicon

        link_prediction_lexicon(self, targets)

    def _train_next_token_bridge(
        self,
        next_phon: str,
        *,
        bridge_rounds: int,
    ) -> None:
        """Apply one Hebbian CONTEXT + phon -> PREDICTION bridge step."""
        self._clear_prediction_activity()
        self.brain.project(
            {next_phon: [PREDICTION]},
            {CONTEXT: [PREDICTION]},
        )
        if bridge_rounds > 1:
            self.brain.project_rounds(
                target=PREDICTION,
                areas_by_stim={next_phon: [PREDICTION]},
                dst_areas_by_src_area={
                    CONTEXT: [PREDICTION],
                    PREDICTION: [PREDICTION],
                },
                rounds=bridge_rounds - 1,
            )

    def _build_prefix_context_incremental(
        self,
        prefix: Tuple[str, ...],
        current: Tuple[str, ...],
        *,
        use_direct_context: bool,
        preserve_topology: bool,
    ) -> Tuple[str, ...]:
        """Advance CONTEXT from *current* to *prefix* via LCP + delta words."""
        if not prefix:
            return current

        if not use_direct_context:
            raise NotImplementedError("fiber context path requires sentence loop")

        shared = lcp_length(current, prefix)
        if shared < len(current):
            self._reset_context_for_bridge(preserve_topology=preserve_topology)
            for w in prefix:
                self._advance_context_direct(
                    w, rounds=self.inference_rounds,
                )
        else:
            for w in prefix[shared:]:
                self._advance_context_direct(
                    w, rounds=self.inference_rounds,
                )
        return prefix

    def train_next_token(
        self,
        sentences: List["GroundedSentence"],
        *,
        rebuild_lexicon: bool = False,
        dedupe_sentences: bool = True,
        use_direct_context: bool = True,
        corpus_index: Optional["CorpusIndex"] = None,
        transition_cache: Optional["TransitionCache"] = None,
        force_link: bool = False,
    ):
        """Train next-token prediction using compiled bridge program."""
        from ..core.corpus_index import compile_corpus

        if rebuild_lexicon:
            self.prediction_lexicon = {}
            self._bridge_topology_linked = False

        if corpus_index is None:
            if dedupe_sentences:
                from ..training.perf import dedupe_grounded_sentences
                sentences = dedupe_grounded_sentences(sentences, self.stim_map)
            corpus_index = compile_corpus(self, sentences)

        transitions = corpus_index.transitions
        if transition_cache is not None:
            transitions = transition_cache.filter_untrained(transitions)
        if not transitions:
            return

        bridge_vocab = corpus_index.bridge_vocab & set(self.stim_map.keys())
        # sorted(), not list(): `bridge_vocab` is a SET INTERSECTION, so its
        # iteration order is randomized per process (PEP 456), and
        # `lex_targets` decides the order the prediction lexicon is BUILT --
        # which projects, which recruits. Second of the two sites that made
        # training irreproducible across processes (#80); the other is
        # `acquisition/pos_inference.infer_holdout_categories`.
        #
        # The `else` branch is a dict, which is insertion-ordered and already
        # deterministic, so it is deliberately left alone.
        lex_targets = (
            sorted(bridge_vocab)
            if bridge_vocab
            else list(self.stim_map.keys())
        )

        compiled_enabled = getattr(self, "_compiled_training_enabled", True)

        if compiled_enabled:
            link_bridge_topology(
                self, corpus_index, lex_targets, force=force_link,
            )
        else:
            self._ensure_prediction_lexicon(lex_targets)

        plan = compile_training_plan(self, corpus_index, transitions=transitions)
        by_prefix = group_bridge_ops_by_prefix(plan.bridge_ops)
        prefix_order = plan.prefix_order or sorted(
            by_prefix.keys(), key=lambda p: (len(p), p),
        )

        use_compiled = (
            compiled_enabled
            and self.fast_training
            and len(plan.topology.ring_capacity) >= 2
        )
        prev_fidelity = self.brain.projection_fidelity
        if self.fast_training and use_compiled:
            self.brain.projection_fidelity = "compiled"

        current_prefix: Tuple[str, ...] = ()

        try:
            # The compiled-topology session freezes connectome growth and
            # ring-reuses a bounded pre-grown column pool. Under norm_init that
            # collapses the CONTEXT/PREDICTION bridge to a context-independent
            # attractor (see _compiled_training_enabled in core.py), so it must
            # be entered ONLY when compiled is actually in use -- gating the flag
            # alone is not enough because this session runs even when
            # use_compiled is False and its freeze+ring setup is itself the
            # collapse mechanism.
            topo_session = (
                compiled_topology(self, plan.topology)
                if use_compiled else nullcontext()
            )
            with topo_session:
                for prefix in prefix_order:
                    if prefix not in by_prefix:
                        continue

                    with self.brain.frozen():
                        current_prefix = self._build_prefix_context_incremental(
                            prefix,
                            current_prefix,
                            use_direct_context=use_direct_context,
                            preserve_topology=self._context_compiled_active(),
                        )

                    for op in by_prefix[prefix]:
                        next_phon = self.stim_map.get(op.next_word)
                        if next_phon is None:
                            continue
                        self._train_next_token_bridge(
                            next_phon, bridge_rounds=op.bridge_rounds,
                        )
        finally:
            self.brain.projection_fidelity = prev_fidelity

        if transition_cache is not None:
            transition_cache.mark_trained(transitions)

    def predict_next(self, words: List[str]) -> List[Tuple[str, float]]:
        """Predict the next word given a partial sentence."""
        if not words:
            return []

        if not hasattr(self, 'prediction_lexicon') or not self.prediction_lexicon:
            return []

        prev_fidelity = self.brain.projection_fidelity
        self.brain.projection_fidelity = "exact"
        try:
            self._bootstrap_prediction_connectivity()
            self.build_context_incremental(words, reset=True, direct=True)

            infer_rounds = self.inference_rounds
            self._clear_prediction_activity()
            self.brain.project(
                {},
                {CONTEXT: [PREDICTION]},
            )
            if infer_rounds > 1:
                self.brain.project_rounds(
                    target=PREDICTION,
                    areas_by_stim={},
                    dst_areas_by_src_area={
                        CONTEXT: [PREDICTION],
                        PREDICTION: [PREDICTION],
                    },
                    rounds=infer_rounds - 1,
                )

            pred_assembly = _snap(self.brain, PREDICTION)
            return readout_all(pred_assembly, self.prediction_lexicon)
        finally:
            self.brain.projection_fidelity = prev_fidelity

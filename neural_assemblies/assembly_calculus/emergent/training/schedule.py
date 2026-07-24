"""Training schedule executor — runs curriculum phases from a CorpusIndex."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from ..core.corpus_index import (
    CorpusIndex,
    TransitionCache,
    compile_corpus,
    grounded_fraction,
    merge_corpus_indices,
)
from ..core.grounding import GroundingContext
from ..curriculum.data import GroundedSentence
from ..training.perf import stage_distributional_reps

if TYPE_CHECKING:
    from ..parser import EmergentParser


@dataclass
class PhaseResult:
    phase: str
    detail: str = ""


@dataclass
class TrainingSchedule:
    """Ordered training phases backed by a shared corpus index."""
    corpus_index: CorpusIndex
    phases: List[str]
    stage_name: str = ""
    distributional_reps: int = 1
    word_order_reps: int = 0
    prediction_index: Optional[CorpusIndex] = None
    conversation_index: Optional[CorpusIndex] = None
    dialogue_pairs: Optional[list] = None
    transition_cache: Optional[TransitionCache] = None
    consolidation_passes: int = 0
    holdout_words: Optional[set] = None
    results: List[PhaseResult] = field(default_factory=list)


class TrainingScheduleExecutor:
    """Execute training phases using precompiled corpus data."""

    GROUNDED_DIST_THRESHOLD = 0.65

    def __init__(self, parser: "EmergentParser"):
        self.parser = parser

    def run(self, schedule: TrainingSchedule) -> List[PhaseResult]:
        from ..train_progress import current_progress

        prog = current_progress()
        idx = schedule.corpus_index
        p = self.parser
        run: List[PhaseResult] = []

        if "lexicon" in schedule.phases:
            with prog.phase("lexicon"):
                p.train_lexicon(
                    skip_known=True,
                    words=idx.corpus_vocab,
                    holdout_words=schedule.holdout_words,
                )
            run.append(PhaseResult("lexicon"))

        if "distributional" in schedule.phases:
            reps = schedule.distributional_reps
            if schedule.holdout_words:
                from ..acquisition.pos_inference import (
                    infer_holdout_categories,
                    ingest_holdout_sentence_stats,
                )

                n_sents = ingest_holdout_sentence_stats(
                    p, schedule.holdout_words,
                )
                if n_sents:
                    prog.info(
                        f"holdout stats ingested from {n_sents} canonical sentences",
                    )
            gf = grounded_fraction(p, idx.corpus_vocab)
            with prog.phase("distributional", f"{len(idx.raw)} sents x{reps}"):
                if gf >= self.GROUNDED_DIST_THRESHOLD:
                    p.train_distributional_from_index(idx, repetitions=reps)
                else:
                    p.train_distributional(idx.raw, repetitions=reps)
            if schedule.holdout_words:
                from ..acquisition.pos_inference import infer_holdout_categories

                infer_holdout_categories(p, schedule.holdout_words)
            run.append(PhaseResult("distributional", f"x{reps}"))

        if "roles" in schedule.phases:
            with prog.phase("roles", f"{len(idx.role_updates)} updates"):
                p.train_unsupervised(
                    idx.grounded,
                    repetitions=schedule.distributional_reps,
                    corpus_index=idx,
                )
            run.append(PhaseResult("roles"))

        if schedule.consolidation_passes > 0 and idx.grounded:
            from ..training.consolidation import (
                consolidate_role_pathways,
                consolidate_vp_pathways,
            )

            passes = schedule.consolidation_passes
            detail_parts = []
            with prog.phase("consolidation", f"x{passes}"):
                if "roles" in schedule.phases or idx.role_updates:
                    consolidate_role_pathways(p, idx.grounded, passes=passes)
                    detail_parts.append("role")
                if "phrases" in schedule.phases:
                    consolidate_vp_pathways(p, idx.grounded, passes=passes)
                    detail_parts.append("vp")
            if detail_parts:
                run.append(PhaseResult("consolidation", "+".join(detail_parts)))

        if "phrases" in schedule.phases:
            with prog.phase("phrases"):
                p.train_phrases(idx.grounded)
            run.append(PhaseResult("phrases"))

        if "word_order" in schedule.phases:
            wo = schedule.word_order_reps
            with prog.phase("word_order", f"typological + seq x{wo}"):
                p.train_word_order_typological(idx.raw)
                if wo > 0:
                    p.train_word_order(idx.grounded, repetitions=wo)
            run.append(PhaseResult("word_order"))

        for morph in ("tense", "mood", "polarity", "conjunctions"):
            if morph in schedule.phases:
                with prog.phase(morph):
                    getattr(p, f"train_{morph}")(idx.raw)
                run.append(PhaseResult(morph))

        if "number" in schedule.phases:
            with prog.phase("number"):
                p.train_number(idx.raw)
            run.append(PhaseResult("number"))

        if "prediction" in schedule.phases:
            pred_idx = schedule.prediction_index or idx
            with prog.phase(
                "prediction",
                f"{len(pred_idx.transitions)} unique bridges",
            ):
                holdout_pred = bool(schedule.holdout_words)
                old_bridge = p.bridge_rounds
                if holdout_pred:
                    p.bridge_rounds = max(old_bridge * 2, 6)
                try:
                    p.train_next_token(
                        pred_idx.grounded,
                        corpus_index=pred_idx,
                        transition_cache=(
                            None if holdout_pred else schedule.transition_cache
                        ),
                        dedupe_sentences=holdout_pred,
                        force_link=holdout_pred,
                    )
                    if schedule.holdout_words:
                        from ..curriculum.holdout_bridges import (
                            train_holdout_bridge_boost,
                        )

                        n_boost = train_holdout_bridge_boost(
                            p,
                            schedule.holdout_words,
                            stage_name=schedule.stage_name or "SENTENCES",
                        )
                        if n_boost:
                            prog.info(
                                f"holdout bridge boost: {n_boost} transitions",
                            )
                finally:
                    p.bridge_rounds = old_bridge
            run.append(PhaseResult("prediction"))

        if "dialogue" in schedule.phases and schedule.dialogue_pairs:
            pairs = schedule.dialogue_pairs
            with prog.phase("dialogue", f"{len(pairs)} Q-A pairs"):
                p.train_dialogue(
                    pairs,
                    transition_cache=schedule.transition_cache,
                )
            run.append(PhaseResult("dialogue"))

        if "conversation" in schedule.phases:
            from ..curriculum.conversation import train_conversation_exposure

            conv_idx = schedule.conversation_index
            conv = conv_idx.grounded if conv_idx else []
            with prog.phase("conversation", f"{len(conv)} turns"):
                if conv_idx is not None:
                    from ..core.corpus_index import ingest_index_stats
                    ingest_index_stats(p, conv_idx)
                else:
                    for sent in conv:
                        p.ingest_raw_sentence(sent.words)
                if conv_idx is not None:
                    p.train_next_token(
                        conv,
                        corpus_index=conv_idx,
                        transition_cache=schedule.transition_cache,
                    )
                elif conv:
                    p.train_next_token(
                        conv,
                        transition_cache=schedule.transition_cache,
                    )
                train_conversation_exposure(p)
            run.append(PhaseResult("conversation"))

        schedule.results = run
        return run

    @classmethod
    def build_stage_schedule(
        cls,
        parser: "EmergentParser",
        stage_name: str,
        raw_sentences: List[List[str]],
        phases: List[str],
        *,
        extra_prediction: Optional[List[GroundedSentence]] = None,
        conversation_sents: Optional[List[GroundedSentence]] = None,
        transition_cache: Optional[TransitionCache] = None,
    ) -> TrainingSchedule:
        """Compile sentences once and build a stage training schedule."""
        from ..training.perf import STAGE_WORD_ORDER_REPS, stage_consolidation_passes

        grounded = [
            GroundedSentence(
                words=sent,
                contexts=[
                    parser.word_grounding.get(w, GroundingContext())
                    for w in sent
                ],
                roles=[None] * len(sent),
            )
            for sent in raw_sentences
        ]
        idx = compile_corpus(parser, grounded)

        prediction_index = None
        if "prediction" in phases:
            merge_parts = [idx]
            if extra_prediction:
                merge_parts.append(compile_corpus(parser, extra_prediction))
            if transition_cache is not None:
                prediction_index = transition_cache.merge_indices(*merge_parts)
            elif len(merge_parts) > 1:
                prediction_index = merge_corpus_indices(*merge_parts)

        conversation_index = None
        if "conversation" in phases and conversation_sents:
            conversation_index = compile_corpus(parser, conversation_sents)

        return TrainingSchedule(
            corpus_index=idx,
            phases=phases,
            stage_name=stage_name,
            distributional_reps=stage_distributional_reps(
                stage_name, fast=parser.fast_training,
            ),
            word_order_reps=STAGE_WORD_ORDER_REPS.get(stage_name, 1)
            if "word_order" in phases else 0,
            consolidation_passes=stage_consolidation_passes(
                stage_name, fast=parser.fast_training,
            ),
            prediction_index=prediction_index,
            conversation_index=conversation_index,
            transition_cache=transition_cache,
        )

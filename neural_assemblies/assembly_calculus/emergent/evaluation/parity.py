"""Readout parity probes and dual-metric learnability gates."""

from __future__ import annotations

from contextlib import contextmanager
from typing import (
    Dict, Iterator, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING,
)

if TYPE_CHECKING:
    from ..core.corpus_index import CorpusIndex
    from ..parser import EmergentParser
    from ..training.schedule import TrainingSchedule


def collect_transition_probes(
    corpus_index: "CorpusIndex",
    *,
    max_probes: Optional[int] = 40,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Build (prefix, acceptable_next_words) probes from corpus transitions."""
    prefixes: List[List[str]] = []
    expected: List[List[str]] = []
    seen: set = set()

    for trans in corpus_index.transitions:
        key = (trans.prefix_words, trans.next_word)
        if key in seen:
            continue
        seen.add(key)
        prefixes.append(list(trans.prefix_words))
        expected.append([trans.next_word])

    if max_probes is not None and len(prefixes) > max_probes:
        prefixes = prefixes[:max_probes]
        expected = expected[:max_probes]

    return prefixes, expected


def score_holdout_constrained_probes(
    parser,
    prefixes: Sequence[Sequence[str]],
    expected_any: Sequence[Sequence[str]],
    holdout_words: Set[str],
    *,
    distractor_words: Optional[Set[str]] = None,
    seed: int = 42,
) -> Dict[str, float]:
    """OOV bridge probes — readout over holdouts plus same-corpus distractors."""
    import random

    from neural_assemblies.assembly_calculus.ops import _snap
    from neural_assemblies.assembly_calculus.readout import readout_all

    from ..core.areas import CONTEXT, PREDICTION

    lex = getattr(parser, "prediction_lexicon", None) or {}
    holdout = sorted(set(holdout_words) & set(lex.keys()))
    if not holdout:
        return {
            "top1": 0.0, "top3": 0.0, "top5": 0.0,
            "total": len(prefixes), "hits_top1": 0,
        }

    pool = set(holdout)
    if distractor_words:
        pool |= set(distractor_words) & set(lex.keys())
    else:
        rng = random.Random(seed)
        extras = [w for w in lex if w not in holdout_words]
        rng.shuffle(extras)
        pool |= set(extras[: min(15, len(extras))])

    candidate_lex = {w: lex[w] for w in sorted(pool)}
    top1 = top3 = top5 = 0
    total = len(prefixes)
    if total == 0:
        return {
            "top1": 0.0, "top3": 0.0, "top5": 0.0,
            "total": 0, "hits_top1": 0,
        }

    prev_fidelity = parser.brain.projection_fidelity
    parser.brain.projection_fidelity = "exact"
    try:
        parser._bootstrap_prediction_connectivity()
        for prefix, exp in zip(prefixes, expected_any, strict=True):
            if not prefix:
                continue
            parser.build_context_incremental(list(prefix), reset=True, direct=True)
            infer_rounds = parser.inference_rounds
            parser._clear_prediction_activity()
            parser.brain.project({}, {CONTEXT: [PREDICTION]})
            if infer_rounds > 1:
                parser.brain.project_rounds(
                    target=PREDICTION,
                    areas_by_stim={},
                    dst_areas_by_src_area={
                        CONTEXT: [PREDICTION],
                        PREDICTION: [PREDICTION],
                    },
                    rounds=infer_rounds - 1,
                )
            pred_assembly = _snap(parser.brain, PREDICTION)
            preds = readout_all(pred_assembly, candidate_lex)
            if not preds:
                continue
            words = [w for w, _ in preds]
            exp_set = set(exp)
            if words and words[0] in exp_set:
                top1 += 1
            if exp_set & set(words[:3]):
                top3 += 1
            if exp_set & set(words[:5]):
                top5 += 1
    finally:
        parser.brain.projection_fidelity = prev_fidelity

    return {
        "top1": top1 / total,
        "top3": top3 / total,
        "top5": top5 / total,
        "total": total,
        "hits_top1": top1,
    }


def score_next_token_probes(
    parser,
    prefixes: Sequence[Sequence[str]],
    expected_any: Sequence[Sequence[str]],
) -> Dict[str, float]:
    """Top-1 / top-3 / top-5 hit rate on prefix probes."""
    top1 = top3 = top5 = 0
    total = len(prefixes)
    if total == 0 or not hasattr(parser, "predict_next"):
        return {
            "top1": 0.0, "top3": 0.0, "top5": 0.0,
            "total": 0, "hits_top1": 0,
        }

    for prefix, exp in zip(prefixes, expected_any, strict=True):
        preds = parser.predict_next(list(prefix))
        if not preds:
            continue
        words = [w for w, _ in preds]
        exp_set = set(exp)
        if words and words[0] in exp_set:
            top1 += 1
        if exp_set & set(words[:3]):
            top3 += 1
        if exp_set & set(words[:5]):
            top5 += 1

    return {
        "top1": top1 / total,
        "top3": top3 / total,
        "top5": top5 / total,
        "total": total,
        "hits_top1": top1,
    }


def compare_predict_next_parity(
    parser_a,
    parser_b,
    prefixes: Sequence[Sequence[str]],
) -> Dict[str, float]:
    """Fraction of probes where two parsers agree on top-1 prediction."""
    agree = 0
    total = 0
    for prefix in prefixes:
        pa = parser_a.predict_next(list(prefix))
        pb = parser_b.predict_next(list(prefix))
        if not pa or not pb:
            continue
        total += 1
        if pa[0][0] == pb[0][0]:
            agree += 1
    return {
        "top1_agreement": agree / max(total, 1),
        "total": total,
        "agree": agree,
    }


def evaluate_corpus_parity(
    parser,
    corpus_index: "CorpusIndex",
    *,
    max_probes: Optional[int] = 40,
) -> Dict[str, float]:
    """Score parser on transition-derived next-token probes."""
    prefixes, expected = collect_transition_probes(
        corpus_index, max_probes=max_probes,
    )
    return score_next_token_probes(parser, prefixes, expected)


@contextmanager
def exact_training_mode() -> Iterator[None]:
    """Disable compiled topology sessions for microscopic baseline training."""
    from contextlib import nullcontext

    import neural_assemblies.assembly_calculus.emergent.parser_mixins.prediction as pred_mod
    import neural_assemblies.assembly_calculus.emergent.parser_mixins.unsupervised as unsup_mod

    saved = {
        "pred": pred_mod.compiled_topology,
        "unsup": unsup_mod.compiled_topology,
    }
    noop = lambda _p, _s: nullcontext()  # noqa: E731
    pred_mod.compiled_topology = noop  # type: ignore[assignment]
    unsup_mod.compiled_topology = noop  # type: ignore[assignment]
    try:
        yield
    finally:
        pred_mod.compiled_topology = saved["pred"]
        unsup_mod.compiled_topology = saved["unsup"]


def build_dialogue_stage_schedule(
    parser: "EmergentParser",
    *,
    seed: int = 42,
) -> "TrainingSchedule":
    """Build a DIALOGUE-stage training schedule (preset vocab path)."""
    from ..curriculum import CurriculumTrainer, _STAGE_CONFIG
    from ..curriculum.conversation import (
        get_conversation_curriculum,
        get_conversation_pairs,
    )
    from ..curriculum.dialogue import get_dialogue_pairs
    from ..curriculum.data import create_instruction_sentences
    from ..training.perf import effective_stage_phases, stage_training_rounds
    from ..training.schedule import TrainingScheduleExecutor

    stage_name = "DIALOGUE"
    trainer = CurriculumTrainer(parser)
    config = _STAGE_CONFIG[stage_name]
    phases = effective_stage_phases(
        stage_name, list(config["phases"]), fast=parser.fast_training,
    )

    stage_words = trainer._get_stage_words(stage_name)
    for w in stage_words:
        parser.register_word(w.lemma)
    sentences = trainer.generation.generate(stage_words, config["complexity"])
    # REQUIRED here too, and it was missing: the generator emits finite forms
    # ("builds"), `compile_corpus` skips any token absent from `stim_map`, so
    # without this every verb in this stage is silently dropped -- the same
    # dead-path shape `_register_surface_forms` was written to close on the
    # curriculum path. One registration function, called by both.
    trainer.generation.register_surface_forms(sentences, stage_words)

    rounds_override = stage_training_rounds(stage_name, fast=parser.fast_training)
    if rounds_override is not None:
        parser.rounds = rounds_override
    trainer._set_global_beta(config["beta"])

    schedule = TrainingScheduleExecutor.build_stage_schedule(
        parser,
        stage_name,
        sentences,
        phases,
        extra_prediction=(
            create_instruction_sentences() if "prediction" in phases else None
        ),
        conversation_sents=(
            get_conversation_curriculum(parser.word_grounding)
            if "conversation" in phases else None
        ),
        transition_cache=trainer._transition_cache,
    )
    if "dialogue" in phases:
        schedule.dialogue_pairs = (
            get_dialogue_pairs(seed=seed)
            + get_conversation_pairs(parser.word_grounding)
        )
    return schedule


def run_dialogue_stage(
    parser: "EmergentParser",
    schedule: "TrainingSchedule",
) -> None:
    """Execute all phases in a DIALOGUE-stage schedule."""
    from ..training.schedule import TrainingScheduleExecutor

    TrainingScheduleExecutor(parser).run(schedule)


def evaluate_learnability_metrics(
    parser: "EmergentParser",
    *,
    corpus_index: Optional["CorpusIndex"] = None,
    qa_pairs: Optional[list] = None,
    max_probes: int = 25,
    seed: int = 42,
) -> Dict[str, float]:
    """Dialogue accuracy + next-token top-5 on a trained parser."""
    from ..core.corpus_index import compile_corpus
    from ..curriculum.dialogue import get_dialogue_pairs
    from .suite import EvaluationSuite
    from ..curriculum.data import create_training_sentences

    suite = EvaluationSuite(parser)
    if qa_pairs is None:
        qa_pairs = get_dialogue_pairs(seed=seed)

    dialogue = suite.evaluate_dialogue(
        [
            {
                "question": " ".join(p.question.words),
                "acceptable": p.answer.words[:3],
                "pattern_type": p.pattern_type,
            }
            for p in qa_pairs
        ],
    )

    if corpus_index is None:
        sents = create_training_sentences()
        corpus_index = compile_corpus(parser, sents)

    next_token = evaluate_corpus_parity(
        parser, corpus_index, max_probes=max_probes,
    )
    return {
        "dialogue_accuracy": dialogue["accuracy"],
        "dialogue_total": dialogue["total"],
        "next_token_top5": next_token["top5"],
        "next_token_top1": next_token["top1"],
        "next_token_total": next_token["total"],
    }


def compare_learnability_parity(
    exact_parser: "EmergentParser",
    compiled_parser: "EmergentParser",
    *,
    corpus_index: Optional["CorpusIndex"] = None,
    qa_pairs: Optional[list] = None,
    max_probes: int = 25,
    seed: int = 42,
    micro_parity_prefixes: Optional[Sequence[Sequence[str]]] = None,
) -> Dict[str, object]:
    """Learnability gate: task metrics + optional micro bridge parity."""
    from ..core.corpus_index import compile_corpus
    from ..curriculum.data import create_training_sentences

    exact_metrics = evaluate_learnability_metrics(
        exact_parser,
        corpus_index=corpus_index,
        qa_pairs=qa_pairs,
        max_probes=max_probes,
        seed=seed,
    )
    compiled_metrics = evaluate_learnability_metrics(
        compiled_parser,
        corpus_index=corpus_index,
        qa_pairs=qa_pairs,
        max_probes=max_probes,
        seed=seed,
    )

    if corpus_index is None:
        corpus_index = compile_corpus(
            exact_parser, create_training_sentences(),
        )
    prefixes, _ = collect_transition_probes(corpus_index, max_probes=max_probes)
    if micro_parity_prefixes is not None:
        prefixes = list(micro_parity_prefixes)

    optimizer = compare_predict_next_parity(
        exact_parser, compiled_parser, prefixes,
    )

    return {
        "exact": exact_metrics,
        "compiled": compiled_metrics,
        "optimizer_parity": optimizer,
        "dialogue_delta": (
            compiled_metrics["dialogue_accuracy"]
            - exact_metrics["dialogue_accuracy"]
        ),
        "next_token_top5_delta": (
            compiled_metrics["next_token_top5"]
            - exact_metrics["next_token_top5"]
        ),
        "next_token_top1_delta": (
            compiled_metrics["next_token_top1"]
            - exact_metrics["next_token_top1"]
        ),
    }


def run_dual_metric_gate(
    *,
    n: int = 3000,
    k: int = 30,
    seed: int = 42,
    max_probes: int = 25,
    qa_subset: Optional[int] = 15,
) -> Dict[str, object]:
    """Train exact vs compiled DIALOGUE stages and return dual-metric report."""
    from ..vocabulary_builder import build_vocabulary_preset
    from ..curriculum.dialogue import get_dialogue_pairs
    from ..parser import EmergentParser

    vocab = build_vocabulary_preset("medium")
    qa_pairs = get_dialogue_pairs(seed=seed)
    if qa_subset is not None:
        qa_pairs = qa_pairs[:qa_subset]

    def _train(*, exact: bool) -> EmergentParser:
        parser = EmergentParser(
            n=n, k=k, seed=seed, vocabulary=vocab, fast_training=True,
        )
        schedule = build_dialogue_stage_schedule(parser, seed=seed)
        if exact:
            parser._compiled_training_enabled = False
            parser.brain.projection_fidelity = "exact"
            with exact_training_mode():
                run_dialogue_stage(parser, schedule)
        else:
            run_dialogue_stage(parser, schedule)
        return parser

    exact = _train(exact=True)
    compiled = _train(exact=False)
    return compare_learnability_parity(
        exact,
        compiled,
        qa_pairs=qa_pairs,
        max_probes=max_probes,
        seed=seed,
    )

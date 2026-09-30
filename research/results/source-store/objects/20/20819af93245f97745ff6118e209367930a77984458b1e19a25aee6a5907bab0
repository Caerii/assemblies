#!/usr/bin/env python3
"""DIALOGUE medium benchmark — phase decomposition + algorithmic cost model."""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import time
from collections import defaultdict
from typing import Dict, List, Tuple

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ["TRAIN_PROGRESS"] = "0"

from neural_assemblies.assembly_calculus.emergent import (
    EmergentParser,
    build_vocabulary_preset,
)
from neural_assemblies.assembly_calculus.emergent.curriculum import (
    CurriculumTrainer,
    _STAGE_CONFIG,
)
from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
    create_instruction_sentences,
)
from neural_assemblies.assembly_calculus.emergent.training.perf import (
    effective_stage_phases,
    stage_distributional_reps,
    stage_training_rounds,
)
from neural_assemblies.assembly_calculus.emergent.training.schedule import (
    TrainingScheduleExecutor,
)


def _count_ops(parser, schedule) -> Dict[str, int]:
    idx = schedule.corpus_index
    pred = schedule.prediction_index or idx
    return {
        "stage_vocab": len(idx.corpus_vocab),
        "stage_sentences": len(idx.raw),
        "role_updates": len(idx.role_updates),
        "role_updates_x_reps": len(idx.role_updates)
        * stage_distributional_reps(schedule.stage_name, fast=parser.fast_training),
        "transitions_stage": len(idx.transitions),
        "transitions_prediction": len(pred.transitions),
        "unique_prefixes": len({t.prefix_words for t in pred.transitions}),
        "bridge_steps": len(pred.transitions),
        "dialogue_pairs": len(schedule.dialogue_pairs or []),
        "conversation_turns": len(
            schedule.conversation_index.grounded
            if schedule.conversation_index else []
        ),
    }


def bench_dialogue_stage_only(
    preset: str = "medium",
    *,
    seed: int = 42,
    n: int = 3000,
    k: int = 30,
) -> Tuple[float, Dict[str, float], Dict[str, int], EmergentParser]:
    """Time DIALOGUE stage with per-phase wall clock for a vocab preset."""
    vocab = build_vocabulary_preset(preset)
    parser = EmergentParser(
        n=n, k=k, seed=seed, vocabulary=vocab, fast_training=True,
    )
    trainer = CurriculumTrainer(parser)

    stage_name = "DIALOGUE"
    config = _STAGE_CONFIG[stage_name]
    beta = config["beta"]
    complexity = config["complexity"]
    phases = effective_stage_phases(
        stage_name, list(config["phases"]), fast=parser.fast_training,
    )

    stage_words = trainer._get_stage_words(stage_name)
    for w in stage_words:
        parser.register_word(w.lemma)
    sentences = trainer.generation.generate(stage_words, complexity)

    rounds_override = stage_training_rounds(stage_name, fast=parser.fast_training)
    old_rounds = parser.rounds
    if rounds_override is not None:
        parser.rounds = rounds_override

    trainer._set_global_beta(beta)

    from neural_assemblies.assembly_calculus.emergent.curriculum.conversation import (
        get_conversation_curriculum,
    )
    from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
        create_instruction_sentences as instr,
    )

    schedule = TrainingScheduleExecutor.build_stage_schedule(
        parser, stage_name, sentences, phases,
        extra_prediction=instr() if "prediction" in phases else None,
        conversation_sents=(
            get_conversation_curriculum(parser.word_grounding)
            if "conversation" in phases else None
        ),
        transition_cache=trainer._transition_cache,
    )

    if "dialogue" in phases:
        from neural_assemblies.assembly_calculus.emergent.curriculum.dialogue import (
            get_dialogue_pairs,
        )
        from neural_assemblies.assembly_calculus.emergent.curriculum.conversation import (
            get_conversation_pairs,
        )

        schedule.dialogue_pairs = (
            get_dialogue_pairs() + get_conversation_pairs(parser.word_grounding)
        )

    ops = _count_ops(parser, schedule)

    phase_times: Dict[str, float] = {}
    executor = TrainingScheduleExecutor(parser)

    # Instrument executor by running phases manually with timers
    from neural_assemblies.assembly_calculus.emergent.train_progress import (
        current_progress,
    )
    from neural_assemblies.assembly_calculus.emergent.core.corpus_index import (
        grounded_fraction,
    )

    prog = current_progress()
    idx = schedule.corpus_index
    p = parser

    t_total = time.perf_counter()
    for phase in schedule.phases:
        t0 = time.perf_counter()
        if phase == "lexicon":
            p.train_lexicon(skip_known=True, words=idx.corpus_vocab)
        elif phase == "distributional":
            reps = schedule.distributional_reps
            gf = grounded_fraction(p, idx.corpus_vocab)
            if gf >= TrainingScheduleExecutor.GROUNDED_DIST_THRESHOLD:
                p.train_distributional_from_index(idx, repetitions=reps)
            else:
                p.train_distributional(idx.raw, repetitions=reps)
        elif phase == "roles":
            p.train_unsupervised(
                idx.grounded, repetitions=schedule.distributional_reps,
                corpus_index=idx,
            )
        elif phase == "phrases":
            p.train_phrases(idx.grounded)
        elif phase == "word_order":
            p.train_word_order_typological(idx.raw)
            if schedule.word_order_reps > 0:
                p.train_word_order(idx.grounded, repetitions=schedule.word_order_reps)
        elif phase in ("tense", "mood", "polarity", "conjunctions"):
            getattr(p, f"train_{phase}")(idx.raw)
        elif phase == "number":
            p.train_number(idx.raw)
        elif phase == "prediction":
            pred_idx = schedule.prediction_index or idx
            p.train_next_token(pred_idx.grounded, corpus_index=pred_idx)
        elif phase == "dialogue" and schedule.dialogue_pairs:
            p.train_dialogue(schedule.dialogue_pairs)
        elif phase == "conversation":
            from neural_assemblies.assembly_calculus.emergent.curriculum.conversation import (
                train_conversation_exposure,
            )
            from neural_assemblies.assembly_calculus.emergent.core.corpus_index import (
                ingest_index_stats,
            )

            conv_idx = schedule.conversation_index
            conv = conv_idx.grounded if conv_idx else []
            if conv_idx is not None:
                ingest_index_stats(p, conv_idx)
            else:
                for sent in conv:
                    p.ingest_raw_sentence(sent.words)
            if conv_idx is not None:
                p.train_next_token(conv, corpus_index=conv_idx)
            elif conv:
                p.train_next_token(conv)
            train_conversation_exposure(p)
        else:
            continue
        phase_times[phase] = time.perf_counter() - t0

    total = time.perf_counter() - t_total
    parser.rounds = old_rounds
    ops["preset"] = preset
    ops["vocab_size"] = len(parser.stim_map)
    ops["lexicon_linked_words"] = sum(
        getattr(parser, "_lexicon_linked_words_by_core", {}).values(),
    )
    return total, phase_times, ops, parser


def bench_full_conversation_path() -> Tuple[float, bool, dict]:
    """Mirror chat_emergent bootstrap (DIALOGUE, no agent blocks)."""
    from neural_assemblies.assembly_calculus.emergent.core.corpus_index import (
        compile_corpus,
    )
    from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
        create_training_sentences,
    )
    from neural_assemblies.assembly_calculus.emergent.evaluation.parity import (
        evaluate_corpus_parity,
    )

    vocab = build_vocabulary_preset("medium")
    t0 = time.perf_counter()
    parser = EmergentParser(
        n=3000, k=30, seed=42, vocabulary=vocab, fast_training=True,
    )
    parser.train_for_conversation(max_stage="DIALOGUE", include_agent=False)
    elapsed = time.perf_counter() - t0

    sents = create_training_sentences()
    idx = compile_corpus(parser, sents)
    parity = evaluate_corpus_parity(parser, idx, max_probes=30)

    return elapsed, parser.fast_training, parity


def print_parity_metrics(parity: dict) -> None:
    print("\n=== READOUT PARITY (transition probes, post-train) ===")
    print(
        f"  top-1: {parity['top1']:.1%}  "
        f"top-3: {parity['top3']:.1%}  "
        f"top-5: {parity['top5']:.1%}  "
        f"(n={parity['total']})"
    )


def profile_prediction() -> str:
    vocab = build_vocabulary_preset("medium")
    parser = EmergentParser(
        n=3000, k=30, seed=42, vocabulary=vocab, fast_training=True,
    )
    trainer = CurriculumTrainer(parser)
    stage_words = trainer._get_stage_words("DIALOGUE")
    for w in stage_words:
        parser.register_word(w.lemma)
    sentences = trainer.generation.generate(stage_words, 4)
    trainer.generation.register_surface_forms(sentences, stage_words)
    phases = effective_stage_phases(
        "DIALOGUE", list(_STAGE_CONFIG["DIALOGUE"]["phases"]),
        fast=True,
    )
    schedule = TrainingScheduleExecutor.build_stage_schedule(
        parser, "DIALOGUE", sentences, phases,
        extra_prediction=create_instruction_sentences(),
        transition_cache=trainer._transition_cache,
    )
    # Warm prior phases minimally
    idx = schedule.corpus_index
    parser.train_lexicon(skip_known=True, words=idx.corpus_vocab)
    parser.train_distributional_from_index(idx, repetitions=1)
    pred_idx = schedule.prediction_index or idx

    expand_n = {"n": 0, "t": 0.0}

    def wrap(orig):
        def _w(*a, **k):
            t0 = time.perf_counter()
            r = orig(*a, **k)
            expand_n["n"] += 1
            expand_n["t"] += time.perf_counter() - t0
            return r
        return _w

    parser.brain._engine._expand_connectomes = wrap(
        parser.brain._engine._expand_connectomes,
    )

    pr = cProfile.Profile()
    pr.enable()
    parser.train_next_token(pred_idx.grounded, corpus_index=pred_idx)
    pr.disable()
    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(20)
    header = (
        f"expand_connectomes: {expand_n['n']} calls, {expand_n['t']:.2f}s\n"
        f"transitions={len(pred_idx.transitions)} "
        f"prefixes={len({t.prefix_words for t in pred_idx.transitions})}\n"
    )
    return header + buf.getvalue()


def print_cost_model(ops: Dict[str, int], phase_times: Dict[str, float]) -> None:
    """Rough algorithmic cost accounting."""
    r = 4  # fast DIALOGUE training rounds (approx)
    infer = 2
    bridge = 2

    lex_words = ops["stage_vocab"]
    role_steps = ops["role_updates_x_reps"]
    bridges = ops["bridge_steps"]
    prefixes = ops["unique_prefixes"]

    # Each lexicon word: O(rounds * project_cost(n,k))
  # Each role update: ~3 projections * rounds
    # Each bridge: O(prefix_len * infer + bridge_rounds * project)
    # expand_connectomes: O(new_winners * fanout) per project with growth

    est = {
        "lexicon_proj": lex_words * r,
        "role_proj": role_steps * 3 * r,
        "context_advances": prefixes * 4,  # avg prefix len ~4
        "bridge_proj": bridges * bridge,
    }

    print("\n=== ALGORITHMIC COST MODEL (DIALOGUE stage) ===")
    print(f"  vocab words trained (lexicon):     {lex_words}")
    print(f"  role update steps (with reps):     {role_steps}")
    print(f"  unique prefix groups:              {prefixes}")
    print(f"  bridge transitions (prediction):   {bridges}")
    print(f"  dialogue QA pairs:                 {ops['dialogue_pairs']}")
    print()
    print("  Estimated dominant projection counts:")
    for k, v in est.items():
        print(f"    {k:20} ~{v}")
    print()
    print("  Complexity drivers:")
    print("    - train_next_token: O(P * L_infer + T * R_bridge) * project(n,k)")
    print("      P=unique prefixes, T=transitions, each project may expand O(n*p) matrix")
    print("    - train_unsupervised: O(role_updates * reps * rounds * 3 projections)")
    print("    - train_lexicon: O(|corpus_vocab| * rounds); skip_known cuts repeat work")
    print("    - expand_connectomes early-exit helps but CONTEXT growth still ~O(n) per new winner")
    print()
    if phase_times:
        total = sum(phase_times.values())
        print("  Observed phase share:")
        for name, sec in sorted(phase_times.items(), key=lambda x: -x[1]):
            print(f"    {name:16} {sec:6.2f}s  ({100*sec/total:5.1f}%)")


def print_dual_metric_report(report: dict) -> None:
    print("\n=== DUAL-METRIC QUALITY GATE (exact vs compiled DIALOGUE) ===")
    opt = report["optimizer_parity"]
    exact = report["exact"]
    compiled = report["compiled"]
    print(
        f"  optimizer top-1 agreement: {opt['top1_agreement']:.1%} "
        f"(n={opt['total']})"
    )
    print(
        f"  dialogue acc  exact={exact['dialogue_accuracy']:.1%}  "
        f"compiled={compiled['dialogue_accuracy']:.1%}  "
        f"delta={report['dialogue_delta']:+.1%}"
    )
    print(
        f"  next-token top-5  exact={exact['next_token_top5']:.1%}  "
        f"compiled={compiled['next_token_top5']:.1%}  "
        f"delta={report['next_token_top5_delta']:+.1%}"
    )
    print(
        "  (optimizer parity on post-train weights — informational only; "
        f"agreement={opt['top1_agreement']:.1%})"
    )


def main() -> None:
    print("=== FULL train_for_conversation (medium, n=3000, k=30, fast) ===")
    full_t, fast, parity = bench_full_conversation_path()
    print(f"  TOTAL: {full_t:.1f}s  fast_training={fast}")
    print_parity_metrics(parity)

    if os.environ.get("DUAL_METRIC", "").strip() in ("1", "true", "yes"):
        from neural_assemblies.assembly_calculus.emergent.evaluation.parity import (
            run_dual_metric_gate,
        )

        print("\n=== Running dual-metric gate (set DUAL_METRIC=0 to skip) ===")
        report = run_dual_metric_gate(max_probes=25, qa_subset=15)
        print_dual_metric_report(report)

    if os.environ.get("GENERALIZATION", "").strip() in ("1", "true", "yes"):
        from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (
            run_generalization_gate,
        )

        print("\n=== GENERALIZATION GATE (holdout lexicon + novel composition) ===")
        gen = run_generalization_gate(max_bridge_probes=20, qa_subset=12)
        exact = gen["exact"]
        compiled = gen["compiled"]
        print(
            f"  lexicon holdout  exact={exact['lexicon_holdout']['accuracy']:.1%}  "
            f"compiled={compiled['lexicon_holdout']['accuracy']:.1%}  "
            f"delta={gen['lexicon_holdout_delta']:+.1%}"
        )
        print(
            f"  novel roles      exact={exact['novel_composition']['accuracy']:.1%}  "
            f"compiled={compiled['novel_composition']['accuracy']:.1%}  "
            f"delta={gen['novel_composition_delta']:+.1%}"
        )
        print(
            f"  bridge OOV top-5 exact={exact['bridge_oov']['top5']:.1%}  "
            f"compiled={compiled['bridge_oov']['top5']:.1%}  "
            f"delta={gen['bridge_oov_top5_delta']:+.1%}"
        )
        print(
            f"  composite        exact={exact['composite']:.1%}  "
            f"compiled={compiled['composite']:.1%}  "
            f"delta={gen['composite_delta']:+.1%}"
        )

    if os.environ.get("CURRICULUM_SWEEP", "").strip() in ("1", "true", "yes"):
        from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (
            format_curriculum_sweep_table,
            run_curriculum_depth_sweep,
        )

        depths = os.environ.get(
            "SWEEP_DEPTHS", "TWO_WORD,SENTENCES,DIALOGUE_FAST,FULL_TRAIN",
        ).split(",")
        depths = [d.strip() for d in depths if d.strip()]
        seed = int(os.environ.get("SWEEP_SEED", "42"))
        qa_subset = int(os.environ.get("SWEEP_QA_SUBSET", "12"))
        print("\n=== CURRICULUM DEPTH GENERALIZATION SWEEP ===")
        sweep = run_curriculum_depth_sweep(
            depths=depths,
            seed=seed,
            max_bridge_probes=20,
            qa_subset=qa_subset,
        )
        print(format_curriculum_sweep_table(sweep))

    print("\n=== DIALOGUE STAGE ONLY (phase decomposition) ===")
    total, phases, ops, _ = bench_dialogue_stage_only()
    print(f"  stage TOTAL: {total:.1f}s")
    for name, sec in sorted(phases.items(), key=lambda x: -x[1]):
        print(f"    {name:16} {sec:6.2f}s")

    print_cost_model(ops, phases)

    print("\n=== cProfile: train_next_token (DIALOGUE prediction index) ===")
    print(profile_prediction())


if __name__ == "__main__":
    main()

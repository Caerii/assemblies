#!/usr/bin/env python3
"""DIALOGUE large-vocab benchmark — wall time + phase share + lexicon link stats."""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ["TRAIN_PROGRESS"] = "0"

_medium_path = Path(__file__).with_name("_perf_dialogue_medium.py")
_spec = importlib.util.spec_from_file_location("perf_dialogue_medium", _medium_path)
assert _spec and _spec.loader
_pm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pm)

bench_dialogue_stage_only = _pm.bench_dialogue_stage_only
print_cost_model = _pm.print_cost_model
print_parity_metrics = _pm.print_parity_metrics

from neural_assemblies.assembly_calculus.emergent import (
    EmergentParser,
    build_vocabulary_preset,
)
from neural_assemblies.assembly_calculus.emergent.core.corpus_index import compile_corpus
from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
    create_training_sentences,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.parity import (
    evaluate_corpus_parity,
)


def bench_incremental_lexicon_link(*, preset: str = "large") -> dict:
    """Measure lexicon link amortization on incremental corpus-vocab batches."""
    import neural_assemblies.assembly_calculus.emergent.training.linker as tl

    parser = EmergentParser(
        n=3000, k=30, seed=42,
        vocabulary=build_vocabulary_preset(preset),
        fast_training=True,
    )
    words = list(parser.stim_map.keys())
    mid = max(1, len(words) // 2)
    first_batch = set(words[:mid])
    second_batch = set(words[mid : mid + max(1, len(words) // 4)])

    link_calls = {"n": 0}
    orig = tl.link_lexicon_topology

    def spy(pp, plan, **kw):
        if plan.lexicon_ops and (
            tl.lexicon_topology_needs_link(pp, plan) or kw.get("force")
        ):
            link_calls["n"] += 1
        return orig(pp, plan, **kw)

    tl.link_lexicon_topology = spy
    try:
        parser.train_lexicon(skip_known=False, words=first_batch)
        first_links = link_calls["n"]
        link_calls["n"] = 0
        parser.train_lexicon(skip_known=True, words=second_batch)
        incremental_links = link_calls["n"]
        link_calls["n"] = 0
        parser.train_lexicon(skip_known=True, words=second_batch)
        repeat_links = link_calls["n"]
    finally:
        tl.link_lexicon_topology = orig

    from neural_assemblies.assembly_calculus.emergent.training.compiler import (
        compile_lexicon_plan,
    )

    repeat_plan = compile_lexicon_plan(
        parser, skip_known=True, words=second_batch,
    )
    return {
        "first_batch_words": len(first_batch),
        "second_batch_words": len(second_batch),
        "linked_words_total": sum(parser._lexicon_linked_words_by_core.values()),
        "first_link_calls": first_links,
        "incremental_link_calls": incremental_links,
        "repeat_link_calls": repeat_links,
        "needs_link_on_repeat": tl.lexicon_topology_needs_link(parser, repeat_plan),
    }


def main() -> None:
    preset = "large"
    print(f"=== FULL train_for_conversation ({preset}, n=3000, k=30, fast) ===")
    vocab = build_vocabulary_preset(preset)
    t0 = time.perf_counter()
    parser = EmergentParser(
        n=3000, k=30, seed=42, vocabulary=vocab, fast_training=True,
    )
    parser.train_for_conversation(max_stage="DIALOGUE", include_agent=False)
    full = time.perf_counter() - t0
    print(f"  TOTAL: {full:.1f}s  vocab={len(parser.stim_map)}")

    idx = compile_corpus(parser, create_training_sentences())
    parity = evaluate_corpus_parity(parser, idx, max_probes=30)
    print_parity_metrics(parity)

    print(f"\n=== DIALOGUE STAGE ONLY ({preset} preset) ===")
    stage_t, phases, ops, stage_parser = bench_dialogue_stage_only(preset=preset)
    print(f"  stage TOTAL: {stage_t:.1f}s  vocab={ops['vocab_size']}")
    print(f"  lexicon linked words (post-stage): {ops['lexicon_linked_words']}")
    for name, sec in sorted(phases.items(), key=lambda x: -x[1]):
        print(f"    {name:16} {sec:6.2f}s")

    print_cost_model(ops, phases)

    print("\n=== MEDIUM vs LARGE (stage totals) ===")
    med_t, med_phases, med_ops, _ = bench_dialogue_stage_only(preset="medium")
    print(f"  medium stage: {med_t:.1f}s  lexicon={med_phases.get('lexicon', 0):.2f}s")
    print(f"  large stage:  {stage_t:.1f}s  lexicon={phases.get('lexicon', 0):.2f}s")
    print(
        f"  lexicon linked words  medium={med_ops['lexicon_linked_words']}  "
        f"large={ops['lexicon_linked_words']}"
    )

    print("\n=== LEXICON LINK AMORTIZATION (incremental batch) ===")
    link_stats = bench_incremental_lexicon_link(preset=preset)
    for key, val in link_stats.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()

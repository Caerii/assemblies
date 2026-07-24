#!/usr/bin/env python3
"""Third-pass performance analysis — post Tier A/B optimizations."""

from __future__ import annotations

import cProfile
import io
import pstats
import time
from typing import Callable

from neural_assemblies.assembly_calculus.emergent import EmergentParser
from neural_assemblies.assembly_calculus.emergent.curriculum.dialogue import (
    get_dialogue_curriculum,
)
from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
    create_instruction_sentences,
    create_training_sentences,
)

N, K, ROUNDS = 10000, 100, 8


def _agent_sentences():
    return (
        create_training_sentences()
        + create_instruction_sentences()
        + get_dialogue_curriculum()
    )


def train_for_agent_breakdown() -> None:
    """Mirror actual fast train_for_agent(include_blocks=False) path."""
    p = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS, fast_training=True)
    sents = _agent_sentences()
    raw = [s.words for s in sents]

    phases: list[tuple[str, float]] = []
    t0 = time.perf_counter()

    t = time.perf_counter()
    for sent in raw:
        p.ingest_raw_sentence(sent)
    phases.append(("ingest", time.perf_counter() - t))

    t = time.perf_counter()
    p.train_lexicon(skip_known=False)
    phases.append(("lexicon", time.perf_counter() - t))

    for name, fn in [
        ("roles", lambda: p.train_roles(sents)),
        ("phrases", lambda: p.train_phrases(sents)),
        ("tense", lambda: p.train_tense(raw)),
        ("mood", lambda: p.train_mood(raw)),
        ("polarity", lambda: p.train_polarity(raw)),
        ("number", lambda: p.train_number(raw)),
        ("conjunctions", lambda: p.train_conjunctions(raw)),
        ("next_token", lambda: p.train_next_token(sents)),
    ]:
        t = time.perf_counter()
        fn()
        phases.append((name, time.perf_counter() - t))

    phases.append(("TOTAL", time.perf_counter() - t0))

    print("=== train_for_agent FAST PATH (n=10k, no word_order, no extra dialogue) ===")
    total = phases[-1][1]
    for name, sec in phases[:-1]:
        print(f"  {name:14} {sec:6.2f}s  ({100 * sec / total:5.1f}%)")
    print(f"  {'TOTAL':14} {total:6.2f}s")
    print(f"  rounds={p.rounds} infer={p.inference_rounds} bridge={p.bridge_rounds}")


def next_token_subphase() -> None:
    """Break down train_next_token internals."""
    p = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS, fast_training=True)
    sents = _agent_sentences()
    p.train_lexicon(skip_known=False)

    counts = {
        "expand_connectomes": 0,
        "reset_context": 0,
        "advance_direct": 0,
        "bridge": 0,
        "project": 0,
        "project_rounds": 0,
    }
    expand_time = 0.0

    orig_ex = p.brain._engine._expand_connectomes
    orig_p = p.brain.project
    orig_pr = p.brain.project_rounds
    orig_reset = p._reset_context_state
    orig_adv = p._advance_context_direct
    orig_bridge = p._train_next_token_bridge

    def wrap_ex(*a, **k):
        nonlocal expand_time
        t = time.perf_counter()
        r = orig_ex(*a, **k)
        expand_time += time.perf_counter() - t
        counts["expand_connectomes"] += 1
        return r

    def wrap_p(*a, **k):
        counts["project"] += 1
        return orig_p(*a, **k)

    def wrap_pr(*a, **k):
        counts["project_rounds"] += 1
        return orig_pr(*a, **k)

    def wrap_reset(*a, **k):
        counts["reset_context"] += 1
        return orig_reset(*a, **k)

    def wrap_adv(*a, **k):
        counts["advance_direct"] += 1
        return orig_adv(*a, **k)

    def wrap_bridge(*a, **k):
        counts["bridge"] += 1
        return orig_bridge(*a, **k)

    p.brain._engine._expand_connectomes = wrap_ex
    p.brain.project = wrap_p
    p.brain.project_rounds = wrap_pr
    p._reset_context_state = wrap_reset
    p._advance_context_direct = wrap_adv
    p._train_next_token_bridge = wrap_bridge

    t = time.perf_counter()
    p.train_next_token(sents)
    elapsed = time.perf_counter() - t

    print("\n=== train_next_token INTERNALS ===")
    print(f"  wall time: {elapsed:.2f}s")
    print(f"  expand_connectomes: {counts['expand_connectomes']} calls, {expand_time:.2f}s")
    print(f"  reset_context: {counts['reset_context']}")
    print(f"  advance_direct: {counts['advance_direct']}")
    print(f"  bridge steps: {counts['bridge']}")
    print(f"  project: {counts['project']}, project_rounds: {counts['project_rounds']}")


def profile_hot_functions(label: str, fn: Callable[[], None]) -> None:
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(20)
    print(f"\n=== cProfile: {label} ===")
    print(buf.getvalue())


def morphology_detail() -> None:
    p = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS, fast_training=True)
    sents = _agent_sentences()
    raw = [s.words for s in sents]
    p.train_lexicon(skip_known=False)
    p.train_roles(sents)
    p.train_phrases(sents)

    print("\n=== MORPHOLOGY PHASE DETAIL ===")
    for name, fn in [
        ("tense", lambda: p.train_tense(raw)),
        ("mood", lambda: p.train_mood(raw)),
        ("polarity", lambda: p.train_polarity(raw)),
        ("number", lambda: p.train_number(raw)),
        ("conjunctions", lambda: p.train_conjunctions(raw)),
    ]:
        t = time.perf_counter()
        fn()
        print(f"  {name:14} {time.perf_counter() - t:6.2f}s")


def roles_phrases_detail() -> None:
    p = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS, fast_training=True)
    sents = _agent_sentences()
    p.train_lexicon(skip_known=False)

    print("\n=== ROLES / PHRASES DETAIL ===")
    for name, fn in [
        ("roles", lambda: p.train_roles(sents)),
        ("phrases", lambda: p.train_phrases(sents)),
    ]:
        t = time.perf_counter()
        fn()
        print(f"  {name:14} {time.perf_counter() - t:6.2f}s")


def per_bridge_step_cost() -> None:
    p = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS, fast_training=True)
    sents = _agent_sentences()
    p.train_lexicon(skip_known=False)
    p._ensure_prediction_lexicon(list(p.stim_map.keys()))
    p._pregrow_context_capacity(sents)

    sent = sents[0]
    words = [w for w in sent.words if w in p.stim_map]

    # One bridge step: advance + bridge
    p._reset_context_state()
    t0 = time.perf_counter()
    p._advance_context_direct(words[0], rounds=p.inference_rounds)
    p._train_next_token_bridge(p.stim_map[words[1]], bridge_rounds=p.bridge_rounds)
    one_step = time.perf_counter() - t0

    n_steps = sum(
        max(0, len([w for w in s.words if w in p.stim_map]) - 1)
        for s in sents
    )
    print("\n=== PER BRIDGE STEP ===")
    print(f"  one step (advance+bridge): {one_step*1000:.1f} ms")
    print(f"  total bridge steps in corpus: {n_steps}")
    print(f"  extrapolated bridge loop: {one_step * n_steps:.1f}s")


def inference_micro() -> None:
    p = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS, fast_training=True)
    p.train_lexicon(skip_known=False)
    p.train_next_token(_agent_sentences())

    words = ["the", "dog", "chases", "the", "cat"]
    ctx2 = words[:2]

    benches = [
        ("build_context_direct x100", lambda: [
            p.build_context_incremental(words, reset=True, direct=True)
            for _ in range(100)
        ]),
        ("build_context_fiber x100", lambda: [
            p.build_context_incremental(words, reset=True, direct=False)
            for _ in range(100)
        ]),
        ("parse_full x100", lambda: [
            p.parse_incremental(words, reset=True, light=False) for _ in range(100)
        ]),
        ("predict_next x100", lambda: [
            p.predict_next(ctx2) for _ in range(100)
        ]),
        ("present_turn x100", lambda: [
            p.present_turn(words, learn=False) for _ in range(100)
        ]),
    ]
    print("\n=== INFERENCE MICRO (n=10k, chat hot paths) ===")
    for label, fn in benches:
        fn()
        t0 = time.perf_counter()
        fn()
        print(f"  {label}: {(time.perf_counter() - t0) / 100 * 1000:.1f} ms/op")


def chat_n_comparison() -> None:
    print("\n=== train_for_agent by n/k (fast) ===")
    for n, k in [(3000, 30), (5000, 50), (10000, 100)]:
        p = EmergentParser(n=n, k=k, seed=42, rounds=ROUNDS, fast_training=True)
        t0 = time.perf_counter()
        p.train_for_agent(include_blocks=False)
        elapsed = time.perf_counter() - t0
        print(f"  n={n:>5} k={k:>3}: {elapsed:5.1f}s  engine={p.engine_name}")


def main() -> None:
    train_for_agent_breakdown()
    next_token_subphase()
    roles_phrases_detail()
    morphology_detail()
    per_bridge_step_cost()
    inference_micro()
    chat_n_comparison()

    p = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS, fast_training=True)
    sents = _agent_sentences()
    p.train_lexicon(skip_known=False)
    profile_hot_functions("train_next_token", lambda: p.train_next_token(sents))

    p2 = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS, fast_training=True)
    p2.train_lexicon(skip_known=False)
    profile_hot_functions("train_roles", lambda: p2.train_roles(sents))


if __name__ == "__main__":
    main()

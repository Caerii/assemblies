#!/usr/bin/env python3
"""Second-pass performance analysis for emergent training."""

from __future__ import annotations

import cProfile
import io
import pstats
import time

from neural_assemblies.assembly_calculus.emergent import (
    EmergentParser,
    build_vocabulary_preset,
)
from neural_assemblies.assembly_calculus.emergent.curriculum.dialogue import (
    get_dialogue_curriculum,
)
from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
    create_instruction_sentences,
    create_training_sentences,
)

N, K, ROUNDS = 10000, 100, 8


def phase_breakdown() -> None:
    vocab = build_vocabulary_preset("core")
    p = EmergentParser(
        n=N, k=K, seed=42, rounds=ROUNDS, vocabulary=vocab, fast_training=True,
    )
    sents = (
        create_training_sentences()
        + create_instruction_sentences()
        + get_dialogue_curriculum()
    )
    raw = [s.words for s in sents]

    phases: list[tuple[str, float]] = []
    t = time.perf_counter()
    p.train_lexicon(skip_known=False)
    phases.append(("lexicon", time.perf_counter() - t))

    for name, fn in [
        ("roles", lambda: p.train_roles(sents)),
        ("phrases", lambda: p.train_phrases(sents)),
        ("word_order", lambda: p.train_word_order(sents)),
        ("tense", lambda: p.train_tense(raw)),
        ("mood", lambda: p.train_mood(raw)),
        ("polarity", lambda: p.train_polarity(raw)),
        ("number", lambda: p.train_number(raw)),
        ("conjunctions", lambda: p.train_conjunctions(raw)),
        ("next_token", lambda: p.train_next_token(sents)),
        ("dialogue", lambda: p.train_dialogue()),
    ]:
        t = time.perf_counter()
        fn()
        phases.append((name, time.perf_counter() - t))

    total = sum(s for _, s in phases)
    print("=== PHASE BREAKDOWN (n=10k, fast_training) ===")
    for name, sec in phases:
        print(f"  {name:14} {sec:6.2f}s  ({100 * sec / total:5.1f}%)")
    print(f"  {'TOTAL':14} {total:6.2f}s")


def profile_next_token() -> None:
    vocab = build_vocabulary_preset("core")
    sents = (
        create_training_sentences()
        + create_instruction_sentences()
        + get_dialogue_curriculum()
    )
    p = EmergentParser(
        n=N, k=K, seed=42, rounds=ROUNDS, vocabulary=vocab, fast_training=True,
    )
    p.train_lexicon(skip_known=False)
    p._ensure_prediction_lexicon(list(p.stim_map.keys()))

    pr = cProfile.Profile()
    pr.enable()
    p.train_next_token(sents)
    pr.disable()

    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(25)
    print("\n=== cProfile train_next_token ===")
    print(buf.getvalue())


def inference_micro() -> None:
    vocab = build_vocabulary_preset("core")
    p = EmergentParser(
        n=N, k=K, seed=42, rounds=ROUNDS, vocabulary=vocab, fast_training=True,
    )
    p.train_lexicon(skip_known=False)
    p._ensure_prediction_lexicon(list(p.stim_map.keys()))
    from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
        create_training_sentences,
    )
    p.train_next_token(create_training_sentences()[:10])

    words = ["the", "dog", "chases", "the", "cat"]
    ctx2 = words[:2]

    benches = [
        ("build_context x100", lambda: [
            p.build_context_incremental(words, reset=True) for _ in range(100)
        ]),
        ("parse_light x100", lambda: [
            p.parse_incremental(words, reset=True, light=True) for _ in range(100)
        ]),
        ("parse_full x100", lambda: [
            p.parse_incremental(words, reset=True, light=False) for _ in range(100)
        ]),
        ("predict_next x100", lambda: [
            p.predict_next(ctx2) for _ in range(100)
        ]),
    ]
    print("\n=== INFERENCE MICRO (after partial train) ===")
    for label, fn in benches:
        fn()  # warmup
        t0 = time.perf_counter()
        fn()
        print(f"  {label}: {(time.perf_counter() - t0) / 100 * 1000:.1f} ms/op")


def projection_count() -> None:
    vocab = build_vocabulary_preset("core")
    p = EmergentParser(n=3000, k=30, seed=42, rounds=6, vocabulary=vocab)
    p.train_lexicon(skip_known=False)
    p._ensure_prediction_lexicon(list(p.stim_map.keys()))

    sent = create_training_sentences()[0]
    counts = {"project": 0, "project_rounds": 0, "expand": 0}

    orig_p = p.brain.project
    orig_pr = p.brain.project_rounds
    orig_ex = p.brain._engine._expand_connectomes

    def wrap_p(*a, **k):
        counts["project"] += 1
        return orig_p(*a, **k)

    def wrap_pr(*a, **k):
        counts["project_rounds"] += 1
        return orig_pr(*a, **k)

    def wrap_ex(*a, **k):
        counts["expand"] += 1
        return orig_ex(*a, **k)

    p.brain.project = wrap_p
    p.brain.project_rounds = wrap_pr
    p.brain._engine._expand_connectomes = wrap_ex
    p.train_next_token([sent])

    words = [w for w in sent.words if w in p.stim_map]
    print("\n=== PROJECTION COUNT (one sentence, n=3k) ===")
    print(f"  words: {len(words)}")
    for k, v in counts.items():
        print(f"  {k}: {v}")


def engine_micro() -> None:
    from neural_assemblies.core.brain import Brain
    from neural_assemblies.assembly_calculus.ops import project

    print("\n=== ENGINE MICRO (stable connectomes) ===")
    for n, k in [(3000, 30), (10000, 100)]:
        b = Brain(p=0.05, seed=42, engine="numpy_sparse", n_hint=n)
        b.add_stimulus("s", k)
        b.add_area("A", n, k, 0.1)
        b.add_area("B", n, k, 0.1)
        project(b, "s", "A", rounds=8)
        for _ in range(20):
            b.project({}, {"A": ["B"]})
        t0 = time.perf_counter()
        for _ in range(200):
            b.project({}, {"A": ["B"]})
        ms = (time.perf_counter() - t0) / 200 * 1000
        print(f"  n={n:>5} k={k:>3} A->B: {ms:.2f} ms/proj")


def main() -> None:
    phase_breakdown()
    profile_next_token()
    inference_micro()
    projection_count()
    engine_micro()


if __name__ == "__main__":
    main()

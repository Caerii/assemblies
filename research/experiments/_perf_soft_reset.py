#!/usr/bin/env python3
"""Compare hard vs soft CONTEXT reset during train_next_token."""

import time

from neural_assemblies.assembly_calculus.emergent import EmergentParser, build_vocabulary_preset
from neural_assemblies.assembly_calculus.emergent.core.areas import CONTEXT
from neural_assemblies.assembly_calculus.emergent.curriculum import CurriculumTrainer
from neural_assemblies.assembly_calculus.emergent.core.grounding import GroundingContext
from neural_assemblies.assembly_calculus.emergent.curriculum.data import GroundedSentence


def soft_reset(parser):
    parser.brain.inhibit_areas([CONTEXT])
    if parser.brain._engine.is_fixed(CONTEXT):
        parser.brain._engine.unfix_assembly(CONTEXT)


def bench(label, parser, sentences, reset_fn):
    counts = {"n": 0, "t": 0.0}
    orig = parser.brain._engine._expand_connectomes

    def wrap(*a, **k):
        t0 = time.perf_counter()
        r = orig(*a, **k)
        counts["t"] += time.perf_counter() - t0
        counts["n"] += 1
        return r

    parser._reset_context_state = lambda: reset_fn(parser)
    parser.brain._engine._expand_connectomes = wrap
    t0 = time.perf_counter()
    parser.train_next_token(sentences)
    elapsed = time.perf_counter() - t0
    print(f"  {label}: {elapsed:.2f}s  expand={counts['n']} calls {counts['t']:.2f}s")


def main():
    vocab = build_vocabulary_preset("medium")
    trainer = CurriculumTrainer(EmergentParser(n=3000, k=30, vocabulary=vocab))
    stage_words = trainer._get_stage_words("DIALOGUE")
    sentences_raw = trainer._generate_sentences(stage_words, 4)[:15]

    gs = [
        GroundedSentence(
            words=s,
            contexts=[GroundingContext()] * len(s),
            roles=[None] * len(s),
        )
        for s in sentences_raw
    ]

    for seed, reset_fn, label in [
        (42, lambda p: p._reset_area_activity(CONTEXT), "hard reset"),
        (43, soft_reset, "soft reset (winners only)"),
    ]:
        p = EmergentParser(
            n=3000, k=30, seed=seed, vocabulary=vocab, fast_training=True,
        )
        for w in stage_words:
            p.register_word(w.lemma)
        p.train_lexicon(skip_known=True)
        p._ensure_prediction_lexicon(list(p.stim_map.keys())[:80])
        p._pregrow_context_capacity(gs)
        bench(label, p, gs, reset_fn)


if __name__ == "__main__":
    main()

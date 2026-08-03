"""CONTEXT extension. Implements PREREG_context_beyond_bigram.md.

    stim[w]  -> LEX
    gstim[w] -> PRED                        (grounding, unchanged)
    LEX -> CONTEXT, CONTEXT -> CONTEXT      accumulating prefix trace
    LEX + CONTEXT -> PRED

CONTEXT resets between SENTENCES, not between words.
"""
from __future__ import annotations
import random
from typing import Dict, List
import numpy as np
from neural_assemblies.core.brain import Brain
from ntp import (N, K, P, TRAIN_ROUNDS, GROUND_ROUNDS, SETTLE_ROUNDS,
                 vocabulary, generate, _snap, _ov)


def build(seed, words, beta, rec_beta=None, engine="numpy_sparse"):
    np.random.seed(seed); random.seed(seed)
    b = Brain(p=P, seed=seed, engine=engine)
    for w in words:
        b.add_stimulus(f"s_{w}", K); b.add_stimulus(f"g_{w}", K)
    for a in ("LEX", "PRED", "CONTEXT"):
        b.add_area(a, N, K, beta)
    if rec_beta is not None:
        # CONTEXT->CONTEXT stays OPEN (so the fiber materialises and carries
        # real drive) but does not potentiate. Separates "recurrence" from
        # "plasticity ON the recurrence".
        #
        # MUST be Brain.update_plasticity: Area.update_beta_by_area writes a
        # dict the sparse engine never reads, so the first run of this arm was
        # bit-identical to the ungated one (task #88). Gating cannot express
        # this either -- project_map derives CONTEXT->CONTEXT from the INPUT
        # fiber, so it is open whenever CONTEXT is being written to.
        b.update_plasticity("CONTEXT", "CONTEXT", rec_beta)
        assert b._engine.get_beta("CONTEXT", "CONTEXT") == rec_beta
    pred_sig: Dict[str, np.ndarray] = {}
    for w in words:
        b.inhibit_areas(["LEX", "PRED", "CONTEXT"])
        for _ in range(GROUND_ROUNDS):
            b.project({f"s_{w}": ["LEX"]}, {})
        b.inhibit_areas(["PRED"])
        for _ in range(GROUND_ROUNDS):
            b.project({f"g_{w}": ["PRED"]}, {})
        pred_sig[w] = _snap(b, "PRED")
    return b, pred_sig


def train(b, corpus, ctx_recurrent_train=True):
    """`ctx_recurrent_train=False` CLOSES CONTEXT->CONTEXT while weights change."""
    for s in corpus:
        b.inhibit_areas(["LEX", "PRED", "CONTEXT"])      # reset per SENTENCE
        for a, nxt in zip(s, s[1:]):
            for _ in range(TRAIN_ROUNDS):
                src = {"LEX": ["CONTEXT", "PRED"], "CONTEXT": ["PRED"]}
                if ctx_recurrent_train:
                    src["CONTEXT"] = ["CONTEXT", "PRED"]
                b.project({f"s_{a}": ["LEX"], f"g_{nxt}": ["PRED"]}, src)


def score(b, corpus, words, pred_sig, tie_seed=0, collect_ctx=None,
          ctx_recurrent_read=True):
    rng = random.Random(tie_seed)
    rr, n = 0.0, 0
    prev = b.disable_plasticity
    b.disable_plasticity = True
    try:
        for s in corpus:
            b.inhibit_areas(["LEX", "PRED", "CONTEXT"])
            for pos, (a, truth) in enumerate(zip(s, s[1:])):
                for _ in range(SETTLE_ROUNDS):
                    src = {"LEX": ["CONTEXT"]}
                    if ctx_recurrent_read:
                        src["CONTEXT"] = ["CONTEXT"]
                    b.project({f"s_{a}": ["LEX"]}, src)
                if collect_ctx is not None:
                    collect_ctx.setdefault(pos, []).append(_snap(b, "CONTEXT"))
                b.project({}, {"LEX": ["PRED"], "CONTEXT": ["PRED"]})
                got = _snap(b, "PRED")
                keyed = [(-_ov(got, pred_sig[w]), rng.random(), w) for w in words]
                keyed.sort()
                rr += 1.0 / ([w for _o, _t, w in keyed].index(truth) + 1); n += 1
    finally:
        b.disable_plasticity = prev
    return rr / max(n, 1)


def run(seed, beta, vocab_size=50, n_train=200, n_test=25, ctx_probe=False,
        rec_train=True, rec_read=True, rec_beta=None,
        engine="numpy_sparse"):
    words = vocabulary(vocab_size); keep = set(words)
    tr = [[w for w in s if w in keep] for s in generate(n_train, seed)]
    te = [[w for w in s if w in keep] for s in generate(n_test, seed + 500)]
    b, pred_sig = build(seed, words, beta, rec_beta=rec_beta, engine=engine)
    train(b, tr, ctx_recurrent_train=rec_train)
    ctx = {} if ctx_probe else None
    mrr = score(b, te, words, pred_sig, collect_ctx=ctx,
                ctx_recurrent_read=rec_read)
    if not ctx_probe:
        return mrr
    ovs = []
    for pos, arrs in ctx.items():
        for i in range(min(len(arrs), 12)):
            for j in range(i + 1, min(len(arrs), 12)):
                ovs.append(_ov(arrs[i], arrs[j]))
    return mrr, (float(np.mean(ovs)) if ovs else float("nan"))

"""Three-pathway next-token predictor. Implements PREREG_next_token_predictive.md.

    stim[w]  -> LEX     word representation
    gstim[w] -> PRED    grounding signature, independent of LEX
    LEX      -> PRED    the learned transition

Parameters are FIXED by the pre-registration and must not be tuned here.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List, cast

import numpy as np

from neural_assemblies.core.brain import Brain

N, K, P, BETA = 10000, 200, 0.05, 0.10
TRAIN_ROUNDS, GROUND_ROUNDS, SETTLE_ROUNDS = 3, 5, 3

WORD_CLASSES = {
    "DET": ["the", "a", "this", "that", "some"],
    "ADJ": ["big", "small", "red", "old", "fast",
            "new", "good", "bad", "hot", "cold"],
    "NOUN": ["dog", "cat", "bird", "man", "woman", "boy", "girl", "ball",
             "house", "car", "tree", "fish", "book", "door", "hand"],
    "VERB": ["sees", "chases", "eats", "likes", "gives",
             "takes", "hits", "finds", "holds", "makes"],
    "PREP": ["in", "on", "by", "with", "near",
             "to", "from", "at", "under", "over"],
}


def vocabulary(size: int = 50) -> List[str]:
    words = [w for ws in WORD_CLASSES.values() for w in ws]
    return words[:size]


def generate(n: int, seed: int) -> List[List[str]]:
    rng = random.Random(seed)
    C = WORD_CLASSES
    out = []
    for _ in range(n):
        s = [rng.choice(C["DET"])]
        if rng.random() < 0.5:
            s.append(rng.choice(C["ADJ"]))
        s += [rng.choice(C["NOUN"]), rng.choice(C["VERB"])]
        if rng.random() < 0.5:
            s.append(rng.choice(C["PREP"]))
            s.append(rng.choice(C["DET"]))
            s.append(rng.choice(C["NOUN"]))
        else:
            s.append(rng.choice(C["DET"]))
            if rng.random() < 0.5:
                s.append(rng.choice(C["ADJ"]))
            s.append(rng.choice(C["NOUN"]))
        out.append(s)
    return out


def _snap(brain, area) -> np.ndarray:
    return np.array(brain.areas[area].winners, dtype=np.uint32, copy=True)


def _ov(a, b) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return len(set(a.tolist()) & set(b.tolist())) / min(len(a), len(b))


def build(seed: int, words: List[str], beta: float, grounded: bool,
          engine: str = "numpy_sparse"):
    """`grounded=False` is the H3 ablation: PRED signatures come from LEX->PRED."""
    cast(Any, np.random).seed(seed)
    random.seed(seed)
    b = Brain(p=P, seed=seed, engine=engine)
    for w in words:
        b.add_stimulus(f"s_{w}", K)
        if grounded:
            b.add_stimulus(f"g_{w}", K)
    b.add_area("LEX", N, K, beta)
    b.add_area("PRED", N, K, beta)

    lex_sig: Dict[str, np.ndarray] = {}
    pred_sig: Dict[str, np.ndarray] = {}
    for w in words:
        b.inhibit_areas(["LEX", "PRED"])
        for _ in range(GROUND_ROUNDS):
            b.project({f"s_{w}": ["LEX"]}, {})
        lex_sig[w] = _snap(b, "LEX")
        b.inhibit_areas(["PRED"])
        for _ in range(GROUND_ROUNDS):
            if grounded:
                b.project({f"g_{w}": ["PRED"]}, {})
            else:
                # ABLATION: the same fiber that must later carry a -> b is
                # also asked to define PRED(w). They compete.
                b.project({f"s_{w}": ["LEX"]}, {"LEX": ["PRED"]})
        pred_sig[w] = _snap(b, "PRED")
    return b, lex_sig, pred_sig


def train(b, corpus, grounded: bool):
    for s in corpus:
        for a, nxt in zip(s, s[1:]):
            b.inhibit_areas(["LEX", "PRED"])
            for _ in range(TRAIN_ROUNDS):
                if grounded:
                    # a in LEX and b's grounding in PRED, CO-ACTIVE, so the
                    # LEX->PRED fiber learns a -> b.
                    b.project({f"s_{a}": ["LEX"], f"g_{nxt}": ["PRED"]},
                              {"LEX": ["PRED"]})
                else:
                    b.project({f"s_{a}": ["LEX"]}, {"LEX": ["PRED"]})


def score(b, corpus, words, pred_sig, tie_seed: int = 0) -> float:
    """Mean reciprocal rank of the true next word.

    TIES ARE BROKEN RANDOMLY, not by vocabulary order. Overlap is quantised to
    multiples of 1/k and is frequently 0 for every candidate, so `sorted()`
    would fall through to the order of `words` -- DET, ADJ, NOUN, VERB, PREP --
    which correlates with which classes actually follow. The beta=0 null
    control scored 0.1191 that way, i.e. the unigram baseline, with nothing
    learned. A random tie-break makes an uninformative readout score chance,
    which is what an uninformative readout must score.
    """
    rng = random.Random(tie_seed)
    rr, n = 0.0, 0
    prev = b.disable_plasticity
    b.disable_plasticity = True
    try:
        for s in corpus:
            for a, truth in zip(s, s[1:]):
                b.inhibit_areas(["LEX", "PRED"])
                for _ in range(SETTLE_ROUNDS):
                    b.project({f"s_{a}": ["LEX"]}, {})
                # PRED driven ONLY by LEX -- no grounding, so what it holds is
                # a prediction rather than an echo of the input.
                b.project({}, {"LEX": ["PRED"]})
                got = _snap(b, "PRED")
                keyed = [(-_ov(got, pred_sig[w]), rng.random(), w)
                         for w in words]
                keyed.sort()
                ranked = [w for _o, _t, w in keyed]
                rr += 1.0 / (ranked.index(truth) + 1)
                n += 1
    finally:
        b.disable_plasticity = prev
    return rr / max(n, 1)


def run(seed: int, beta: float, grounded: bool, vocab_size: int = 50,
        n_train: int = 200, n_test: int = 25,
        engine: str = "numpy_sparse") -> float:
    words = vocabulary(vocab_size)
    train_c = generate(n_train, seed)
    test_c = generate(n_test, seed + 500)
    keep = set(words)
    train_c = [[w for w in s if w in keep] for s in train_c]
    test_c = [[w for w in s if w in keep] for s in test_c]
    b, _lex, pred_sig = build(seed, words, beta, grounded, engine=engine)
    train(b, train_c, grounded)
    return score(b, test_c, words, pred_sig)


def baselines(seed: int, vocab_size: int = 50):
    words = set(vocabulary(vocab_size))
    tr = [[w for w in s if w in words] for s in generate(2000, seed)]
    te = [[w for w in s if w in words] for s in generate(25, seed + 500)]
    uni, big = Counter(), {}
    for s in tr:
        for a, nxt in zip(s, s[1:]):
            uni[nxt] += 1
            big.setdefault(a, Counter())[nxt] += 1
    V = sorted(words)

    def mrr(scorer):
        rr, n = 0.0, 0
        for s in te:
            for a, truth in zip(s, s[1:]):
                ranked = sorted(V, key=lambda w: -scorer(a, w))
                rr += 1.0 / (ranked.index(truth) + 1)
                n += 1
        return rr / max(n, 1)

    return (sum(1.0 / i for i in range(1, len(V) + 1)) / len(V),
            mrr(lambda a, w: uni[w]),
            mrr(lambda a, w: big.get(a, Counter())[w]))

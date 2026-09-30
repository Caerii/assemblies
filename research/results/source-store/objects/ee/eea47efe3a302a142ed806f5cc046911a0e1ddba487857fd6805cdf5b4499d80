"""A next-token corpus in which HISTORY is worth something.

PREREG_agreement_corpus.md. The study-4 corpus (`ntp.py`) is a nine-phase
template in which a perfect state adds 0.02 MRR over a bigram
(seq_a3_oracle_ceiling.py), so no state effect is measurable on it. This
generator adds number agreement across intervening material:

    DET_n (ADJ) NOUN_n [PREP DET_m (ADJ) NOUN_m] VERB_n DET_o (ADJ) NOUN_o TAG_n

The subject noun's number n is drawn once; the verb and the sentence-final
tag agree with it. When the optional prepositional phrase is present its
noun has an independent number m, so a bigram at the verb sees the wrong
noun half the time; an oracle state that carries the subject's number does
not. Determiners agree with the noun that follows (local; a bigram sees
that). The vocabulary is 50 words, as in study 4.

`oracle_gap(seed)` returns the MRR of a bigram and of an oracle state
(subject number and template phase) on the seed's own train/test corpora,
in the study's units (random tie-break); the corpus is accepted for the
transducer study only if the gap is at least GAP_BAR.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

WORD_CLASSES: Dict[str, List[str]] = {
    "DET_sg": ["a", "this", "that", "every"],
    "DET_pl": ["these", "those", "some", "many"],
    "ADJ": ["big", "small", "red", "old", "new", "good"],
    "NOUN_sg": ["dog", "cat", "bird", "man", "girl", "ball", "house", "tree"],
    "NOUN_pl": ["dogs", "cats", "birds", "men", "girls", "balls", "houses", "trees"],
    "VERB_sg": ["sees", "chases", "likes", "takes", "finds", "holds"],
    "VERB_pl": ["see", "chase", "like", "take", "find", "hold"],
    "PREP": ["in", "on", "by", "with", "near", "under"],
    "TAG_sg": ["doesnt"],
    "TAG_pl": ["dont"],
}
#: the CHAIN variant: every other word carries the subject's number, the
#: words between are nouns of independent number (the distractors), so half
#: of the positions are agreement sites behind a distractor
CHAIN_CLASSES: Dict[str, List[str]] = {
    "AUX_sg": ["does", "is"], "AUX_pl": ["do", "are"],
    "VERB_sg": ["sees", "likes"], "VERB_pl": ["see", "like"],
    "PRON_sg": ["it", "itself"], "PRON_pl": ["they", "themselves"],
    "TAG_sg": ["doesnt", "isnt"], "TAG_pl": ["dont", "arent"],
    "NOUN_sg": ["dog", "cat", "bird", "man", "girl", "ball", "house", "tree",
                "car", "fish", "book", "door", "hand", "boy", "woman", "road", "cup"],
    "NOUN_pl": ["dogs", "cats", "birds", "men", "girls", "balls", "houses", "trees",
                "cars", "fish2", "books", "doors", "hands", "boys", "women", "roads", "cups"],
}
CHAIN_ORDER = ["AUX", "VERB", "PRON", "TAG"]
CLASS = {w: c for c, ws in WORD_CLASSES.items() for w in ws}
N_TRAIN, N_TEST = 200, 25
GAP_BAR = 0.10


CHAIN = False   # set by `use_chain()`
GAP = 1         # distractor nouns between agreeing words (Amendment 1: 2)


def use_chain(on: bool = True, gap: int = 1) -> None:
    global CHAIN, CLASS, GAP
    CHAIN = on
    GAP = int(gap)
    CLASS = {w: c for c, ws in (CHAIN_CLASSES if on else WORD_CLASSES).items() for w in ws}


def vocabulary(size: int = 50) -> List[str]:
    words = [w for ws in (CHAIN_CLASSES if CHAIN else WORD_CLASSES).values() for w in ws]
    assert len(words) == 50, len(words)
    return words[:size]


def generate_chain(n: int, seed: int, *, gap: int | None = None) -> List[List[str]]:
    """AUX_n NOUN_m VERB_n NOUN_m' PRON_n NOUN_m'' TAG_n: four agreement sites
    behind distractors in seven positions."""
    gap = GAP if gap is None else int(gap)
    if gap < 1:
        raise ValueError("chain gap must be positive")
    rng = random.Random(seed)
    C = CHAIN_CLASSES
    out = []
    for _ in range(n):
        subj = rng.choice(("sg", "pl"))
        s = []
        for j, cls in enumerate(CHAIN_ORDER):
            s.append(rng.choice(C[f"{cls}_{subj}"]))
            if j < len(CHAIN_ORDER) - 1:
                for _g in range(gap):
                    s.append(rng.choice(C[f"NOUN_{rng.choice(('sg', 'pl'))}"]))
        out.append(s)
    return out


def generate(n: int, seed: int) -> List[List[str]]:
    if CHAIN:
        return generate_chain(n, seed)
    rng = random.Random(seed)
    C = WORD_CLASSES

    def np_(num, out):
        out.append(rng.choice(C[f"DET_{num}"]))
        if rng.random() < 0.5:
            out.append(rng.choice(C["ADJ"]))
        out.append(rng.choice(C[f"NOUN_{num}"]))

    out = []
    for _ in range(n):
        s: List[str] = []
        subj = rng.choice(("sg", "pl"))
        np_(subj, s)
        if rng.random() < 0.5:
            s.append(rng.choice(C["PREP"]))
            np_(rng.choice(("sg", "pl")), s)
        s.append(rng.choice(C[f"VERB_{subj}"]))
        np_(rng.choice(("sg", "pl")), s)
        s.append(rng.choice(C[f"TAG_{subj}"]))
        out.append(s)
    return out


# --- the oracle state --------------------------------------------------------------------

def states(sentence: List[str]) -> List[Tuple[int, str]]:
    """The oracle state AFTER each word: (template phase, subject number).
    Phases: 1 after subject DET, 2 after subject ADJ, 3 after subject NOUN,
    4 after PREP, 5 after PP DET, 6 after PP ADJ, 7 after PP NOUN, 8 after
    VERB, 9 after object DET, 10 after object ADJ, 11 after object NOUN,
    12 after TAG."""
    out, ph, subj = [], 0, None
    if CHAIN:
        subj = CLASS[sentence[0]].split("_")[1]
        return [(i + 1, subj) for i in range(len(sentence))]
    for w in sentence:
        c = CLASS[w]
        base = c.split("_")[0]
        if ph == 0:
            subj = c.split("_")[1]
        nxt = {(0, "DET"): 1, (1, "ADJ"): 2, (1, "NOUN"): 3, (2, "NOUN"): 3,
               (3, "PREP"): 4, (4, "DET"): 5, (5, "ADJ"): 6, (5, "NOUN"): 7, (6, "NOUN"): 7,
               (3, "VERB"): 8, (7, "VERB"): 8, (8, "DET"): 9, (9, "ADJ"): 10, (9, "NOUN"): 11,
               (10, "NOUN"): 11, (11, "TAG"): 12}[(ph, base)]
        out.append((nxt, subj))
        ph = nxt
    return out


def _mrr(te, words, scorer, rng):
    rr, n = 0.0, 0
    for s in te:
        st = states(s)
        for i, (a, truth) in enumerate(zip(s, s[1:])):
            keyed = sorted(((-scorer(a, st[i], w), rng.random(), w) for w in words))
            rr += 1.0 / ([w for _, _, w in keyed].index(truth) + 1)
            n += 1
    return rr / max(n, 1)


def oracle_gap(seed: int, vocab_size: int = 50, n_train: int = N_TRAIN,
               n_test: int = N_TEST, tie_seed: int = 0):
    """(unigram, bigram, phase-only oracle, phase+number oracle) MRR on the
    seed's corpora, estimated from its train sentences."""
    words = vocabulary(vocab_size)
    keep = set(words)
    tr = [[w for w in s if w in keep] for s in generate(n_train, seed)]
    te = [[w for w in s if w in keep] for s in generate(n_test, seed + 500)]
    uni, big, pha, full = Counter(), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for s in tr:
        st = states(s)
        for i, (a, nxt) in enumerate(zip(s, s[1:])):
            uni[nxt] += 1
            big[a][nxt] += 1
            pha[st[i][0]][nxt] += 1
            full[st[i]][nxt] += 1
    rng = random.Random(tie_seed)
    return (_mrr(te, words, lambda a, st, w: uni[w], rng),
            _mrr(te, words, lambda a, st, w: big[a][w], rng),
            _mrr(te, words, lambda a, st, w: pha[st[0]][w], rng),
            _mrr(te, words, lambda a, st, w: full[st][w], rng))


if __name__ == "__main__":
    import numpy as np
    import sys
    if "--chain" in sys.argv:
        gap = int(sys.argv[sys.argv.index("--gap") + 1]) if "--gap" in sys.argv else 1
        use_chain(True, gap=gap)
        print(f"CHAIN variant, gap {gap}")
    rows = [oracle_gap(s) for s in range(42, 62)]
    for name, col in zip(("unigram", "bigram", "phase oracle", "phase+number oracle"), zip(*rows)):
        v = np.array(col)
        print(f"{name:20s} {v.mean():.4f} +/- {1.96 * v.std(ddof=1) / np.sqrt(len(v)):.4f}")
    gap = np.array([r[3] - r[1] for r in rows])
    print(f"oracle - bigram (paired) {gap.mean():+.4f} +/- {1.96 * gap.std(ddof=1) / np.sqrt(len(gap)):.4f}"
          f"   GAP_BAR {GAP_BAR}: {'met' if gap.mean() - 1.96 * gap.std(ddof=1) / np.sqrt(len(gap)) >= GAP_BAR else 'NOT met'}")

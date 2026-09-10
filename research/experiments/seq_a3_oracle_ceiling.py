"""The A3 corpus's ORACLE-STATE ceiling, in the study's own units: MRR of the
true next word with random tie-breaking, per seed's own train/test corpora
(200 / 25 sentences, V = 50), 20 seeds.

Predictors (all estimated from the seed's TRAIN corpus unless 'exact'):
  unigram      P(next)
  bigram       P(next | previous word)         -- the registered optimum
  class-bigram P(next | class of previous)     -- what a bigram over classes sees
  phase        P(next | template PHASE)        -- the oracle state (9 phases)
  phase-exact  the generator's own class distribution given the phase, with
               uniform words within a class -- the true ceiling
"""
import random
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'study4'))
import ntp  # noqa: E402

VOCAB, N_TRAIN, N_TEST = 50, 200, 25
SEEDS = list(range(42, 62))
CLASS = {w: c for c, ws in ntp.WORD_CLASSES.items() for w in ws}


def phases(sentence):
    """The template phase BEFORE each word (the state the machine is in when
    the word arrives), and the phase AFTER it. Phases: 0 start; 1 after
    DET1; 2 after ADJ1; 3 after NOUN1; 4 after VERB; 5 after PREP; 6 after
    DET(after PREP); 7 after DET(after VERB); 8 after ADJ2; 9 after NOUN2."""
    out, ph = [], 0
    for w in sentence:
        c = CLASS[w]
        nxt = {(0, 'DET'): 1, (1, 'ADJ'): 2, (1, 'NOUN'): 3, (2, 'NOUN'): 3,
               (3, 'VERB'): 4, (4, 'PREP'): 5, (4, 'DET'): 7, (5, 'DET'): 6,
               (6, 'NOUN'): 9, (7, 'ADJ'): 8, (7, 'NOUN'): 9, (8, 'NOUN'): 9}[(ph, c)]
        out.append(nxt)          # the state after reading w = the context for predicting the next word
        ph = nxt
    return out


def exact_next_class(ph):
    return {1: {'ADJ': .5, 'NOUN': .5}, 2: {'NOUN': 1}, 3: {'VERB': 1},
            4: {'PREP': .5, 'DET': .5}, 5: {'DET': 1}, 6: {'NOUN': 1},
            7: {'ADJ': .5, 'NOUN': .5}, 8: {'NOUN': 1}}[ph]


def mrr(te, scorer, rng):
    rr, n = 0.0, 0
    for s in te:
        ph = phases(s)
        for i, (a, truth) in enumerate(zip(s, s[1:])):
            keyed = sorted(((-scorer(a, ph[i], w), rng.random(), w) for w in words))
            rr += 1.0 / ([w for _, _, w in keyed].index(truth) + 1)
            n += 1
    return rr / n


res = defaultdict(list)
for seed in SEEDS:
    words = ntp.vocabulary(VOCAB)
    keep = set(words)
    tr = [[w for w in s if w in keep] for s in ntp.generate(N_TRAIN, seed)]
    te = [[w for w in s if w in keep] for s in ntp.generate(N_TEST, seed + 500)]
    uni, big, cbig, pha = Counter(), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for s in tr:
        ph = phases(s)
        for i, (a, nxt) in enumerate(zip(s, s[1:])):
            uni[nxt] += 1; big[a][nxt] += 1; cbig[CLASS[a]][nxt] += 1; pha[ph[i]][nxt] += 1
    rng = random.Random(0)
    res['unigram'].append(mrr(te, lambda a, p, w: uni[w], rng))
    res['bigram'].append(mrr(te, lambda a, p, w: big[a][w], rng))
    res['class-bigram'].append(mrr(te, lambda a, p, w: cbig[CLASS[a]][w], rng))
    res['phase'].append(mrr(te, lambda a, p, w: pha[p][w], rng))
    res['phase-exact'].append(mrr(te, lambda a, p, w: exact_next_class(p).get(CLASS[w], 0.0) / len(ntp.WORD_CLASSES[CLASS[w]]), rng))
for k, v in res.items():
    v = np.array(v); ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    print(f'{k:13s} {v.mean():.4f} +/- {ci:.4f}   (min {v.min():.4f} max {v.max():.4f})')
d = np.array(res['phase']) - np.array(res['bigram'])
print(f'phase - bigram (paired): {d.mean():+.4f} +/- {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.4f}')

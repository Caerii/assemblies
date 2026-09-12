"""Can ONE shared lexical area recover word category from grounding alone?

WHY THIS EXISTS.  The parser routes every word to one of eight core areas by
``GroundingContext.dominant_modality`` -- an annotation.  Nouns and verbs
therefore cannot overlap even in principle, so "category" is architectural,
not learned.  This harness removes the routing: every word is built in a
SINGLE area, and category is read off the overlap structure that results.

THE DEGENERACY THAT ALMOST FAKED A RESULT.  Grounding features are named
``visual:ANIMATE``, ``motor:TRANSITIVE`` -- modality-prefixed.  The category
IS the dominant modality, so every feature is category-pure and the measured
cross-category feature Jaccard is exactly 0.0000: no noun shares a single
feature with any verb.  Run as-is, this experiment scores ~1.00 and measures
a one-hot encoding of the answer.  That is the PREFIXED arm below, kept as a
positive control precisely because it must score high for trivial reasons.

The real test is the STRIPPED arm.  Dropping the modality prefix makes
features collide across categories -- ``properties:QUALITY`` (adjectives) and
``motor:QUALITY`` (verbs) become the same stimulus -- so a word's neighbours
are no longer guaranteed to be category-mates.  PERMUTED shuffles the feature
sets across words and is the null.

PROTOCOL NOTES.  Feed-forward only: recurrence during training collapses
shared areas (see brain.py's note on `recurrent_projection`), and the
feed-forward build has no measured lexicon ceiling.  ``n`` is set from the
critical load alpha* ~ 1.15, i.e. n >= M*k/1.15, so a null result cannot be
an over-capacity artifact.  Each arm re-seeds numpy/random per trial because
Brain(seed=) alone does not determine the run.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics as st
from collections import Counter
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

os.environ.setdefault("PYTHONHASHSEED", "0")

from neural_assemblies.core.brain import Brain

MODALITIES = ("visual", "motor", "properties", "spatial",
              "social", "temporal", "emotional")

LEX = "LEX_ALL"


# ----------------------------------------------------------------- data

def load_vocabulary(preset: str) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """Return (word -> prefixed feature set, word -> true category)."""
    from neural_assemblies.assembly_calculus.emergent.vocabulary_builder import (
        build_vocabulary_preset,
    )
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        GROUNDING_TO_CORE,
    )

    vocab = build_vocabulary_preset(preset)
    feats: Dict[str, Set[str]] = {}
    cats: Dict[str, str] = {}
    for word, ctx in vocab.items():
        fs = {f"{m}:{v}" for m in MODALITIES for v in getattr(ctx, m)}
        if not fs:
            continue                      # ungrounded: nothing to cluster on
        feats[word] = fs
        cats[word] = GROUNDING_TO_CORE.get(ctx.dominant_modality, "DET_CORE")
    return feats, cats


def apply_regime(feats: Dict[str, Set[str]], regime: str, rng: random.Random
                 ) -> Dict[str, Set[str]]:
    """PREFIXED keeps modality; STRIPPED drops it; PERMUTED shuffles words."""
    if regime == "prefixed":
        return {w: set(fs) for w, fs in feats.items()}
    if regime == "stripped":
        return {w: {f.split(":", 1)[1] for f in fs} for w, fs in feats.items()}
    if regime == "permuted":
        words = sorted(feats)
        sets = [set(feats[w]) for w in words]
        rng.shuffle(sets)
        return dict(zip(words, sets))
    raise ValueError(regime)


# ------------------------------------------------------------ the build

def build_shared_area(feats: Dict[str, Set[str]], *, n: int, k: int,
                      p: float, beta: float, rounds: int, seed: int
                      ) -> Dict[str, np.ndarray]:
    """Build every word in ONE area, feed-forward, and return its assembly."""
    np.random.seed(seed)  # pyright: ignore[reportAttributeAccessIssue]
    random.seed(seed)

    brain = Brain(p=p, seed=seed, save_winners=True, norm_init=True,
                  engine="numpy_sparse")
    brain.add_area(LEX, n, k, beta)

    all_feats = sorted({f for fs in feats.values() for f in fs})
    for f in all_feats:
        brain.add_stimulus(f"feat_{f}", k)
    for w in sorted(feats):
        brain.add_stimulus(f"phon_{w}", k)

    assemblies: Dict[str, np.ndarray] = {}
    for w in sorted(feats):
        brain.inhibit_areas([LEX])                 # clear winners, keep weights
        stim = {f"phon_{w}": [LEX]}
        for f in feats[w]:
            stim[f"feat_{f}"] = [LEX]
        for _ in range(rounds):                    # stimulus-anchored, no recurrence
            brain.project(stim, {})
        assemblies[w] = np.sort(np.asarray(brain.areas[LEX].winners).copy())
    return assemblies


# -------------------------------------------------------------- metrics

def _reference_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of the reference cap ``a`` retained by ``b`` (directional)."""
    return len(np.intersect1d(a, b, assume_unique=True)) / max(1, len(a))


def score(assemblies: Dict[str, np.ndarray], cats: Dict[str, str]
          ) -> Dict[str, float]:
    """Rank-1 nearest-neighbour category accuracy + overlap separation."""
    words = sorted(assemblies)
    m = len(words)
    ov = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            o = _reference_overlap(assemblies[words[i]], assemblies[words[j]])
            ov[i, j] = ov[j, i] = o

    correct = 0
    for i, w in enumerate(words):
        row = ov[i].copy()
        row[i] = -1.0
        if cats[words[int(np.argmax(row))]] == cats[w]:
            correct += 1

    within = [ov[i, j] for i in range(m) for j in range(i + 1, m)
              if cats[words[i]] == cats[words[j]]]
    between = [ov[i, j] for i in range(m) for j in range(i + 1, m)
               if cats[words[i]] != cats[words[j]]]

    # The baseline for a NEAREST-NEIGHBOUR score is the collision probability
    # sum(p_c^2) -- the chance that a randomly drawn neighbour shares your
    # class -- NOT the majority class frequency, which is the baseline for a
    # constant classifier. They differ a lot here (0.235 vs 0.336), and the
    # PERMUTED null lands on the former.
    counts = Counter(cats[w] for w in words)
    majority = counts.most_common(1)[0][1] / m
    nn_chance = sum((c / m) ** 2 for c in counts.values())

    return {
        "nn_accuracy": correct / m,
        "majority_baseline": majority,
        "nn_chance": nn_chance,
        "within": st.mean(within) if within else float("nan"),
        "between": st.mean(between) if between else float("nan"),
        "n_words": float(m),
    }


# ------------------------------------------------------------------ run

def run(regimes: Sequence[str], *, preset: str, k: int, p: float,
        beta: float, rounds: int, seeds: Sequence[int],
        n_override: int | None) -> None:
    feats, cats = load_vocabulary(preset)
    m = len(feats)
    n = n_override or max(2000, int(m * k / 1.15) + k * 20)
    print(f"vocabulary={m} grounded words   n={n} k={k} p={p} beta={beta} "
          f"rounds={rounds}   alpha={m * k / n:.3f} (critical ~1.15)")

    for regime in regimes:
        rows: List[Dict[str, float]] = []
        for seed in seeds:
            fs = apply_regime(feats, regime, random.Random(seed))
            asm = build_shared_area(fs, n=n, k=k, p=p, beta=beta,
                                    rounds=rounds, seed=seed)
            rows.append(score(asm, cats))
        acc = [r["nn_accuracy"] for r in rows]
        wi = [r["within"] for r in rows]
        be = [r["between"] for r in rows]
        sd = st.stdev(acc) if len(acc) > 1 else 0.0
        print(f"\n{regime.upper():<9} nn-acc {st.mean(acc):.4f} +/- {sd:.4f}"
              f"   (nn-chance {rows[0]['nn_chance']:.4f}, "
              f"majority {rows[0]['majority_baseline']:.4f})")
        print(f"          overlap within {st.mean(wi):.4f}  "
              f"between {st.mean(be):.4f}  "
              f"ratio {st.mean(wi) / max(1e-9, st.mean(be)):.2f}x")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--regimes", nargs="+",
                    default=["prefixed", "stripped", "permuted"])
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    run(a.regimes, preset=a.preset, k=a.k, p=a.p, beta=a.beta,
        rounds=a.rounds, seeds=a.seeds, n_override=a.n)


if __name__ == "__main__":
    main()

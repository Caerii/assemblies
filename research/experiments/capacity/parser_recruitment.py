"""
Recruitment vs reuse inside the real EmergentParser lexical core areas.

Confirms (or refutes) the synthetic capacity result from
``lexicon_capacity.py`` in the actual 44-area parser, and pins down what the
motivating measurement -- NOUN_CORE w=1808/3000 at n=3000, k=30 after a
TWO_WORD run -- is actually made of.

For every core area A and every word w trained into it (in training order):

    recruited(w) = | assembly(w) \\ U_{w' before w} assembly(w') |

i.e. how many of the word's k neurons had never appeared in ANY earlier
word's stored assembly in that area. Reuse is the complement. Alongside that
we record the area's ever-fired count `w` (materialised neurons), so

    tiling_ratio = w / (n_words * k)

is 1.0 when the area is tiled with no sharing at all and falls toward 0 as
words start to share neurons.

Consequences measured after training:
  * pairwise overlap among the area's stored lexical assemblies (null: k/n)
  * retrieval: re-present each word's phonological stimulus with plasticity
    off; self-overlap with the stored assembly, and whether the nearest
    stored assembly is the right word (null: 1 / n_words)
  * both split by training-order quartile, so catastrophic forgetting of
    early-learned words would show up as a Q1 << Q4 gradient

Sweep: n x vocabulary-preset size, several seeds.

Usage:
    python -m research.experiments.capacity.parser_recruitment
    python -m research.experiments.capacity.parser_recruitment --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")


def _win(a):
    """Accept an Assembly snapshot or a raw winner array."""
    return np.asarray(getattr(a, "winners", a))


def _overlap(a, b) -> float:
    sa, sb = set(_win(a).tolist()), set(_win(b).tolist())
    m = min(len(sa), len(sb))
    return len(sa & sb) / m if m else 0.0


def _cosine(a, b) -> float:
    sa, sb = set(_win(a).tolist()), set(_win(b).tolist())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(np.sqrt(len(sa) * len(sb)))


def _quartile_means(values: Sequence[float]) -> List[float]:
    v = np.asarray(values, dtype=float)
    if len(v) < 4:
        return [float(np.mean(v))] * 4 if len(v) else [float("nan")] * 4
    q = np.array_split(v, 4)
    return [float(np.mean(x)) for x in q]


def analyse_area(parser, core_area: str, k: int, n: int) -> Dict:
    """Recruitment / reuse / interference for one core lexical area."""
    lexicon = parser.core_lexicons.get(core_area, {})
    words = list(lexicon.keys())          # dict preserves training order
    if len(words) < 2:
        return {"area": core_area, "n_words": len(words), "skipped": True}

    asms = [_win(lexicon[w]) for w in words]

    seen: set = set()
    recruit_frac, recruit_null = [], []
    for a in asms:
        s = set(a.tolist())
        new = len(s - seen)
        recruit_frac.append(new / max(1, len(s)))
        # Random-tiling null: an unbiased winner has never been seen with
        # probability 1 - |seen|/n.
        recruit_null.append(1.0 - len(seen) / n)
        seen |= s

    area_w = int(parser.brain.areas[core_area].w)

    # Pairwise overlap among stored assemblies (subsample if large)
    rng = np.random.default_rng(0)
    pairs = [(i, j) for i in range(len(asms)) for j in range(i + 1, len(asms))]
    if len(pairs) > 4000:
        pairs = [pairs[t] for t in rng.choice(len(pairs), 4000, replace=False)]
    pw = np.array([_overlap(asms[i], asms[j]) for i, j in pairs]) if pairs \
        else np.array([0.0])

    # Retrieval probe: re-present each word's phon stimulus, plasticity off.
    from neural_assemblies.assembly_calculus.emergent.parser_mixins.core import _snap
    prev = parser.brain.disable_plasticity
    parser.brain.disable_plasticity = True
    self_ov, ident = [], []
    probe_failures = 0
    try:
        for i, word in enumerate(words):
            phon = parser.stim_map.get(word)
            if phon is None:
                continue
            try:
                parser.brain._engine.reset_area_connections(core_area)
                stim = {phon: [core_area]}
                parser.brain.project(stim, {})
                if parser.rounds > 1:
                    parser.brain.project_rounds(
                        target=core_area, areas_by_stim=stim,
                        dst_areas_by_src_area={core_area: [core_area]},
                        rounds=parser.rounds - 1,
                    )
            except RuntimeError as exc:
                # Retrieval itself can demand fresh neurons; in a saturated
                # area there are none left. Count it and move on -- an
                # unprobeable area is itself a capacity result.
                if "too small to sample" not in str(exc):
                    raise
                probe_failures += 1
                continue
            got = _win(_snap(parser.brain, core_area))
            self_ov.append(_overlap(got, asms[i]))
            scores = np.array([_cosine(got, a) for a in asms])
            ident.append(1.0 if int(np.argmax(scores)) == i else 0.0)
    finally:
        parser.brain.disable_plasticity = prev

    return {
        "area": core_area,
        "skipped": False,
        "n_words": len(words),
        "k": k,
        "n": n,
        "w": area_w,
        "w_over_n": area_w / n,
        "words_times_k_over_n": len(words) * k / n,
        "tiling_ratio": area_w / max(1, len(words) * k),
        "recruit_frac_mean": float(np.mean(recruit_frac)),
        "recruit_frac_q": _quartile_means(recruit_frac),
        "recruit_null_q": _quartile_means(recruit_null),
        "pairwise_overlap_mean": float(np.mean(pw)),
        "pairwise_overlap_p95": float(np.percentile(pw, 95)),
        "chance_overlap": k / n,
        "retrieval_overlap_mean": float(np.mean(self_ov)) if self_ov else float("nan"),
        "retrieval_overlap_q": _quartile_means(self_ov),
        "identification_acc": float(np.mean(ident)) if ident else float("nan"),
        "identification_acc_q": _quartile_means(ident),
        "identification_chance": 1.0 / len(words),
        "probe_failures": probe_failures,
        "probes_ok": len(self_ov),
    }


def run_cell(n: int, k: int, preset: str, seed: int, depth: str = "LEXICON") -> Dict:
    """Train a parser and analyse every core lexical area.

    depth="LEXICON" trains the WHOLE vocabulary into the core areas via
    ``train_lexicon()``. This is the right probe for a vocabulary-size sweep:
    the curriculum's TWO_WORD stage uses a fixed ~30-word sentence list, so
    growing the vocabulary preset does not grow its lexicon at all.
    depth="TWO_WORD" reproduces the motivating measurement instead.
    """
    from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (
        train_parser_to_depth,
    )
    from neural_assemblies.assembly_calculus.emergent.parser import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.vocabulary_builder import (
        build_vocabulary_preset,
    )
    from neural_assemblies.assembly_calculus.emergent.core.areas import CORE_AREAS

    vocab = build_vocabulary_preset(preset)
    t0 = time.perf_counter()
    exhausted = None
    if depth == "LEXICON":
        parser = EmergentParser(n=n, k=k, seed=seed, vocabulary=vocab,
                                fast_training=True)
        try:
            parser.train_lexicon()
        except RuntimeError as exc:
            # HARD CAPACITY LIMIT. The sparse engine refuses to materialise
            # another assembly once fewer than k neurons in the area have
            # never fired: "Remaining size of area too small to sample k new
            # winners". This is not a crash to be worked around -- it is the
            # measurement. Record where it happened and analyse what was
            # learned up to that point.
            if "too small to sample" not in str(exc):
                raise
            exhausted = str(exc)
    else:
        parser = train_parser_to_depth(depth, n=n, k=k, seed=seed,
                                       vocabulary=vocab)
    train_s = time.perf_counter() - t0

    areas = [analyse_area(parser, a, k, n) for a in CORE_AREAS]
    return {
        "n": n, "k": k, "preset": preset, "vocab_size": len(vocab),
        "pool_exhausted": exhausted is not None,
        "exhaustion_message": exhausted,
        "total_words_trained": sum(len(parser.core_lexicons.get(a, {}))
                                   for a in CORE_AREAS),
        "seed": seed, "depth": depth, "train_seconds": train_s,
        "areas": [a for a in areas if not a.get("skipped")],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.quick:
        cells = [(1000, 30, "core", "LEXICON"), (1000, 30, "medium", "LEXICON")]
        seeds = (42,)
    else:
        cells = [(n, 30, pre, "LEXICON")
                 for n in (1000, 3000, 10000)
                 for pre in ("core", "medium", "large")]
        # Reference point: the motivating TWO_WORD curriculum measurement.
        cells += [(3000, 30, "medium", "TWO_WORD")]
        seeds = (42, 43, 44)

    results = []
    t0 = time.perf_counter()
    for (n, k, preset, depth) in cells:
        for s in seeds:
            t = time.perf_counter()
            try:
                r = run_cell(n, k, preset, s, depth=depth)
            except Exception as exc:            # noqa: BLE001
                print(f"  n={n} preset={preset} seed={s} FAILED: {exc!r}",
                      flush=True)
                results.append({"n": n, "k": k, "preset": preset, "seed": s,
                                "depth": depth, "error": repr(exc)})
                continue
            results.append(r)
            tops = {a["area"]: round(a["w_over_n"], 3) for a in r["areas"][:4]}
            print(f"  n={n} {depth} preset={preset}({r['vocab_size']}w) seed={s} "
                  f"trained={r['total_words_trained']}"
                  f"{' EXHAUSTED' if r['pool_exhausted'] else ''} {tops} "
                  f"({time.perf_counter()-t:.1f}s)", flush=True)

    print(f"done in {time.perf_counter()-t0:.1f}s")
    out = Path(args.out) if args.out else (
        Path(__file__).parent / ("results_parser_quick.json" if args.quick
                                 else "results_parser_recruitment.json"))
    out.write_text(json.dumps(results, indent=1, default=float))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

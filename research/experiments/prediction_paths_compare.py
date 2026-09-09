"""CONTEXT-buffer vs bounded-state next-token prediction -- the comparison
``state_prediction.py``'s docstring invites and that had never been run.

The claim under test (from that docstring): accumulating every word of a prefix
into one running CONTEXT assembly is not in the published NEMO model and has a
structural defect -- one area of k winners cannot encode an unbounded prefix, so
the area saturates and successive prefixes map to near-identical assemblies. The
paper-faithful alternative carries sequence in *weights between bounded states*
(core category x syntactic slot x mood), so nothing grows with prefix length.

Design
------
* Vocabulary/lexicon training is shared (that is word acquisition, not
  prediction), so both paths see the same words.
* PREDICTION training is SPLIT: fit on a training subset, score on held-out
  sentences. Two independent parsers, same seed -- the two paths both write into
  PREDICTION's incoming synapses, so training both on one brain would let them
  contaminate each other.
* Metrics: top-1 / top-3 / MRR overall, then broken out BY PREFIX LENGTH, which
  is where the saturation claim makes a differential prediction.
* Mechanism probe, independent of accuracy: how many DISTINCT assemblies does
  each representation produce across all prefixes, and how many distinct neurons
  does CONTEXT ever recruit? A saturating buffer collapses distinct prefixes onto
  the same assembly; a bounded state should not (its capacity is combinatorial in
  synapses, not in one area's neurons).
* Substrate drift: `brain.frozen()` stops plasticity but NOT materialization, so
  we also report how much each area grew during scoring. A path that mutates the
  substrate while being scored is not producing a clean held-out number.
"""

from collections import defaultdict

from neural_assemblies.assembly_calculus.emergent.parser import EmergentParser
from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
    create_training_sentences,
)
from neural_assemblies.assembly_calculus.emergent.core.areas import (
    CONTEXT, MOOD, OBJ, PREDICTION, SUBJ,
)
from neural_assemblies.assembly_calculus.ops import _snap

N, K, P, BETA, SEED, ROUNDS = 1000, 50, 0.05, 0.1, 42, 10


def build_parser():
    p = EmergentParser(n=N, k=K, p=P, beta=BETA, seed=SEED, rounds=ROUNDS)
    p.train(create_training_sentences())          # shared vocabulary acquisition
    return p


def prediction_points(sentences, stim_map):
    pts = []
    for s in sentences:
        words = [w for w in s.words if w in stim_map]
        for i in range(len(words) - 1):
            pts.append((words[:i + 1], words[i + 1]))
    return pts


def score(rank_fn, points):
    """top1/top3/MRR overall and bucketed by prefix length."""
    agg = {"t1": 0, "t3": 0, "rr": 0.0, "n": 0}
    by_len = defaultdict(lambda: {"t1": 0, "n": 0})
    for prefix, actual in points:
        ranked = [w for w, _ in rank_fn(prefix)]
        agg["n"] += 1
        by_len[len(prefix)]["n"] += 1
        if ranked[:1] == [actual]:
            agg["t1"] += 1
            by_len[len(prefix)]["t1"] += 1
        if actual in ranked[:3]:
            agg["t3"] += 1
        if actual in ranked:
            agg["rr"] += 1.0 / (ranked.index(actual) + 1)
    n = max(agg["n"], 1)
    return (
        {"top1": agg["t1"] / n, "top3": agg["t3"] / n, "mrr": agg["rr"] / n,
         "n": agg["n"]},
        {L: v["t1"] / max(v["n"], 1) for L, v in sorted(by_len.items())},
    )


def distinct_context_assemblies(parser, points):
    """How many DISTINCT CONTEXT assemblies across prefixes, and how many
    distinct neurons does CONTEXT ever recruit? (the saturation claim)"""
    seen, neurons = set(), set()
    for prefix, _ in points:
        parser.build_context_incremental(prefix, reset=True, direct=True)
        asm = _snap(parser.brain, CONTEXT)
        ids = frozenset(int(x) for x in asm.winners)
        seen.add(ids)
        neurons |= ids
    return len(seen), len(neurons)


def distinct_states(parser, points):
    """How many DISTINCT bounded states (core x syn x mood) across prefixes?"""
    seen, neurons = set(), set()
    for prefix, _ in points:
        srcs = parser._set_state(prefix, len(prefix))
        if srcs is None:
            continue
        key = []
        for area in sorted(srcs):
            ids = frozenset(
                int(x) for x in parser.brain.areas[area].winners)
            key.append((area, ids))
            neurons |= {(area, i) for i in ids}
        seen.add(tuple(key))
    return len(seen), len(neurons)


def area_sizes(parser, areas):
    """Recruitment per area -- neurons that have ever fired.

    `get_num_ever_fired()`, not `.w`: the parser assigns `area.winners`
    directly when restoring outer state, and that setter clobbers `w` to
    `len(winners)`. Both arms would be clobbered alike so the COMPARISON
    survived, but the absolute numbers did not. See
    research/notes/substrate/w_alias_back_catalogue.md.
    """
    return {a: parser.brain.areas[a].get_num_ever_fired()
            for a in areas if a in parser.brain.areas}


def main():
    sentences = create_training_sentences()
    split = int(len(sentences) * 0.8)
    train_s, test_s = sentences[:split], sentences[split:]
    print(f"corpus {len(sentences)} sentences -> train {len(train_s)} / "
          f"held-out {len(test_s)}")

    # ---------------- CONTEXT-buffer path ----------------
    ctx_parser = build_parser()
    ctx_parser.train_next_token(train_s)
    ctx_pts_tr = prediction_points(train_s, ctx_parser.stim_map)
    ctx_pts_te = prediction_points(test_s, ctx_parser.stim_map)
    before = area_sizes(ctx_parser, [CONTEXT, PREDICTION])
    ctx_te, ctx_te_len = score(lambda pre: ctx_parser.predict_next(pre), ctx_pts_te)
    ctx_tr, _ = score(lambda pre: ctx_parser.predict_next(pre), ctx_pts_tr)
    after = area_sizes(ctx_parser, [CONTEXT, PREDICTION])
    ctx_drift = {a: after[a] - before[a] for a in before}
    ctx_distinct, ctx_neurons = distinct_context_assemblies(ctx_parser, ctx_pts_tr + ctx_pts_te)

    # ---------------- bounded-state path ----------------
    st_parser = build_parser()
    st_parser.train_next_token_state(train_s)
    st_pts_tr = prediction_points(train_s, st_parser.stim_map)
    st_pts_te = prediction_points(test_s, st_parser.stim_map)
    before = area_sizes(st_parser, [SUBJ, OBJ, MOOD, PREDICTION])
    st_te, st_te_len = score(
        lambda pre: st_parser.predict_next_state(pre, top_k=100), st_pts_te)
    st_tr, _ = score(
        lambda pre: st_parser.predict_next_state(pre, top_k=100), st_pts_tr)
    after = area_sizes(st_parser, [SUBJ, OBJ, MOOD, PREDICTION])
    st_drift = {a: after[a] - before[a] for a in before}
    st_distinct, st_neurons = distinct_states(st_parser, st_pts_tr + st_pts_te)

    V = len(ctx_parser.stim_map)
    n_pts = len(ctx_pts_tr) + len(ctx_pts_te)
    print(f"vocab={V} chance={1/V:.3f}  prediction points: "
          f"train={len(ctx_pts_tr)} held-out={len(ctx_pts_te)}\n")

    print(f"{'path':<16} {'train t1':>9} {'held t1':>8} {'held t3':>8} {'held MRR':>9}")
    print("-" * 56)
    print(f"{'CONTEXT buffer':<16} {ctx_tr['top1']:>9.3f} {ctx_te['top1']:>8.3f} "
          f"{ctx_te['top3']:>8.3f} {ctx_te['mrr']:>9.3f}")
    print(f"{'bounded state':<16} {st_tr['top1']:>9.3f} {st_te['top1']:>8.3f} "
          f"{st_te['top3']:>8.3f} {st_te['mrr']:>9.3f}")

    print(f"\nheld-out top-1 by prefix length (saturation should hurt the buffer"
          f" as length grows):")
    lens = sorted(set(ctx_te_len) | set(st_te_len))
    print("  len:      " + "".join(f"{L:>7}" for L in lens))
    print("  buffer:   " + "".join(f"{ctx_te_len.get(L, float('nan')):>7.2f}" for L in lens))
    print("  state:    " + "".join(f"{st_te_len.get(L, float('nan')):>7.2f}" for L in lens))

    print(f"\nrepresentation capacity over all {n_pts} prefixes:")
    print(f"  CONTEXT: {ctx_distinct:>3} distinct assemblies, "
          f"{ctx_neurons} distinct neurons ever recruited (k={K}, n={N})")
    print(f"  state:   {st_distinct:>3} distinct states")

    print(f"\nsubstrate drift during scoring (frozen() stops plasticity, "
          f"NOT materialization):")
    print(f"  buffer: {ctx_drift}")
    print(f"  state:  {st_drift}")


if __name__ == "__main__":
    main()

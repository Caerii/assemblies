"""PHASE B: bound the semantic drive share, and pay for it in generalisation.

THE MECHANISM, measured in parser_drive_share_alpha.py. `apply_lexicon_word`
fires phon PLUS one stimulus per grounding feature, simultaneously, all of size
k. So word identity is 1 of (1+F) equal drivers -- share 0.20-0.33 for F = 2..4
-- and two words sharing all their features have input overlap

    alpha_eff = 2F / (2 + 2F) = 0.667 at F = 2

which the substrate law maps to assembly overlap ~1.0. That IS the duplicate
clusters ({death fear life love ...}, {face foot head mouth nose}).

`phon_weight` (W) makes phon contribute W*k, so identity's share is W/(W+F) and
the worst-pair overlap becomes 2F/(2W+2F). W = 6 puts it under 0.25 at F = 2,
which is where the substrate still preserves overlap at beta = 0.05.

WHY THIS IS THE PAPERS' CHANGE IN THE RELEVANT SENSE. Mitropolsky 2025 delivers
semantics from a BOUNDED set of areas (VISUAL, MOTOR, context areas C_i), each
contributing one assembly. We deliver it as an UNBOUNDED set of stimuli, one per
feature, so semantic drive grows with feature count. Raising W is the cheap test
of that same mechanism -- it does not restructure the pathway, it re-weights it,
and if the re-weighting does nothing then restructuring is unlikely to either.
(Note the premise correction in 77f8f4e: NOUN_CORE/VERB_CORE/ADJ_CORE/ADV_CORE
already IS the papers' LEX1/LEX2 split, so that was never the difference.)

THE COST, AND WHY IT IS IN THE FIRST TABLE RATHER THAN A FOLLOW-UP. Grounding
drive is what lets a word be placed from its features -- the generalisation
pathway, and the entire point of grounded lexicon learning. A W large enough to
guarantee distinctness can make the representation phon-only, which scores
beautifully on every distinctness metric and has destroyed what the grounding
was for. This project has shipped that shape before (`distinctness is not
information`; the Phase A degenerate arm), so the cost metric runs in the same
loop as the benefit, not afterwards.

`grounding_recall` is that metric: fire ONLY the word's grounding stimuli into
the core area, read what the area settles on, and rank it against the stored
core assemblies among 6 candidates. It answers "can semantics alone still find
this word", which is what a novel word with known features needs. Chance is
1/6 = 0.167.

READING, pre-registered:
  * high-alpha overlap falls AND ret@6 holds AND grounding_recall holds
        -> adopt that W; the drive share was the defect.
  * high-alpha overlap falls AND grounding_recall COLLAPSES
        -> the interior optimum is below that W. Report the curve and the
           trade-off, NOT the distinctness gain on its own.
  * nothing moves
        -> the drive share is not the lever, and restructuring the pathway into
           bounded semantic areas is unlikely to help either. Say so.

CACHE. `phon_weight` is cache-key material (it enters `ParserCache` params and
the backbone filename digest). Without that, every W would have loaded W=1's
pickle and this sweep would have printed a flat table -- the exact failure
cf6cb0f fixed for beta.
"""
import os
import statistics
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

import numpy as np                                                     # noqa: E402

from _substrate import check_distinct, read, similarity                # noqa: E402
from core_assembly_duplicates import _clusters, _retrieval_over_pool   # noqa: E402
from neural_assemblies.assembly_calculus.emergent.core.areas import (  # noqa: E402
    NOUN_CORE, ROLE_AGENT, ROLE_PATIENT,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (  # noqa: E402
    train_parser_to_depth,
)

DEPTH = "SENTENCES"
SEEDS = [11, 42]
WEIGHTS = [1.0, 3.0, 6.0]
BETAS = [0.10, 0.05]
ROLE_AREAS = (ROLE_PATIENT, ROLE_AGENT)
CANDIDATES = 6
HI_ALPHA = 0.35


def _grounding_recall(parser, rng):
    """Can GROUNDING ALONE still find the word? The generalisation pathway."""
    brain = parser.brain
    core = parser.core_lexicons.get(NOUN_CORE, {})
    words = sorted(core)
    if len(words) < CANDIDATES + 1:
        return float("nan")
    stored = {w: np.asarray(core[w].winners, dtype=np.int64) for w in words}
    live = {}
    for w in words:
        gs = list(parser._grounding_stim_names(parser.word_grounding[w]))
        if not gs:
            continue
        with brain.read_only():
            parser._clear_core_activity(NOUN_CORE)
            stim = {g: [NOUN_CORE] for g in gs}
            brain.project(stim, {})
            if parser.rounds > 1:
                brain.project_rounds(
                    target=NOUN_CORE, areas_by_stim=stim,
                    dst_areas_by_src_area={NOUN_CORE: [NOUN_CORE]},
                    rounds=parser.rounds - 1,
                )
            live[w] = read(brain, NOUN_CORE)
    usable = sorted(live)
    if len(usable) < CANDIDATES + 1:
        return float("nan")
    hits = total = 0
    for _ in range(40):
        subset = list(rng.choice(usable, size=CANDIDATES, replace=False))
        for w in subset:
            best = max(((similarity(live[w], stored[o]), o) for o in subset))[1]
            hits += int(best == w)
            total += 1
    return hits / total if total else float("nan")


def measure(parser, seed):
    core = parser.core_lexicons.get(NOUN_CORE, {})
    area = parser.brain.areas[NOUN_CORE]
    n, k = int(area.n), int(area.k)
    words = sorted(core)
    asm = {w: np.asarray(core[w].winners, dtype=np.int64) for w in words}
    gstim = {w: frozenset(parser._grounding_stim_names(parser.word_grounding[w]))
             for w in words}

    spread, note = check_distinct(list(asm.values()), n, k, where="NOUN_CORE")
    groups = _clusters(core)
    dup = {w for ws in groups.values() if len(ws) > 1 for w in ws}

    W = float(parser.phon_weight)
    hi = []
    for a, b in combinations(words, 2):
        fa, fb = gstim[a], gstim[b]
        eff = (2 * len(fa & fb) * 1.0) / ((W + len(fa)) + (W + len(fb)))
        if eff >= HI_ALPHA:
            hi.append(similarity(asm[a], asm[b]))

    out = {
        "M": len(words),
        "distinct": len(groups) / max(len(words), 1),
        "dup": len(dup) / max(len(words), 1),
        "spread": spread,
        "note": note,
        "hi_ov": statistics.fmean(hi) if hi else float("nan"),
        "n_hi": len(hi),
        "ground": _grounding_recall(parser, np.random.default_rng(seed)),
    }
    for ra in ROLE_AREAS:
        rl = parser.role_lexicons.get(ra, {})
        pool = sorted(w for w in rl if w in core)
        res = _retrieval_over_pool(parser, ra, pool, np.random.default_rng(seed))
        out[f"ret_{ra}"] = (res[0] / res[1]) if res else float("nan")
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("PHASE B -- raise word identity's drive share, and measure what it costs.")
    print(f"depth={DEPTH} seeds={SEEDS} W={WEIGHTS} betas={BETAS} "
          f"chance@{CANDIDATES}={1 / CANDIDATES:.3f}")
    print("identity share = W/(W+F); worst-pair alpha_eff = 2F/(2W+2F), F=2..4")
    print()
    hdr = (f"{'beta':>5} {'W':>4} {'share':>7} {'distinct':>9} {'dup':>6} "
           f"{'spread':>8} {'hi_ov':>7} {'n_hi':>5} {'retP':>6} {'retA':>6} "
           f"{'GROUND':>7}")
    print(hdr)
    print("-" * len(hdr))
    table = {}
    for beta in BETAS:
        for W in WEIGHTS:
            rows = []
            for seed in SEEDS:
                parser = train_parser_to_depth(
                    DEPTH, seed=seed, beta=beta, phon_weight=W)
                rows.append(measure(parser, seed))
            agg = {key: statistics.fmean(r[key] for r in rows)
                   for key in ("distinct", "dup", "spread", "hi_ov", "n_hi",
                               "ground", f"ret_{ROLE_PATIENT}",
                               f"ret_{ROLE_AGENT}")}
            table[(beta, W)] = agg
            share = W / (W + 3.0)     # F=3, the middle of 2..4
            print(f"{beta:>5.2f} {W:>4.1f} {share:>7.3f} {agg['distinct']:>9.3f} "
                  f"{agg['dup']:>6.3f} {agg['spread']:>8.4f} "
                  f"{agg['hi_ov']:>7.4f} {agg['n_hi']:>5.0f} "
                  f"{agg[f'ret_{ROLE_PATIENT}']:>6.3f} "
                  f"{agg[f'ret_{ROLE_AGENT}']:>6.3f} {agg['ground']:>7.3f}")
        print()

    print("=" * len(hdr))
    chance = 1.0 / CANDIDATES
    for beta in BETAS:
        base = table[(beta, 1.0)]
        best, best_W = None, None
        for W in WEIGHTS:
            a = table[(beta, W)]
            ret = statistics.fmean(
                [a[f"ret_{r}"] for r in ROLE_AREAS])
            # A W only counts as an improvement if the GENERALISATION pathway
            # is still alive. Distinctness bought by making the representation
            # phon-only is the degenerate arm this file exists to expose.
            alive = a["ground"] > chance + 0.10
            if alive and (best is None or ret > best):
                best, best_W = ret, W
        b_ret = statistics.fmean([base[f"ret_{r}"] for r in ROLE_AREAS])
        print(f"beta={beta}: baseline W=1 ret={b_ret:.3f} "
              f"ground={base['ground']:.3f}")
        if best_W is None:
            print("   NO W KEEPS GROUNDING ALIVE -- every arm is phon-only. "
                  "The knob trades away exactly what it was meant to preserve.")
        elif best_W == 1.0:
            print("   BEST W IS THE BASELINE. Raising identity's drive share "
                  "does not help; the drive ratio is not the lever, and "
                  "restructuring into bounded semantic areas is unlikely to be "
                  "either.")
        else:
            print(f"   BEST W = {best_W} at ret={best:.3f} "
                  f"(ground={table[(beta, best_W)]['ground']:.3f}) -- an "
                  f"INTERIOR optimum, report it with the trade-off, not alone.")
    print()
    print("GROUND is retrieval from grounding stimuli ALONE. If it falls to")
    print(f"chance ({chance:.3f}) the representation has become phon-only: a")
    print("novel word with known features can no longer be placed, which is the")
    print("entire purpose of grounded lexicon learning.")


if __name__ == "__main__":
    main()

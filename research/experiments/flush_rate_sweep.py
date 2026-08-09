"""E19: the flush-rate sweep -- the homeostatic schedule is a RATE.

PRE-REGISTERED (task #148), after the mechanism (f5d3d33) and before any
swept cell trained.

E18 (c1f195a): both drift and crowding refuted; the deferred-scaling
FLUSH SCHEDULE is the degradation channel (+0.19/+0.27 PL from 8
interim flushes on identical corpora). Mechanism: within-interval
Hebbian mass concentration -- rich-get-richer inside columns -- which
the eventual column normalization cannot undo (scaling preserves
within-column ratios). The timescale law's two walls: slow relative to
fast dynamics (E9: per-update collided with repetition), fast relative
to accumulated mass (E18). Between the walls there is an interior
optimum in EPISODES PER FLUSH, and this sweep locates it while scoring
the FULL exam -- E18 measured the PL side only, so the E-series 0.75
bar is LIVE and undecided.

ARMS: morph_flush_every K in {1, 40, 160, 0} (0 = phase-end only, the
E17 regime; 1 = per-episode, the E9-adjacent extreme; the morph phase
at zipf-400 holds ~1314 episodes, so K=160 approximates E18's 8-chunk
instrument and K=40 probes finer) x zipf-{200, 400} x seeds 42-46
(40 cells), split architecture, overlap readout primary (MI reported).

MECHANISM INSTRUMENTATION (the account must carry its own check): a
spy on engine.flush_synaptic_scaling records, BEFORE each flush, the
column-mass concentration (max/mean of column sums over the pending
columns) of fibers into the NUMBER value areas. The account predicts
concentration GROWS with K and accuracy FALLS with concentration
across arms; if concentration does not order the arms, the account is
wrong regardless of which arm wins.

REGISTERED BARS:
  F1 INTERIOR OPTIMUM: some middle K's attested balanced (overlap)
     exceeds BOTH endpoints (K=1 and K=0), paired per seed at zipf-400
     (where E18's effect was largest).
  F2 THE BAR: best arm's attested balanced >= 0.75 at either budget
     (fixed exam reported alongside; the honest headline is the lower).
  F3 MECHANISM: mean pre-flush concentration is monotone-increasing in
     K, and across the 8 (K, budget) arms rank-correlates NEGATIVELY
     with accuracy.
  GUARDS: K=0 reproduces E17 (0.700 @200 / 0.570 @400 overlap attested
     -- same code path, must match); tense within -0.05 of its K=0
     value in every arm; roles 4/4.

DECISION RULE: F1+F2 -> the arc CLOSES AT THE BAR; adoption unit flips
the deferred-scaling default to mass-paced flushing citing this run.
F1 passes, F2 misses -> the schedule optimum is real but the bar needs
the next corpus size; close with the law and the measured optimum-K
scaling rule. F3 fails -> the concentration account is wrong; the
schedule effect stands (E18) but its WHY reopens with the spy's data
attached.
"""
from __future__ import annotations

import json
import os
import random
import sys

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

SEEDS = list(range(42, 47))
KS = (1, 40, 160, 0)
BUDGETS = (200, 400)
PRE_STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD")
FINAL = "SENTENCES"
N = int(os.environ.get("FR_N", "3000"))
if os.environ.get("FR_SMOKE") == "1":
    SEEDS = [42]
    KS = (160, 0)

OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "flush_rate_sweep_results.json")

E17_REF = {200: 0.700, 400: 0.570}

from overlap_ceiling import spearman  # noqa: E402
from zipf_synthesis import ROLE_PROBES  # noqa: E402


def run_cell(k: int, frames: int, seed: int) -> dict:
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.vocabulary_builder \
        import build_vocabulary_preset
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        TENSE, NUMBER, FEATURE_VALUE_LABELS, feature_value_area,
    )
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        CurriculumTrainer,
    )
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        generation as gen,
    )
    from neural_assemblies.assembly_calculus.emergent.evaluation \
        .morph_features import attested_morph_sets
    from _parallel import forked_parser

    value_number = {feature_value_area(NUMBER, lab)
                    for lab in FEATURE_VALUE_LABELS[NUMBER]}
    conc_log: list = []

    def build_and_train_pre():
        random.seed(seed)
        np.random.seed(seed)
        p = EmergentParser(n=N, k=30, seed=seed,
                           vocabulary=build_vocabulary_preset("core"),
                           fast_training=True)
        ct = CurriculumTrainer(p)
        for stage in PRE_STAGES:
            ct.train_stage(stage)
        return p

    def arm_setup(p):
        p.split_feature_areas = True
        value_areas = frozenset(
            feature_value_area(f, lab)
            for f in (TENSE, NUMBER)
            for lab in FEATURE_VALUE_LABELS[f])
        eng = p.brain._engine
        eng.synaptic_scaling = value_areas
        p.brain._synaptic_scaling = value_areas
        eng.synaptic_scaling_deferred = True
        p.morph_repetitions = 1
        p.morph_flush_every = k
        gen.FRAMES_PER_STAGE = frames
        gen.SUBJECT_SAMPLING = "zipf"

        # MECHANISM SPY: pre-flush column-mass concentration of pending
        # fibers into the NUMBER value areas. Observation only.
        orig_flush = eng.flush_synaptic_scaling

        def spying_flush():
            pending = getattr(eng, "_pending_scaling", None) or {}
            for (src, tgt), cols in pending.items():
                if tgt not in value_number or not cols:
                    continue
                conn = eng._area_conns.get(src, {}).get(tgt)
                w = getattr(conn, "weights", None)
                if w is None or getattr(w, "ndim", 0) != 2:
                    continue
                cs = [c for c in sorted(cols) if c < w.shape[1]]
                if not cs:
                    continue
                colsums = np.asarray(w[:, cs]).sum(axis=0)
                mean = float(colsums.mean())
                if mean > 0:
                    conc_log.append(float(colsums.max()) / mean)
            return orig_flush()

        eng.flush_synaptic_scaling = spying_flush

    def train_final(p):
        CurriculumTrainer(p).train_stage(FINAL)
        return p

    parser = forked_parser(f"zs-n{N}", seed, build_and_train_pre,
                           arm_setup, train_final)

    sets_ = attested_morph_sets(parser)

    def score(pl_forms, sg_forms):
        mi = {"PL": 0, "SG": 0}
        ov = {"PL": 0, "SG": 0}
        for label, forms in (("PL", pl_forms), ("SG", sg_forms)):
            for form in forms:
                got, diag = parser.recall_number(form)
                mi[label] += got == label
                ov[label] += diag.get("overlap_answer") == label
        return {
            "mi_balanced": (mi["PL"] / len(pl_forms)
                            + mi["SG"] / len(sg_forms)) / 2,
            "ov_balanced": (ov["PL"] / len(pl_forms)
                            + ov["SG"] / len(sg_forms)) / 2,
            "pl_ov": ov["PL"] / len(pl_forms),
            "sg_ov": ov["SG"] / len(sg_forms),
        }

    from neural_assemblies.assembly_calculus.emergent.evaluation \
        .morph_features import score_recall
    attested = score(sets_["PL"], sets_["SG"])
    tense = score_recall(parser.recall_tense,
                         {t: sets_[t] for t in ("PRESENT", "PAST")})

    g_ok = g_total = 0
    for text, expected in ROLE_PROBES:
        roles, _d = parser.parse_roles_by_reconstruction(text.split())
        for w, want in expected.items():
            g_total += 1
            g_ok += roles.get(w) == want

    conc = float(np.mean(conc_log)) if conc_log else None
    print(f"[E19 K={k} zipf-{frames} seed={seed}] "
          f"ov bal={attested['ov_balanced']:.3f} "
          f"(PL={attested['pl_ov']:.3f} SG={attested['sg_ov']:.3f}) | "
          f"mi bal={attested['mi_balanced']:.3f} | "
          f"tense={tense['_balanced']:.3f} | conc={conc} | "
          f"flushes={len(conc_log)} | roles {g_ok}/{g_total}", flush=True)
    return {"attested": attested, "tense_balanced": tense["_balanced"],
            "mean_concentration": conc, "n_flush_obs": len(conc_log),
            "guards": {"roles_ok": g_ok, "roles_total": g_total}}


def cell_key(k, frames):
    return f"K{k}-{frames}"


def main():
    if os.environ.get("FR_ANALYZE") == "1":
        with open(OUT_PATH) as f:
            raw = json.load(f)
        results = {c: {int(s): v for s, v in by.items()}
                   for c, by in raw.items()}
    else:
        from _parallel import run_cells

        cells = [(k, b, s) for k in KS for b in BUDGETS for s in SEEDS]
        cell_results = run_cells(run_cell, cells)
        results = {}
        for (k, frames, seed), res in cell_results.items():
            results.setdefault(cell_key(k, frames), {})[seed] = res
        with open(OUT_PATH, "w") as f:
            json.dump({c: {str(s): v for s, v in by.items()}
                       for c, by in results.items()}, f, indent=2)

    if len(SEEDS) < 3:
        print("\n(smoke mode: too few seeds for ensembles -- see JSON)")
        return

    from neural_assemblies.diagnostics import ensemble

    def bal(k, frames, s):
        return results[cell_key(k, frames)][s]["attested"]["ov_balanced"]

    print("\n=== K x budget (attested balanced, overlap readout) ===")
    for frames in BUDGETS:
        for k in KS:
            print(ensemble(lambda s: bal(k, frames, s), SEEDS,
                           label=f"K={k:<4d} zipf-{frames}"))

    print("\n=== registered bars ===")
    mids = [k for k in KS if k not in (0, 1)]
    for k in mids:
        print(ensemble(
            lambda s: bal(k, 400, s) - max(bal(1, 400, s), bal(0, 400, s)),
            SEEDS, label=f"F1 K={k} minus best endpoint @400 (>0)"))
    best = None
    for frames in BUDGETS:
        for k in KS:
            m = float(np.mean([bal(k, frames, s) for s in SEEDS]))
            if best is None or m > best[0]:
                best = (m, k, frames)
    _, bk, bf = best
    print(ensemble(lambda s: bal(bk, bf, s), SEEDS,
                   label=f"F2 best arm K={bk} zipf-{bf} (bar >= 0.75)"))

    arm_conc, arm_acc = [], []
    for frames in BUDGETS:
        for k in KS:
            cs = [results[cell_key(k, frames)][s]["mean_concentration"]
                  for s in SEEDS]
            cs = [c for c in cs if c is not None]
            if not cs:
                continue
            arm_conc.append(float(np.mean(cs)))
            arm_acc.append(float(np.mean(
                [bal(k, frames, s) for s in SEEDS])))
            print(f"F3 K={k:<4d} zipf-{frames}: mean concentration "
                  f"{arm_conc[-1]:.3f}  acc {arm_acc[-1]:.3f}")
    print(f"F3 rank corr(concentration, accuracy) across arms = "
          f"{spearman(arm_conc, arm_acc):.4f}  (bar < 0)")

    print("\n=== guards ===")
    for frames in BUDGETS:
        print(ensemble(lambda s: bal(0, frames, s), SEEDS,
                       label=f"K=0 zipf-{frames} (E17 ref "
                             f"{E17_REF[frames]:.3f})"))
    for frames in BUDGETS:
        for k in KS:
            key = cell_key(k, frames)
            t = float(np.mean([results[key][s]["tense_balanced"]
                               for s in SEEDS]))
            roles = sum(results[key][s]["guards"]["roles_ok"]
                        for s in SEEDS)
            tot = sum(results[key][s]["guards"]["roles_total"]
                      for s in SEEDS)
            print(f"K={k:<4d} zipf-{frames}: tense {t:.3f} | "
                  f"roles {roles}/{tot}")


if __name__ == "__main__":
    sys.exit(main())

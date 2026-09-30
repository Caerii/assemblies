"""E16: competition dynamics x the last exposure gap -- end the number arc.

PRE-REGISTERED (task #145), after the modes landed (9884615) and before
any scaled cell was scored.

WHERE E15 LEFT IT: the split architecture made data monotone and the
within-area OVERLAP readout reads zipf-200 at 0.700 +/- 0.037, while the
registered one-shot MI comparison reads 0.637 with margins pinned at
7-10% (the #24 weak-primitive finding, third measurement). The residue
to the 0.75 bar decomposed into (i) the DECISION RULE and (ii) EXPOSURE
(most items at 1-2 episodes; reliable regime ~3-4).

DESIGN. One trained parser per (budget, seed) -- the readout variants
are scored on the SAME substrate, so the comparison is paired by
construction. Cells: ZIPF x FRAMES {200, 400} x seeds 42-51 (20 cells),
split architecture, SLOW scaling on the value areas. Variants scored per
cell: MI-oneshot, MI-latched T in {3,10}, MI-settled T in {3,10}, and
the overlap answer (carried in every diag).

PRE-RUN ARITHMETIC (E10 lesson). zipf-400: ~400 x 0.85 ~= 340 subject
draws, x PLURAL_RATE 0.3 ~= 100 PL episodes over the zipf-attested set
(~30 at 200; the tail attests sub-linearly so expect ~30-40 at 400)
~= 2.5-3.3 exposures/form -- most of the exam enters the reliable
regime. Each cell prints total attested exposure so the arithmetic is
checked before any bar is read.

REGISTERED BARS:
  L  THE LATCH PREDICTION (an understanding check, falsifiable): paired
     acc(latched,T) - acc(oneshot) == 0 exactly per seed (MI silences
     the loser at step 1; its recurrence is gone, so the decision CANNOT
     change), while margin(latched) > margin(oneshot). If latching DOES
     change accuracy, our model of the dynamics is wrong -- report
     loudly, do not celebrate whichever direction it moved.
  S  SETTLED CLOSES THE GAP: paired acc(settled,10) - acc(oneshot) > 0
     with CI excluding zero at zipf-200, and settled-10 reaches the
     overlap readout (delta vs overlap, CI containing 0 or above) --
     the neural mechanism matching the Python argmax it implements.
  X  THE BAR: best variant at zipf-400 attested balanced >= 0.75.
     ASPIRATIONAL: overlap AND settled both clear it (mechanism-robust).
  GUARDS: roles 4/4; attested PL n and total exposure per cell (the
     growing-exam decomposition); oneshot zipf-200 reproduces E15
     (0.637 envelope) -- same substrate, same readout, must match.

DECISION RULE: S+X pass -> the number arc CLOSES: corpus statistics
(Zipf x budget), area architecture (per-value + MI), and competition
dynamics (evidence accumulation before commitment) each fixed at its
level; adoption discussion (defaults + CHILDES) opens citing the chain
E12->E16. X fails with exposure arithmetic confirmed -> the
exposure->accuracy curve at 3 episodes is flatter than the 50-frame
calibration promised; fit the curve from this run's per-item data and
register the budget that clears it. S fails -> evidence accumulation
does not close the gap; the overlap readout stands as production and
the drive-comparison weakness gets its own investigation.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from research.json_documents import write_checkpoint_document

SEEDS = list(range(42, 52))
BUDGETS = (200, 400)
PRE_STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD")
FINAL = "SENTENCES"
N = int(os.environ.get("CD_N", "3000"))
if os.environ.get("CD_SMOKE") == "1":
    SEEDS = [42]

OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "competition_dynamics_results.json")

VARIANTS = (("oneshot", 1), ("latched", 3), ("latched", 10),
            ("settled", 3), ("settled", 10))

from overlap_ceiling import spearman  # noqa: E402,F401
from zipf_synthesis import ROLE_PROBES  # noqa: E402


def run_cell(frames: int, seed: int) -> dict:
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
        p.brain._engine.synaptic_scaling = value_areas
        p.brain._synaptic_scaling = value_areas
        p.brain._engine.synaptic_scaling_deferred = True
        p.morph_repetitions = 1
        gen.FRAMES_PER_STAGE = frames
        gen.SUBJECT_SAMPLING = "zipf"

    def train_final(p):
        CurriculumTrainer(p).train_stage(FINAL)
        return p

    parser = forked_parser(f"zs-n{N}", seed, build_and_train_pre,
                           arm_setup, train_final)

    sets_ = attested_morph_sets(parser)
    exposure = getattr(parser, "_morph_exposure", {})
    items_by_label = {"PL": sets_["PL"], "SG": sets_["SG"]}
    total_exp = sum(exposure.get(f"NUMBER:{f}", 0) for f in sets_["PL"])

    out: dict = {"variants": {}, "attested_pl_n": len(sets_["PL"]),
                 "total_attested_exposure": total_exp}
    overlap_stats = None
    for mode, t in VARIANTS:
        parser.mi_readout_mode = mode
        parser.mi_latch_rounds = t
        per_label: dict = {}
        margins = []
        items = []
        ov_per_label: dict = {}
        for label, forms in items_by_label.items():
            ok = ov_ok = 0
            for form in forms:
                got, diag = parser.recall_number(form)
                ok += got == label
                ov_ok += diag.get("overlap_answer") == label
                if diag.get("margin") is not None:
                    margins.append(diag["margin"])
                if label == "PL":
                    items.append({
                        "form": form,
                        "exposure": exposure.get(f"NUMBER:{form}", 0),
                        "correct": bool(got == label),
                    })
            per_label[label] = ok / len(forms) if forms else None
            ov_per_label[label] = ov_ok / len(forms) if forms else None
        out["variants"][f"{mode}-{t}"] = {
            "balanced": float(np.mean([v for v in per_label.values()
                                       if v is not None])),
            "per_label": per_label,
            "mean_margin": float(np.mean(margins)) if margins else None,
            "items": items,
        }
        if overlap_stats is None:
            # The overlap answer is mode-independent (same probe protocol
            # inside every recall); scored once, from the first variant.
            overlap_stats = {
                "balanced": float(np.mean([v for v in ov_per_label.values()
                                           if v is not None])),
                "per_label": ov_per_label,
            }
    assert overlap_stats is not None
    out["overlap"] = overlap_stats

    g_ok = g_total = 0
    for text, expected in ROLE_PROBES:
        roles, _d = parser.parse_roles_by_reconstruction(text.split())
        for w, want in expected.items():
            g_total += 1
            g_ok += roles.get(w) == want
    out["guards"] = {"roles_ok": g_ok, "roles_total": g_total}

    line = " | ".join(
        f"{k}={v['balanced']:.3f}" for k, v in out["variants"].items())
    print(f"[E16 zipf-{frames} seed={seed}] {line} | "
          f"overlap={overlap_stats['balanced']:.3f} | "
          f"PL n={len(sets_['PL'])} exp={total_exp} | "
          f"roles {g_ok}/{g_total}", flush=True)
    return out


def main():
    if os.environ.get("CD_ANALYZE") == "1":
        with open(OUT_PATH) as f:
            raw = json.load(f)
        results = {int(b): {int(s): v for s, v in by.items()}
                   for b, by in raw.items()}
    else:
        from _parallel import run_cells

        cells = [(b, s) for b in BUDGETS for s in SEEDS]
        cell_results = run_cells(run_cell, cells)
        results = {}
        for (frames, seed), res in cell_results.items():
            results.setdefault(frames, {})[seed] = res
        write_checkpoint_document(Path(OUT_PATH), {str(b): {str(s): v for s, v in by.items()}
                                      for b, by in results.items()})

    if len(SEEDS) < 3:
        print("\n(smoke mode: too few seeds for ensembles -- see JSON)")
        return

    from neural_assemblies.diagnostics import ensemble

    def vbal(frames, variant, s):
        return results[frames][s]["variants"][variant]["balanced"]

    def vmargin(frames, variant, s):
        return results[frames][s]["variants"][variant]["mean_margin"]

    def obal(frames, s):
        return results[frames][s]["overlap"]["balanced"]

    print("\n=== variants x budget (attested number balanced) ===")
    for frames in BUDGETS:
        for mode, t in VARIANTS:
            print(ensemble(lambda s: vbal(frames, f"{mode}-{t}", s),
                           SEEDS, label=f"zipf-{frames} {mode}-{t}"))
        print(ensemble(lambda s: obal(frames, s), SEEDS,
                       label=f"zipf-{frames} overlap"))

    print("\n=== registered bars ===")
    for t in (3, 10):
        print(ensemble(
            lambda s: vbal(200, f"latched-{t}", s)
            - vbal(200, "oneshot-1", s),
            SEEDS, label=f"L  acc(latched-{t}) - acc(oneshot) @200 "
                         f"(bar: == 0)"))
    print(ensemble(
        lambda s: vmargin(200, "latched-10", s)
        - vmargin(200, "oneshot-1", s),
        SEEDS, label="L  margin(latched-10) - margin(oneshot) @200 (>0)"))
    print(ensemble(
        lambda s: vbal(200, "settled-10", s) - vbal(200, "oneshot-1", s),
        SEEDS, label="S  acc(settled-10) - acc(oneshot) @200 (>0)"))
    print(ensemble(
        lambda s: vbal(200, "settled-10", s) - obal(200, s),
        SEEDS, label="S  acc(settled-10) - acc(overlap) @200 (~0 or >)"))
    best = {}
    for frames in BUDGETS:
        cand = {f"{m}-{t}": np.mean([vbal(frames, f"{m}-{t}", s)
                                     for s in SEEDS])
                for m, t in VARIANTS}
        cand["overlap"] = np.mean([obal(frames, s) for s in SEEDS])
        best[frames] = max(cand.items(), key=lambda kv: kv[1])[0]
    for frames in BUDGETS:
        name = best[frames]

        def get(s, fr=frames, nm=name):
            return (obal(fr, s) if nm == "overlap"
                    else vbal(fr, nm, s))
        print(ensemble(get, SEEDS,
                       label=f"X  best variant zipf-{frames} = {name} "
                             f"(bar @400 >= 0.75)"))

    print("\n=== exposure arithmetic check + guards ===")
    for frames in BUDGETS:
        print(ensemble(
            lambda s: results[frames][s]["total_attested_exposure"],
            SEEDS, label=f"total attested PL exposure zipf-{frames}"))
        print(ensemble(
            lambda s: results[frames][s]["attested_pl_n"], SEEDS,
            label=f"attested PL n           zipf-{frames}"))
        roles = [results[frames][s]["guards"] for s in SEEDS]
        ok = sum(r["roles_ok"] for r in roles)
        tot = sum(r["roles_total"] for r in roles)
        print(f"roles zipf-{frames}: {ok}/{tot}")
    print("E15 reference: oneshot zipf-200 = 0.637 +/- 0.048, "
          "overlap zipf-200 = 0.700 +/- 0.037")

    print("\n=== zipf-400 per-item, best MI variant (pooled) ===")
    from collections import defaultdict
    mi_best = best[400] if best[400] != "overlap" else "settled-10"
    agg = defaultdict(lambda: [0, 0, 0])
    for s in SEEDS:
        for it in results[400][s]["variants"][mi_best]["items"]:
            a = agg[it["form"]]
            a[0] += it["exposure"]
            a[1] += it["correct"]
            a[2] += 1
    for form, (e, ok, n) in sorted(agg.items(),
                                   key=lambda kv: -kv[1][0])[:20]:
        print(f"  {form:10s} exp/seed={e / n:5.1f}  acc={ok / n:.2f}")


if __name__ == "__main__":
    sys.exit(main())

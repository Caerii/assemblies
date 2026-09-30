"""#52 recipe propagation: the {n} x {role_bind_gain} 2x2 on the
reconstruction-readout exam -- the k*p law's predictions tested where it
can FAIL.

Context: #151 closed with the paper regime at ceiling on morphology
(kp=1.5, gain 4 = the margin lever). Role binding's standing exam
(sentence_conditioned_readout: 12/12 both voices, occupant gap ~0.95 at
phon_weight=6) runs at the SAME operating point kp = 30*0.05 = 1.5, but
with bind's T=2 plasticity rounds: one episode boosts (1.05)^2 ~ 1.10
(gain 4: (1.2)^2 = 1.44) against an extreme-value demand of
sqrt(2 ln(n/k)/kp) = 2.48 at n=3e3 and 3.29 at n=1e5. Compounding over
E episodes, baseline crosses the margin near E~9 (n=3e3) / E~12 (n=1e5)
and gain 4 near E~3 / E~4.

INFRASTRUCTURE THIS REGISTERS AGAINST (same commit): `role_bind_gain`,
a parser property writing the ENGINE's per-fiber beta store so all five
role-binding writers see it; and `set_base_beta`, the single beta-policy
owner (the stage schedule used to erase per-fiber overlays at every
stage boundary -- caught by test_role_bind_gain's liveness test, which
trains through the curriculum path and fails on the pre-fix code).
No-gain path verified BYTE-IDENTICAL to the pre-change engine.

REGISTERED PREDICTIONS (before any cell; each can fail):
  P-BASE   gain=1 @ n=3000 reproduces the standing exam: parse
           accuracy at/near ceiling on the 8 frames, occupant gap
           positive with seed variance. A failure here is a harness
           bug, not a finding.
  P-GAIN@3e3  The exam's content words are HIGH-exposure (core corpus,
           dozens of episodes), far above the E~9 crossover: parse
           accuracy predicted INSENSITIVE to gain at n=3000. Retrieval
           over ALL bound words may move only in its low-exposure tail.
           A parse-accuracy JUMP here would REFUTE the margin account
           (it would mean the gain acts through something other than
           the under-margin tail).
  P-SCALE  At n=1e5 the collision load falls ~33x, so the occupant gap
           RISES for G1; the margin demand rises 2.48 -> 3.29, so the
           low-exposure retrieval tail falls for G1 and gain 4 restores
           it: the (G4 - G1) retrieval delta is predicted LARGER at
           n=1e5 than at n=3e3 (the interaction). If gain instead HURTS
           at n=1e5, the crowding side of the kp law took over and the
           recipe's gain axis does NOT propagate to roles.
  DECISION RULE (pre-stated): role_bind_gain=4 becomes the documented
           at-scale recipe for roles ONLY if the P-SCALE interaction
           shows with nothing regressing (parse, gap, retrieval all
           >= G1 within CI at n=1e5). Otherwise gain stays 1.0 and the
           law's scoping ("the margin lever only pays below the
           margin") is the finding.
  SEQUENTIAL PLAN (pre-stated): n=3000 arms run seeds 42-51; n=1e5
           arms run seeds 42-46 first (runtime), extended to 51 before
           ANY conclusion if the interaction is within its CI of zero.

Exposure stratification uses the vocabulary's static frequency field as
the exposure proxy (the curriculum samples by it); the exam's per-word
correctness is recorded with it for the strata plots.

Run: python role_recipe_2x2.py
"""
from __future__ import annotations

import json
import os
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np

from research.json_documents import write_new_document

from sentence_conditioned_readout import (  # noqa: E402
    EVENTS,
    _frame_roles,
    gated_parse_and_reconstruct,
)

SEEDS_SMALL = list(range(42, 52))
SEEDS_LARGE = list(range(42, 47))
K = 30
OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "role_recipe_2x2_results.json")


def build(seed, n, gain):
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        CurriculumTrainer,
    )
    from neural_assemblies.assembly_calculus.emergent.vocabulary_builder \
        import build_vocabulary_preset

    random.seed(seed)
    np.random.seed(seed)
    p = EmergentParser(n=n, k=K, seed=seed,
                      vocabulary=build_vocabulary_preset("core"),
                      fast_training=True)
    p.role_bind_gain = gain
    t = CurriculumTrainer(p)
    for stage in ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD",
                  "SENTENCES"):
        t.train_stage(stage)
    return p


def word_freq(p, word):
    """Realized training count from the parser's own distributional stats
    -- the honest exposure proxy (counted, not the vocabulary's static
    frequency field, which the parser does not expose)."""
    wc = getattr(getattr(p, "dist_stats", None), "word_count", None) or {}
    return float(wc.get(word, 0))


def retrieval(p, max_per_area=40):
    """Top-1 readback over the stored role lexicons, per word.

    Symmetry with training: the SAME `ops.bind` under read_only()
    replays the word's core assembly into the role area; identity is
    argmax overlap against the stored bound assemblies. Chance falls
    with lexicon size; per-word rows carry the frequency proxy so the
    tail is separable.
    """
    from neural_assemblies.assembly_calculus.assembly import (
        overlap as assembly_overlap,
    )
    from neural_assemblies.assembly_calculus.ops import bind
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        ROLE_AGENT, ROLE_PATIENT,
    )

    rows = []
    brain = p.brain
    for role_area in (ROLE_AGENT, ROLE_PATIENT):
        lex = p.role_lexicons.get(role_area, {})
        words = sorted(lex)[:max_per_area]
        if len(words) < 2:
            continue
        for w in words:
            core_area = p._word_core_area(w)
            stored_core = p.core_lexicons.get(core_area, {}).get(w)
            if stored_core is None:
                continue
            with brain.read_only():
                probe = bind(brain, core_area, role_area, stored_core)
            scores = {cand: assembly_overlap(lex[cand], probe)
                      for cand in words}
            got = max(scores, key=scores.get)
            rows.append({"role": role_area, "word": w, "ok": got == w,
                         "freq": word_freq(p, w),
                         "n_candidates": len(words)})
    return rows


def cell(seed, n, gain):
    p = build(seed, n, gain)
    frames, gaps = [], []
    for active, passive, _same in EVENTS:
        for kind, text in (("active", active), ("passive", passive)):
            words = text.split()
            agent, patient = _frame_roles(text)
            recon, diag = gated_parse_and_reconstruct(p, words)
            ok = (recon.get(agent) == "AGENT"
                  and recon.get(patient) == "PATIENT")
            frames.append({"kind": kind, "ok": ok})
            # diag["gaps"] entries are (role, occupant, top, runner, gap).
            for entry in diag.get("gaps") or []:
                gaps.append(float(entry[-1]))
    ret = retrieval(p)
    out = {
        "parse_acc": st.mean(f["ok"] for f in frames),
        "parse_active": st.mean(
            f["ok"] for f in frames if f["kind"] == "active"),
        "parse_passive": st.mean(
            f["ok"] for f in frames if f["kind"] == "passive"),
        "gap_mean": st.mean(gaps) if gaps else None,
        "ret_acc": st.mean(r["ok"] for r in ret) if ret else None,
        "ret_n": len(ret),
        "ret_rows": ret,
    }
    return out


def mci(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return {"mean": xs[0] if xs else None, "ci": None, "n": len(xs)}
    t = {5: 2.776, 10: 2.262}.get(len(xs), 2.262)
    return {"mean": st.mean(xs), "ci": t * st.stdev(xs) / len(xs) ** 0.5,
            "n": len(xs)}


def main():
    cells = {}
    for n, seeds in ((3000, SEEDS_SMALL), (100000, SEEDS_LARGE)):
        for gain in (1.0, 4.0):
            for seed in seeds:
                c = cell(seed, n, gain)
                cells[f"{n}-G{int(gain)}-{seed}"] = c
                print(f"n={n} G{int(gain)} seed={seed} "
                      f"parse={c['parse_acc']:.3f} "
                      f"(A {c['parse_active']:.2f}/P "
                      f"{c['parse_passive']:.2f}) "
                      f"ret={c['ret_acc'] if c['ret_acc'] is None else round(c['ret_acc'], 3)} "
                      f"(n={c['ret_n']})", flush=True)

    def arm(n, gain, key, seeds):
        return mci([cells[f"{n}-G{int(gain)}-{s}"][key] for s in seeds])

    analysis = {}
    for n, seeds in ((3000, SEEDS_SMALL), (100000, SEEDS_LARGE)):
        for gain in (1.0, 4.0):
            tag = f"n{n}_G{int(gain)}"
            analysis[f"{tag}_parse"] = arm(n, gain, "parse_acc", seeds)
            analysis[f"{tag}_ret"] = arm(n, gain, "ret_acc", seeds)
            analysis[f"{tag}_gap"] = arm(n, gain, "gap_mean", seeds)
    # The registered interaction: paired (G4-G1) retrieval delta per n.
    for n, seeds in ((3000, SEEDS_SMALL), (100000, SEEDS_LARGE)):
        ds = []
        for s in seeds:
            a = cells[f"{n}-G4-{s}"]["ret_acc"]
            b = cells[f"{n}-G1-{s}"]["ret_acc"]
            if a is not None and b is not None:
                ds.append(a - b)
        analysis[f"interaction_ret_delta_n{n}"] = mci(ds)

    out = {"cells": {k: {kk: vv for kk, vv in v.items()
                         if kk != "ret_rows"} for k, v in cells.items()},
           "ret_rows": {k: v["ret_rows"] for k, v in cells.items()},
           "analysis": analysis}
    write_new_document(Path(OUT_PATH), out)
    print(json.dumps(analysis, indent=2))
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()

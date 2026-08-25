"""WHY does substrate C merge assemblies, when a per-column scalar preserves
within-column ratios exactly?

It does preserve them -- and that is beside the point, because k-WTA never
compares within a column. It compares ACROSS columns, and a per-column scalar
is precisely what is not preserved across columns. So the question is not
whether the scalar distorts (it must) but WHAT SELECTS IT.

    THE TWO DIVISORS, side by side -- the whole difference between B and C:

      B  norm_init        drive_j / d_j       d_j = COUNT of present synapses
                                              -> potentiation-INVARIANT
                                              -> applied to EVERY column
      C  synaptic_scaling w[:,j] *= S/mass_j  mass_j = SUM of current weights
                                              -> potentiation-DEPENDENT
                                              -> applied to WINNER columns ONLY

    S = rows * p (`_sparse.py`, `_scale_columns_now`) is a POPULATION CONSTANT:
    the AMBIENT expected in-degree, identical for every column.

On a uniform random graph those two agree, because every column's degree IS the
ambient one. This engine is not a uniform random graph. Lazy recruitment
materializes a neuron's afferents CONDITIONED ON IT HAVING WON, so realized
degree is not ambient, and the disagreement is exactly the residue a population
setpoint cannot cancel -- an assembly-INDEPENDENT bias, which is what a merger
needs.

THREE ARMS, so the claim is causal rather than a story fitted to two columns:

    B  norm_init               divide by the column's own COUNT, every column
    C  synaptic_scaling        pin mass to the POPULATION setpoint, winners only
    D  synaptic_scaling +      pin mass to the COLUMN'S OWN degree, winners only
       setpoint="degree"
    E  synaptic_scaling +      population setpoint, but rescale EVERY
       scope="all"            materialized column, not only the winners

D asks whether the merger is the setpoint: is the residue that survives a
POPULATION setpoint each column's own degree excess? E asks whether it is the
SELECTIVITY: k-WTA selects among CANDIDATES, and C only ever touches columns
that have already won, so it cannot influence the selection that produced them.
E normalizes the candidates before the comparison instead of the winners after
it -- and changes nothing else, so C's potentiation-dependent divisor, and
therefore C's separate gain deficit, survives into E untouched.

The two are mutually exclusive predictions about the same table, registered
before the run: D fixes it, or E does, or neither and this note is wrong.

COMPLETION IS MEASURED, NOT ASSUMED. `_normalize_area_columns`'s docstring
already records that restoring a fiber to its OWN original total cancels the
net potentiation an attractor needs, collapsing pattern completion to 0.000.
D is a per-COLUMN rather than per-FIBER version of that idea, so it is exactly
the arm at risk of buying distinctness by destroying the assembly. Half-cue
rank-1 is therefore a headline number here, not an afterthought: a substrate
that separates assemblies by making none of them attractors has fixed nothing
([[completion-works-in-regime-under-norm-init]]).

Nothing is read off a bare seed mean; every table carries per-seed values.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

from neural_assemblies.assembly_calculus.assembly import Assembly
from neural_assemblies.assembly_calculus.ops import (
    _compact_index, _snap, activate_assembly,
)
from neural_assemblies.core.index_spaces import NeuronIds
from neural_assemblies.core.brain import Brain
from neural_assemblies.diagnostics import assembly_overlap

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# IN REGIME. p=0.5 gives kp=25 against the 3*ln(2000)=22.8 floor. Everything
# here is void at the ratchet study's p=0.05 (kp=2.5) -- the exact error
# [[completion-works-in-regime-under-norm-init]] records having been made twice.
N, K, BETA, P, T, M = 2000, 50, 0.10, 0.5, 8, 8
AREA = "A"
ARMS = ("B", "C", "D", "E", "G")
SEEDS = [42, 43, 44]


def _train(arm, seed):
    random.seed(seed)
    np.random.seed(seed)
    brain = Brain(p=P, seed=seed, engine="numpy_sparse",
                  recurrent_projection=True,
                  norm_init=(arm in ("B", "G")),
                  synaptic_scaling=(arm in ("C", "D", "E", "G")))
    brain.add_area(AREA, N, K, BETA)
    if arm == "D":
        brain._engine_for(brain.areas[AREA]).synaptic_scaling_setpoint = "degree"
    if arm == "E":
        brain._engine_for(brain.areas[AREA]).synaptic_scaling_scope = "all"
    stims = []
    for i in range(M):
        s = f"s{i}"
        brain.add_stimulus(s, K)
        stims.append(s)
    stored = []
    for s in stims:
        brain.inhibit_areas([AREA])
        for _ in range(T):
            brain.project({s: [AREA]}, {AREA: [AREA]})
        stored.append(_snap(brain, AREA))
    return brain, stims, stored


def _retrieve(brain, stim=None, half_of=None):
    """One retrieval inside a probe; stimulus-cued or half-assembly-cued.

    `brain.probe()` because RECRUITMENT, not plasticity, is the channel by
    which a readout changes what it reads, and the snap is taken INSIDE the
    block ([[probe-isolation-required]]).
    """
    with brain.probe():
        brain.inhibit_areas([AREA])
        if half_of is not None:
            ids = np.asarray(half_of.winners)[: K // 2]
            activate_assembly(brain, Assembly(AREA, NeuronIds(ids)))
            for _ in range(T):
                brain.project({}, {AREA: [AREA]})      # recurrence alone
        else:
            for _ in range(T):
                brain.project({stim: [AREA]}, {AREA: [AREA]})
        return _snap(brain, AREA)


def _column_stats(brain):
    """Per-column in-degree, mass, and the accumulated per-entry scalar.

    The scalar is the MEDIAN nonzero entry, not the mean: the ~k*p potentiated
    entries sit far above the rest, and the quantity of interest is the
    baseline every OTHER synapse of that neuron was multiplied by -- the part
    that carries over to unrelated cues. Under B weights stay on the unit scale
    by construction (the 1/d_j lives at read time), so B's median is 1.0 for
    everyone; B's effective scalar is reconstructed from the degree instead.
    """
    eng = brain._engine_for(brain.areas[AREA])
    conn = eng._area_conns[AREA][AREA]
    w = np.asarray(conn.weights)
    rows = int(eng._areas[AREA].w)
    cols = min(rows, w.shape[1])
    sub = np.asarray(w[:rows, :cols], dtype=np.float64)

    nz = sub != 0.0
    nnz = nz.sum(axis=0).astype(np.float64)
    mass = sub.sum(axis=0)
    med = np.ones(cols, dtype=np.float64)
    for j in range(cols):
        v = sub[nz[:, j], j]
        if v.size:
            med[j] = float(np.median(v))
    return nnz, mass, med, rows, cols


def _multiplicity(brain, stored, cols):
    """How many of the M stored assemblies contain each compact column.

    Compact index is also RECRUITMENT ORDER, which is why it is used below: the
    elite turns out to live at the very front of it.
    """
    inv = _compact_index(brain._engine_for(brain.areas[AREA]), AREA) or {}
    mult = np.zeros(cols, dtype=np.int32)
    for st in stored:
        c = [inv[int(x)] for x in np.asarray(st.winners)
             if int(x) in inv and inv[int(x)] < cols]
        mult[c] += 1
    return mult


def _spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def worker(arm, seed):
    brain, stims, stored = _train(arm, seed)
    nnz, mass, med, rows, cols = _column_stats(brain)
    mult = _multiplicity(brain, stored, cols)
    seen = mult > 0

    full_hits = half_hits = 0
    for i, s in enumerate(stims):
        live = _retrieve(brain, stim=s)
        ov = [assembly_overlap(np.asarray(live.winners), np.asarray(st.winners))
              for st in stored]
        full_hits += int(np.argmax(ov) == i)
        liveh = _retrieve(brain, half_of=stored[i])
        ovh = [assembly_overlap(np.asarray(liveh.winners),
                                np.asarray(st.winners)) for st in stored]
        half_hits += int(np.argmax(ovh) == i)

    pair = [assembly_overlap(np.asarray(stored[i].winners),
                             np.asarray(stored[j].winners))
            for i in range(M) for j in range(i + 1, M)]

    def _m(x, mask):
        return float(np.mean(x[mask])) if mask.any() else float("nan")

    hist = np.bincount(mult, minlength=M + 1)[: M + 1]
    return {
        "arm": arm, "seed": seed, "rows": rows, "cols": cols,
        "rank1_full": full_hits / M, "rank1_half": half_hits / M,
        "pairwise": float(np.mean(pair)),
        "union": int(seen.sum()),
        "max_mult": int(mult.max()),
        "frac_shared": float((mult >= 2).sum() / max(1, seen.sum())),
        "hist": [int(x) for x in hist],
        "deg_pop": float(np.mean(nnz)),
        "deg_private": _m(nnz, mult == 1),
        "deg_shared": _m(nnz, mult >= 2),
        "deg_elite": _m(nnz, mult >= 4),
        "deg_decile0": float(np.mean(nnz[: max(1, cols // 10)])),
        "rho_deg_mult": _spearman(nnz[seen], mult[seen].astype(float)),
        "rho_rank_deg": _spearman(np.arange(cols, dtype=float), nnz),
        "med_private": _m(med, mult == 1),
        "med_shared": _m(med, mult >= 2),
        "mass_cv": float(np.std(mass[seen]) / max(np.mean(mass[seen]), 1e-12)),
    }


def worker_cfg(arm, seed, beta, load):
    """`worker` with beta and M passed EXPLICITLY.

    Windows spawns rather than forks, so module globals set in the parent do
    not reach the pool. A sweep over beta or M must therefore carry them in the
    cell tuple, not mutate `BETA`/`M` and hope -- that silent-no-op shape is
    how a sweep comes back looking flat.
    """
    global BETA, M
    BETA, M = beta, load
    r = worker(arm, seed)
    r.update(beta=beta, load=load)
    return r


def _per_seed(res, key):
    return {a: [res[(a, s)][key] for s in SEEDS] for a in ARMS}


def _fmt(v, w=9, prec=4):
    return " ".join(f"{x:{w}.{prec}f}" for x in v)


def main():
    from _parallel import run_cells

    chance = K / N
    print("=== substrate C: WHAT merges the assemblies? ===")
    print(f"    n={N} k={K} beta={BETA} p={P} T={T} M={M}, recurrence ON")
    print(f"    kp = {K*P:.1f} vs floor 3*ln(n) = {3*np.log(N):.1f} -> IN REGIME")
    print(f"    chance pairwise = k/n = {chance:.4f};  "
          f"chance union over M={M} ~ {N*(1-(1-chance)**M):.0f}")
    print("    B = norm_init | C = scaling, population setpoint | "
          "D = scaling, per-column degree setpoint\n")

    cells = [(a, s) for a in ARMS for s in SEEDS]
    res = run_cells(worker, cells, max_workers=min(len(cells), 12))

    print("--- IS IT AN ASSEMBLY AT ALL? (per seed 42/43/44) ---")
    for key, lab in (("rank1_full", "rank1 full cue"),
                     ("rank1_half", "rank1 HALF cue (completion)")):
        r = _per_seed(res, key)
        for a in ARMS:
            print(f"    {lab:28s} {a}  {_fmt(r[a], 7, 3)}")
        print()

    print("--- ARE THEY DISTINCT? ---")
    for key, lab, pr in (("pairwise", f"pairwise (chance {chance:.4f})", 4),
                         ("union", "union of the M assemblies", 0),
                         ("max_mult", "max multiplicity", 0),
                         ("frac_shared", "frac of seen in >=2", 3)):
        r = _per_seed(res, key)
        for a in ARMS:
            print(f"    {lab:28s} {a}  {_fmt(r[a], 8, pr)}")
        print()

    print("--- THE ELITE: multiplicity histogram, seed 42 (index = mult) ---")
    for a in ARMS:
        print(f"    {a}  {res[(a, 42)]['hist']}")

    print("\n--- THE MECHANISM: is the per-column degree residue cancelled? ---")
    print("    (B cancels each column's own COUNT, so degree must be FLAT")
    print("     across multiplicity. A population setpoint cannot cancel a")
    print("     per-column excess, so C's must RISE. D tests whether that is")
    print("     what merges.)")
    for key in ("deg_pop", "deg_private", "deg_shared", "deg_elite",
                "deg_decile0", "rho_deg_mult", "rho_rank_deg",
                "med_private", "med_shared"):
        r = _per_seed(res, key)
        pr = 3 if key.startswith(("rho", "med")) else 1
        for a in ARMS:
            print(f"    {key:14s} {a}  {_fmt(r[a], 9, pr)}")

    def mn(a, k):
        return float(np.mean([res[(a, s)][k] for s in SEEDS]))

    print("\n=== BARS ===")
    b1 = mn("B", "rank1_half") > 0.9 and mn("B", "pairwise") < 2 * chance
    print(f"  {'PASS' if b1 else 'FAIL'}  F1 control: B completes "
          f"({mn('B','rank1_half'):.3f}) AND is distinct "
          f"({mn('B','pairwise'):.4f} < {2*chance:.4f})")
    b2 = mn("C", "pairwise") > 3 * chance and mn("C", "max_mult") > 4
    print(f"  {'PASS' if b2 else 'FAIL'}  F2 defect reproduces: C pairwise "
          f"{mn('C','pairwise'):.4f} > {3*chance:.4f}, max mult "
          f"{mn('C','max_mult'):.1f} > 4")
    b3 = mn("C", "rho_deg_mult") > 0.15 and abs(mn("B", "rho_deg_mult")) < 0.10
    print(f"  {'PASS' if b3 else 'FAIL'}  F3 residue is degree: "
          f"rho(deg,mult) C {mn('C','rho_deg_mult'):+.3f} vs "
          f"B {mn('B','rho_deg_mult'):+.3f}")
    b4 = mn("D", "pairwise") < 0.5 * mn("C", "pairwise")
    print(f"  {'PASS' if b4 else 'FAIL'}  F4 THE CLAIM: per-column setpoint "
          f"halves the merger -- D {mn('D','pairwise'):.4f} vs "
          f"C {mn('C','pairwise'):.4f}")
    b5 = mn("D", "rank1_half") > mn("C", "rank1_half")
    print(f"  {'PASS' if b5 else 'FAIL'}  F5 and does not buy it by killing "
          f"the attractor: D half-cue {mn('D','rank1_half'):.3f} vs "
          f"C {mn('C','rank1_half'):.3f} (B {mn('B','rank1_half'):.3f})")

    if b4 and not b5:
        print("\n  -> D separates the assemblies by DESTROYING them, exactly as")
        print("     `_normalize_area_columns`'s docstring predicts for an")
        print("     own-mass setpoint. Distinctness bought that way is not a")
        print("     repair; report it as the trade it is.")
    if not b4:
        print("\n  -> The setpoint is NOT the mechanism. The degree residue is")
        print("     real (F3) but not what merges; look elsewhere and do not")
        print("     report the population-setpoint story.")

    path = os.path.join(_HERE, "seq_scaling_merger_forensics_results.json")
    with open(path, "w") as fh:
        json.dump({"n": N, "k": K, "beta": BETA, "p": P, "T": T, "M": M,
                   "seeds": SEEDS, "chance": chance,
                   "bars": {"F1": bool(b1), "F2": bool(b2), "F3": bool(b3),
                            "F4": bool(b4), "F5": bool(b5)},
                   "cells": {f"{a}/{s}": res[(a, s)] for (a, s) in cells}},
                  fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

"""Task 1 + Task 2 analysis on the CORRECTED (norm_init) substrate.

- Task 1: show the vstar bug before/after. For each run recompute both the old
  churn-based recruit_horizon and the new identification-based V* and print a
  before/after table for representative cells.
- Task 2: beta control. V*(n) per beta (norm ON and OFF), the fitted exponent
  d log V*/d log n, and identification_acc at V=400 per (beta, n). Mean +/- sd
  over seeds.

Reads results_norm_all_rec.json and merges results_confirm_beta_rec.json (extra
seeds) if present. Does NOT re-run training -- vstar is a function of the stored
checkpoints, so the fix is applied post-hoc to the salvaged data.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.experiments.capacity import lexicon_capacity as lc
from research.experiments.capacity.analyze import recruit_horizon

HERE = Path(__file__).parent
NS = (1000, 3000, 10000)
BETAS = (0.0, 0.01, 0.05, 0.2)
THRESH = 0.5


def load_runs():
    runs = json.loads((HERE / "results_norm_all_rec.json").read_text())
    conf = HERE / "results_confirm_beta_rec.json"
    if conf.exists():
        extra = json.loads(conf.read_text())
        runs = runs + extra
        print(f"[merged {len(extra)} confirmation runs -> {len(runs)} total]")
    return runs


def ident_at(run, V):
    for c in run["checkpoints"]:
        if c["V"] == V:
            return c.get("identification_acc")
    return None


def corrected_vstar(run):
    return lc.identification_capacity(run["checkpoints"], THRESH)["vstar"]


def old_vstar(run):
    return recruit_horizon(run["per_word"])


def ms(vals, fmt="{:.1f}"):
    v = [x for x in vals if x is not None
         and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return "n/a"
    if len(v) == 1:
        return fmt.format(v[0])
    return f"{fmt.format(np.mean(v))}+/-{fmt.format(np.std(v))}"


def exponent(vstar_by_n):
    ns = sorted(n for n in vstar_by_n if vstar_by_n[n])
    if len(ns) < 2:
        return float("nan")
    xs = np.log(ns)
    ys = np.log([max(1e-9, np.mean(vstar_by_n[n])) for n in ns])
    return float(np.polyfit(xs, ys, 1)[0])


def main():
    runs = load_runs()
    g = defaultdict(list)
    for r in runs:
        c = r["config"]
        g[(c["n"], c["beta"], bool(r["norm_init"]))].append(r)

    # ---------- TASK 1: vstar bug before / after ----------
    print("\n" + "=" * 78)
    print("TASK 1 -- V* bug: OLD (recruit_horizon / churn) vs NEW (identification)")
    print("=" * 78)
    print(f"{'n':>6}{'beta':>6}{'norm':>5}{'seeds':>6}"
          f"{'OLD vstar':>12}{'NEW vstar':>12}{'ident@400':>12}"
          f"{'vocab_lrn':>10}{'exhaust':>9}")
    demo = [(1000, 0.05, True), (3000, 0.05, True), (10000, 0.05, True),
            (1000, 0.0, True), (10000, 0.2, True), (10000, 0.05, False)]
    for key in demo:
        rs = g.get(key, [])
        if not rs:
            print(f"{key}  MISSING")
            continue
        n, beta, norm = key
        old = [old_vstar(r) for r in rs]
        new = [corrected_vstar(r) for r in rs]
        idn = [ident_at(r, 400) for r in rs]
        vl = [r["vocab_learned"] for r in rs]
        exh = [r["exhausted_at"] for r in rs]
        exh_s = "none" if all(e is None for e in exh) else \
            ms([e for e in exh if e is not None], "{:.0f}")
        print(f"{n:>6}{beta:>6}{int(norm):>5}{len(rs):>6}"
              f"{ms(old):>12}{ms(new):>12}{ms(idn,'{:.3f}'):>12}"
              f"{ms(vl,'{:.0f}'):>10}{exh_s:>9}")

    # ---------- TASK 2: beta control, corrected V* ----------
    for norm in (True, False):
        print("\n" + "=" * 78)
        print(f"TASK 2 -- beta control, norm_init={int(norm)}, rec, topk. "
              f"NEW V* = largest V with mean ident >= {THRESH}")
        print("=" * 78)
        print(f"{'beta':>6}" + "".join(f"{'V*@'+str(n):>16}" for n in NS)
              + f"{'exponent':>10}"
              + "".join(f"{'id@400,n='+str(n):>16}" for n in NS))
        for beta in BETAS:
            vstar_by_n = {}
            vcols, icols = [], []
            for n in NS:
                rs = g.get((n, beta, norm), [])
                if not rs:
                    vcols.append("n/a")
                    icols.append("n/a")
                    continue
                vs = [corrected_vstar(r) for r in rs]
                vstar_by_n[n] = vs
                vcols.append(ms(vs, "{:.0f}"))
                icols.append(ms([ident_at(r, 400) for r in rs], "{:.3f}"))
            exp = exponent(vstar_by_n)
            print(f"{beta:>6}" + "".join(f"{c:>16}" for c in vcols)
                  + f"{exp:>10.3f}" + "".join(f"{c:>16}" for c in icols))

    # exponent for OLD churn metric too (norm ON) to show the artifact
    print("\n[contrast] OLD recruit_horizon exponent per beta, norm_init=1 "
          "(the artifact the fix removes):")
    for beta in BETAS:
        vbn = {}
        for n in NS:
            rs = g.get((n, beta, True), [])
            if rs:
                vbn[n] = [old_vstar(r) for r in rs]
        cols = " ".join(f"n={n}:{ms(vbn.get(n,[]),'{:.0f}')}" for n in NS)
        print(f"  beta={beta:<5} {cols}  exponent(churn)={exponent(vbn):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

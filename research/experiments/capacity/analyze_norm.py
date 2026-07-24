"""
Aggregate the norm_init capacity re-measurement into tables.

Reads results_norm_all_rec.json / results_norm_all_ff.json (produced by
norm_init_capacity.py) and prints, per mode:
  - V*(recruit-horizon) OFF vs ON, per n, with the log-log scaling exponent
  - identification / retrieval-self / pairwise-other capacity at V=400
  - within / between / chance overlap decomposition
All numbers are mean +/- sd over seeds; tests via base.ttest_vs_null.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.experiments.base import summarize, ttest_vs_null  # noqa: E402

HERE = Path(__file__).parent


def ms(vals, fmt="{:.3f}"):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return "n/a"
    s = summarize(list(v))
    if len(v) == 1:
        return fmt.format(s["mean"])
    return f"{fmt.format(s['mean'])}+/-{fmt.format(s['std'])}"


def last_cp(run):
    return run["checkpoints"][-1]


def cp_at(run, V):
    for c in run["checkpoints"]:
        if c["V"] == V:
            return c
    return None


def load(mode):
    p = HERE / f"results_norm_all_{mode}.json"
    if not p.exists():
        return None
    runs = json.loads(p.read_text())
    g = defaultdict(list)
    for r in runs:
        c = r["config"]
        g[(c["n"], c["beta"], bool(r["norm_init"]))].append(r)
    return g


def exponent(vstar_by_n):
    ns = sorted(vstar_by_n)
    if len(ns) < 2:
        return float("nan")
    xs = np.log(ns)
    ys = np.log([np.mean(vstar_by_n[n]) for n in ns])
    return float(np.polyfit(xs, ys, 1)[0])


def report_mode(mode):
    g = load(mode)
    if g is None:
        print(f"\n=== mode={mode}: no data ===")
        return
    ns = sorted({k[0] for k in g})
    betas = sorted({k[1] for k in g})
    print(f"\n{'='*78}\nMODE = {mode.upper()}\n{'='*78}")

    # ---- Table A: recruit-horizon V* and exponent, beta=0.05 ----
    print("\n[A] Recruit-horizon V* (original V* definition), beta=0.05, "
          "topk. Exponent = d logV*/d logn.")
    print(f"{'norm':<6}{'n':>7}{'V* (recruit)':>16}{'w/n final':>14}"
          f"{'V_learned':>12}")
    for norm in (False, True):
        vstar_by_n = {}
        for n in ns:
            runs = g.get((n, 0.05, norm))
            if not runs:
                continue
            vst = [r["vstar"] for r in runs]
            vstar_by_n[n] = vst
            wn = [last_cp(r)["w_over_n"] for r in runs]
            vl = [r["vocab_learned"] for r in runs]
            print(f"{int(norm):<6}{n:>7}{ms(vst,'{:.1f}'):>16}"
                  f"{ms(wn):>14}{ms(vl,'{:.0f}'):>12}")
        exp = exponent(vstar_by_n)
        print(f"   -> norm={int(norm)} recruit-horizon exponent = {exp:.3f}")

    # ---- Table B: identification / retrieval / pairwise capacity ----
    print("\n[B] Discriminability at each checkpoint, beta=0.05, topk.")
    print(f"{'norm':<6}{'n':>7}{'V':>5}{'ident':>16}{'retr_self':>16}"
          f"{'pairwise_other':>18}")
    for norm in (False, True):
        for n in ns:
            runs = g.get((n, 0.05, norm))
            if not runs:
                continue
            Vs = [c["V"] for c in runs[0]["checkpoints"]]
            for V in Vs:
                cs = [cp_at(r, V) for r in runs]
                cs = [c for c in cs if c]
                if not cs:
                    continue
                idn = [c["identification_acc"] for c in cs]
                rs = [c["retrieval_overlap_mean"] for c in cs]
                pw = [c["pairwise_overlap_mean"] for c in cs]
                print(f"{int(norm):<6}{n:>7}{V:>5}{ms(idn):>16}"
                      f"{ms(rs):>16}{ms(pw):>18}")

    # ---- identification-capacity: largest V with ident>=0.5 ----
    print("\n[B2] Identification-capacity V_id (largest checkpoint with "
          "mean ident >= 0.5); '>=400' = right-censored (still good at Vmax).")
    print(f"{'norm':<6}{'n':>7}{'V_id':>10}{'ident@Vmax':>14}")
    for norm in (False, True):
        for n in ns:
            runs = g.get((n, 0.05, norm))
            if not runs:
                continue
            Vs = [c["V"] for c in runs[0]["checkpoints"]]
            vid = 0
            for V in Vs:
                cs = [cp_at(r, V) for r in runs]
                cs = [c for c in cs if c]
                m = np.mean([c["identification_acc"] for c in cs])
                if m >= 0.5:
                    vid = V
            tag = f">={Vs[-1]}" if vid == Vs[-1] else str(vid)
            imax = ms([cp_at(r, Vs[-1])["identification_acc"] for r in runs])
            print(f"{int(norm):<6}{n:>7}{tag:>10}{imax:>14}")

    # ---- Table C: beta control (norm ON) ----
    print("\n[C] Beta control: recruit-horizon V* and exponent per beta.")
    print(f"{'norm':<6}{'beta':>6}"
          + "".join(f"{'V*@'+str(n):>14}" for n in ns) + f"{'exponent':>12}"
          + f"{'ident@400 (n=%d)'%ns[-1]:>18}")
    for norm in (True, False):
        for beta in betas:
            vstar_by_n = {}
            cols = []
            for n in ns:
                runs = g.get((n, beta, norm))
                if not runs:
                    cols.append("n/a")
                    continue
                vst = [r["vstar"] for r in runs]
                vstar_by_n[n] = vst
                cols.append(ms(vst, "{:.1f}"))
            exp = exponent(vstar_by_n)
            big = g.get((ns[-1], beta, norm))
            idn = ms([last_cp(r)["identification_acc"] for r in big]) if big else "n/a"
            print(f"{int(norm):<6}{beta:>6}"
                  + "".join(f"{c:>14}" for c in cols)
                  + f"{exp:>12.3f}{idn:>18}")

    # ---- Table D: overlap decomposition at V=400, beta=0.05 ----
    print("\n[D] Overlap decomposition at Vmax, beta=0.05 (resolves the "
          "0.82 paradox). within vs between vs chance; self=retrieval overlap.")
    print(f"{'norm':<6}{'n':>7}{'within':>14}{'between':>14}{'separation':>16}"
          f"{'chance':>10}{'self(retr)':>14}{'ident':>14}")
    for norm in (False, True):
        for n in ns:
            runs = g.get((n, 0.05, norm))
            if not runs:
                continue
            c0 = last_cp(runs[0])
            V = c0["V"]
            cs = [last_cp(r) for r in runs]
            wi = [c["within_cat_overlap"] for c in cs]
            bt = [c["between_cat_overlap"] for c in cs]
            sep = [c["category_separation"] for c in cs]
            ch = cs[0]["chance_overlap"]
            selfo = [c["retrieval_overlap_mean"] for c in cs]
            idn = [c["identification_acc"] for c in cs]
            t = ttest_vs_null(sep, 0.0)
            star = t.get("degenerate") or ("*" if t["significant"] else "ns")
            print(f"{int(norm):<6}{n:>7}{ms(wi):>14}{ms(bt):>14}"
                  f"{ms(sep,'{:.4f}'):>16}{ch:>10.4f}{ms(selfo):>14}"
                  f"{ms(idn):>14}  sep_test={star}")


def main():
    for mode in ("ff", "rec"):
        report_mode(mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

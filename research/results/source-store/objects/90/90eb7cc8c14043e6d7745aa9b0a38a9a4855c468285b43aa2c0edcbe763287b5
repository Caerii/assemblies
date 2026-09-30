"""Aggregate the three result JSONs into REPORT.md (tables recomputed each run)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.base import paired_ttest, ttest_vs_null  # noqa: E402

HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_text())


def ms(row, key, fmt="{:.2f}"):
    return f"{fmt.format(row[key])} +/- {fmt.format(row[key + '_sd'])}"


def test_str(t):
    if "degenerate" in t:
        return f"degenerate:{t['degenerate']}"
    star = "*" if t["significant"] else "ns"
    return f"p={t['p']:.2g} d={t['d']:.2f} {star}"


def regime_tables(rm):
    ps = sorted({r["p_s"] for r in rm["rows"]})
    ks = sorted({r["k_s"] for r in rm["rows"]})
    idx = {(r["mode"], r["p_s"], r["k_s"]): r for r in rm["rows"]}
    out = []
    for mode in ("epsilon", "sigma"):
        out.append(f"\n#### `window=\"{mode}\"`\n")
        out.append("| p_s \\ k_s | " + " | ".join(str(k) for k in ks) + " |")
        out.append("|---" * (len(ks) + 1) + "|")
        for p in ps:
            cells = []
            for k in ks:
                r = idx[(mode, p, k)]
                cells.append(f"{r['size_mean']:.1f}+/-{r['size_sd']:.1f} "
                             f"({r['formed_rate']:.2f})")
            out.append(f"| **{p}** | " + " | ".join(cells) + " |")
    return "\n".join(out)


def usable_table(rm):
    lines = ["| p_s | k_s | epsilon size (formed) | sigma size (formed) | "
             "usable for epsilon? | usable for sigma? |", "|---|---|---|---|---|---|"]
    idx = {(r["mode"], r["p_s"], r["k_s"]): r for r in rm["rows"]}
    for p in sorted({r["p_s"] for r in rm["rows"]}):
        for k in sorted({r["k_s"] for r in rm["rows"]}):
            e, s = idx[("epsilon", p, k)], idx[("sigma", p, k)]

            def flag(r):
                ok = r["size_mean"] >= 6.0 and r["formed_rate"] >= 0.7
                return "**yes**" if ok else "no"
            lines.append(
                f"| {p} | {k} | {e['size_mean']:.1f} ({e['formed_rate']:.2f}) | "
                f"{s['size_mean']:.1f} ({s['formed_rate']:.2f}) | "
                f"{flag(e)} | {flag(s)} |")
    return "\n".join(lines)


def capacity_tables(cap):
    rows = cap["rows"]
    cps = cap["params"]["checkpoints"]
    arms = []
    for r in rows:
        if (r["arm"], r["meta"]) not in arms:
            arms.append((r["arm"], r["meta"]))
    idx = {(r["arm"], r["meta"], r["V"]): r for r in rows}

    out = []
    for meta in sorted({r["meta"] for r in rows}):
        out.append(f"\n#### metaplasticity = {meta}\n")
        out.append("| arm | metric | " + " | ".join(f"V={v}" for v in cps) + " |")
        out.append("|---" * (len(cps) + 2) + "|")
        for arm, m in arms:
            if m != meta:
                continue
            for label, key, fmt in (("assembly size", "size_mean", "{:.1f}"),
                                    ("identification", "ident", "{:.2f}"),
                                    ("recovery", "recovery", "{:.2f}"),
                                    ("formed rate", "formed_rate", "{:.2f}")):
                cells = []
                for v in cps:
                    r = idx[(arm, meta, v)]
                    cells.append(f"{fmt.format(r[key])}+/-{fmt.format(r[key+'_sd'])}")
                out.append(f"| {arm} | {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def vcap_table(cap, deep):
    """V_cap over the union of the V<=128 and V<=512 grids."""
    idx = {}
    for src in (cap, deep):
        for r in src["rows"]:
            idx.setdefault((r["arm"], r["meta"]), {})[r["V"]] = r
    arms = []
    for r in cap["rows"]:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    lines = ["| arm | meta | max V tested | V_cap (ident >= 0.9) | "
             "V_cap (recovery >= 0.7) | size @ V_cap | ident @ max V |",
             "|---|---|---|---|---|---|---|"]
    for meta in sorted({r["meta"] for r in cap["rows"]}):
        for arm in arms:
            series = idx[(arm, meta)]
            cps = sorted(series)

            def first_drop(key, thr):
                # largest V such that EVERY checkpoint up to V is above thr.
                # Not `max(v: metric>=thr)`: recovery is non-monotone for the
                # E% arms, because once an assembly has collapsed to one or two
                # neurons it is trivially "recovered".
                best = 0
                for v in cps:
                    if series[v][key] < thr:
                        break
                    best = v
                return best
            vi_max = first_drop("ident", 0.9)
            vr_max = first_drop("recovery", 0.7)
            sz = f"{series[vi_max]['size_mean']:.1f}" if vi_max else "n/a"
            top = cps[-1]

            def show(v):
                return f">={top}" if v == top else str(v)
            lines.append(
                f"| {arm} | {meta} | {top} | {show(vi_max)} | {show(vr_max)} | "
                f"{sz} | {ms(series[top], 'ident')} |")
    return "\n".join(lines)


def deep_table(deep):
    cps = deep["params"]["checkpoints"]
    idx = {(r["arm"], r["meta"], r["V"]): r for r in deep["rows"]}
    arms = []
    for r in deep["rows"]:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    out = []
    for meta in sorted({r["meta"] for r in deep["rows"]}):
        out.append(f"\n#### metaplasticity = {meta}\n")
        out.append("| arm | metric | " + " | ".join(f"V={v}" for v in cps) + " |")
        out.append("|---" * (len(cps) + 2) + "|")
        for arm in arms:
            for label, key, fmt in (("assembly size", "size_mean", "{:.1f}"),
                                    ("identification", "ident", "{:.2f}"),
                                    ("recovery", "recovery", "{:.2f}")):
                cells = [f"{fmt.format(idx[(arm, meta, v)][key])}"
                         f"+/-{fmt.format(idx[(arm, meta, v)][key+'_sd'])}"
                         for v in cps]
                out.append(f"| {arm} | {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def gradient_table(cap):
    rows = [r for r in cap["rows"] if r["V"] in (32, 128)]
    lines = ["| arm | meta | V | recovery early | recovery late | gradient "
             "(late-early) | size early | size late | pairwise (chance) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["meta"], x["V"], x["arm"])):
        grad = np.array(r["recovery_late_vals"]) - np.array(r["recovery_early_vals"])
        t = ttest_vs_null(list(grad), 0.0)
        lines.append(
            f"| {r['arm']} | {r['meta']} | {r['V']} | {ms(r,'recovery_early')} | "
            f"{ms(r,'recovery_late')} | {np.mean(grad):+.2f} {test_str(t)} | "
            f"{r['size_early']:.1f} | {r['size_late']:.1f} | "
            f"{ms(r,'pairwise','{:.3f}')} ({r['pairwise_chance']:.3f}) |")
    return "\n".join(lines)


def ident_test_table(cap):
    lines = ["| arm | meta | V | ident | chance | test vs chance |",
             "|---|---|---|---|---|---|"]
    for r in sorted(cap["rows"], key=lambda x: (x["meta"], x["arm"], x["V"])):
        if r["V"] not in (32, 128):
            continue
        t = ttest_vs_null(r["ident_vals"], r["ident_chance"])
        lines.append(f"| {r['arm']} | {r['meta']} | {r['V']} | {ms(r,'ident')} | "
                     f"{r['ident_chance']:.3f} | {test_str(t)} |")
    return "\n".join(lines)


def paired_table(cap):
    idx = {(r["arm"], r["meta"], r["V"]): r for r in cap["rows"]}
    pairs = [("epsilon", "topk9"), ("sigma", "topk25"), ("epsilon", "topk25"),
             ("sigma", "topk9")]
    lines = ["| comparison | meta | V | E% ident | fixed-k ident | "
             "difference | paired test |", "|---|---|---|---|---|---|---|"]
    for meta in sorted({r["meta"] for r in cap["rows"]}):
        for V in (32, 128):
            for a, b in pairs:
                ra, rb = idx[(a, meta, V)], idx[(b, meta, V)]
                t = paired_ttest(ra["ident_vals"], rb["ident_vals"])
                lines.append(
                    f"| {a} vs {b} | {meta} | {V} | {ms(ra,'ident')} | "
                    f"{ms(rb,'ident')} | {ra['ident']-rb['ident']:+.2f} | "
                    f"p={t['p']:.2g} d={t['d']:.2f} "
                    f"{'*' if t['significant'] else 'ns'} |")
    return "\n".join(lines)


def beta_table(bc):
    lines = ["| beta | arm | assembly size | ident | recovery | "
             "pairwise (chance) | formed rate |", "|---|---|---|---|---|---|---|"]
    for r in bc["rows"]:
        lines.append(
            f"| {r['beta']} | {r['arm']} | {ms(r,'size_mean','{:.1f}')} | "
            f"{ms(r,'ident')} | {ms(r,'recovery')} | "
            f"{ms(r,'pairwise','{:.3f}')} ({r['pairwise_chance']:.3f}) | "
            f"{ms(r,'formed_rate')} |")
    return "\n".join(lines)


def main():
    rm, cap, bc, deep = (load("results_regime_map.json"),
                         load("results_capacity.json"),
                         load("results_beta_control.json"),
                         load("results_deep.json"))
    prose = (HERE / "PROSE.md").read_text()
    tables = "\n\n".join([
        "### Table 1 - Regime map: emergent assembly size (formation rate)\n"
        "Fresh substrate, one assembly, n=1000, beta=0.01, 10 seeds. Cell is "
        "`mean size +/- sd (formation rate)`; formation requires all of "
        "Eq. 8-11 plus `|A| >= 6`.\n" + regime_tables(rm),
        "### Table 2 - Which cells are usable\n"
        "`usable` = mean emergent size >= 6 (the paper's `min_size`) AND "
        "formation rate >= 0.7.\n" + usable_table(rm),
        "### Table 3 - Capacity curves on a shared substrate\n"
        "n=1000, pool=1000, k_s=100, p_s=0.5, beta=0.01, 8 seeds. Fixed-k arms "
        "are matched to each E% arm's V=1 emergent size.\n" + capacity_tables(cap),
        "### Table 4 - Capacity summary\n"
        "`V_cap` is the largest tested V whose mean stays at/above threshold; "
        "it is quantised to the checkpoint grid.\n" + vcap_table(cap, deep),
        "### Table 5 - Forgetting gradient and pairwise overlap\n"
        "`early`/`late` are the first and last fifth of the stored items. A "
        "negative gradient means EARLY items retrieve better than LATE ones "
        "(anterograde); positive means classic retrograde forgetting.\n"
        + gradient_table(cap),
        "### Table 6 - Identification against chance (1/V)\n" + ident_test_table(cap),
        "### Table 7 - E%-WTA vs fixed-k, paired by seed\n" + paired_table(cap),
        "### Table 8 - beta control (V=64, no metaplasticity, 6 seeds)\n"
        "At beta=0 nothing is written: assemblies are read off a static "
        "substrate.\n" + beta_table(bc),
        "### Table 9 - Deep run, V up to 512 (4 seeds)\n"
        "Same operating point; separate seeds from Table 3, so the V=128 "
        "column is an independent replication of it.\n" + deep_table(deep),
    ])
    (HERE / "REPORT.md").write_text(prose.rstrip() + "\n\n---\n\n## Tables\n\n"
                                    + tables + "\n")
    print("wrote REPORT.md")


if __name__ == "__main__":
    main()

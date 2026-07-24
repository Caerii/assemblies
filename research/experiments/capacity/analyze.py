"""
Aggregate the capacity experiments into a markdown report.

Reads:
    results_lexicon_capacity.json    (synthetic shared-substrate lexicon)
    results_parser_recruitment.json  (real EmergentParser core areas)

Writes:
    REPORT.md

All numbers are means over seeds with the spread quoted; every headline claim
is tested against an explicit null via research.experiments.base.ttest_vs_null.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.experiments.base import summarize, ttest_vs_null  # noqa: E402

HERE = Path(__file__).parent
RECRUIT_FLOOR = 0.05      # "recruitment has stopped" threshold
WINDOW = 10               # words averaged when locating the horizon


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def ms(values: Sequence[float], fmt: str = "{:.3f}") -> str:
    """mean +/- sd over seeds."""
    v = [x for x in values if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return "n/a"
    s = summarize(list(v))
    if len(v) == 1:
        return fmt.format(s["mean"])
    return f"{fmt.format(s['mean'])} +/- {fmt.format(s['std'])}"


def sig(values: Sequence[float], null: float) -> str:
    t = ttest_vs_null(list(values), null)
    if t.get("degenerate"):
        return f"degenerate:{t['degenerate']}"
    star = "*" if t["significant"] else "ns"
    return f"p={t['p']:.2g} d={t['d']:.2f} {star}"


def recruit_horizon(per_word: List[Dict]) -> int:
    """Vocabulary size past which recruitment never resumes.

    The smallest V such that EVERY window of WINDOW consecutive words
    starting at or after V has mean recruit fraction below RECRUIT_FLOOR.
    Taking the last qualifying window rather than the first makes the
    statistic robust to a single quiet stretch early on, which matters for
    the beta=0 control where recruitment is noisy but persistent.
    """
    fr = [d["recruit_frac"] for d in per_word]
    last_active = 0
    for i in range(len(fr) - WINDOW + 1):
        if float(np.mean(fr[i:i + WINDOW])) >= RECRUIT_FLOOR:
            last_active = i + 1
    return min(last_active + WINDOW, len(fr))


def final_wn(run: Dict) -> float:
    """Final w/n taken from the last learned word.

    `run["final_w_over_n"]` reads `area.w` after training, but when a run
    ends by pool exhaustion the last thing that happened was an
    inhibit_areas() call, and the Area.winners setter resets .w to 0. The
    per-word record is taken before that, so it is the trustworthy value.
    """
    if run["per_word"]:
        return float(run["per_word"][-1]["w_over_n"])
    return float(run["final_w_over_n"])


def last_common_checkpoint(runs: List[Dict]) -> int | None:
    sets = [{c["V"] for c in r["checkpoints"]} for r in runs]
    common = set.intersection(*sets) if sets else set()
    return max(common) if common else None


def decile_means(per_word: List[Dict], key: str, n_bins: int = 8) -> List[float]:
    v = np.array([d[key] for d in per_word], dtype=float)
    return [float(np.mean(b)) for b in np.array_split(v, n_bins)]


def key_of(run: Dict) -> tuple:
    c = run["config"]
    return (c["n"], c["policy"], c["mode"], c["beta"])


def group(runs: List[Dict]) -> Dict[tuple, List[Dict]]:
    g = defaultdict(list)
    for r in runs:
        g[key_of(r)].append(r)
    return g


def cp(run: Dict, V: int) -> Dict | None:
    for c in run["checkpoints"]:
        if c["V"] == V:
            return c
    return None


# ----------------------------------------------------------------------
# report sections
# ----------------------------------------------------------------------

def section_capacity_curve(g) -> str:
    out = ["### Table 1 — Capacity curve (fixed-k TopK, feedforward, beta=0.05)\n",
           "`V*` is the vocabulary size at which recruitment stops "
           f"(mean recruit fraction over {WINDOW} consecutive words < "
           f"{RECRUIT_FLOOR}). `slots = n/k` is how many disjoint "
           "assemblies the area could hold if it tiled perfectly.\n",
           "| n | slots = n/k | V* (recruitment horizon) | V*/slots | "
           "final w/n | w/k (slots consumed) |",
           "|---|---|---|---|---|---|"]
    for (n, pol, mode, beta), runs in sorted(g.items()):
        if pol != "topk" or mode != "ff" or beta != 0.05:
            continue
        k = runs[0]["config"]["k"]
        slots = n / k
        vstar = [recruit_horizon(r["per_word"]) for r in runs]
        wn = [final_wn(r) for r in runs]
        wk = [final_wn(r) * n / k for r in runs]
        out.append(f"| {n} | {slots:.0f} | {ms(vstar, '{:.1f}')} | "
                   f"{ms([v/slots for v in vstar], '{:.2f}')} | "
                   f"{ms(wn)} | {ms(wk, '{:.0f}')} |")
    return "\n".join(out) + "\n"


def section_recruit_vs_null(g) -> str:
    out = ["### Table 2 — Recruit fraction vs the random-tiling null\n",
           "Observed recruit fraction per word, averaged in eighths of the "
           "400-word vocabulary, against the coupon-collector null "
           "`1 - w/n` (what an unbiased winner-take-all would give). "
           "TopK, feedforward, beta=0.05.\n",
           "| n | source | 1-50 | 51-100 | 101-150 | 151-200 | 201-250 | "
           "251-300 | 301-350 | 351-400 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for (n, pol, mode, beta), runs in sorted(g.items()):
        if pol != "topk" or mode != "ff" or beta != 0.05:
            continue
        obs = np.mean([decile_means(r["per_word"], "recruit_frac") for r in runs], axis=0)
        nul = np.mean([decile_means(r["per_word"], "recruit_frac_null") for r in runs], axis=0)
        out.append(f"| {n} | observed | " + " | ".join(f"{x:.2f}" for x in obs) + " |")
        out.append(f"| {n} | null | " + " | ".join(f"{x:.2f}" for x in nul) + " |")
    out.append("")
    out.append("Test of `recruit_excess = observed - null` over the second "
               "half of the vocabulary, against H0: excess = 0.\n")
    out.append("| n | mean excess (words 201-400) | test |")
    out.append("|---|---|---|")
    for (n, pol, mode, beta), runs in sorted(g.items()):
        if pol != "topk" or mode != "ff" or beta != 0.05:
            continue
        ex = [float(np.mean([d["recruit_excess"] for d in r["per_word"][200:]]))
              for r in runs]
        out.append(f"| {n} | {ms(ex)} | {sig(ex, 0.0)} |")
    return "\n".join(out) + "\n"


def section_consequences(g, checkpoints) -> str:
    out = ["### Table 3 — What breaks past the transition\n",
           "TopK, feedforward, beta=0.05. `pairwise` is the mean overlap "
           "between stored assemblies (null: chance = k/n). `ident` is the "
           "fraction of words whose retrieved assembly is nearest to their "
           "OWN stored assembly (null: 1/V). `retr early`/`retr late` are "
           "retrieval self-overlap for the first and last tenth of the "
           "vocabulary — a forgetting gradient would show early << late.\n",
           "| n | V | w/n | pairwise (chance) | pairwise test | ident (chance) "
           "| ident test | retr early | retr late |",
           "|---|---|---|---|---|---|---|---|---|"]
    for (n, pol, mode, beta), runs in sorted(g.items()):
        if pol != "topk" or mode != "ff" or beta != 0.05:
            continue
        for V in checkpoints:
            cs = [cp(r, V) for r in runs]
            cs = [c for c in cs if c]
            if not cs:
                continue
            chance = cs[0]["chance_overlap"]
            pw = [c["pairwise_overlap_mean"] for c in cs]
            idn = [c["identification_acc"] for c in cs]
            out.append(
                f"| {n} | {V} | {ms([c['w_over_n'] for c in cs])} | "
                f"{ms(pw)} ({chance:.3f}) | {sig(pw, chance)} | "
                f"{ms(idn)} ({1.0/V:.3f}) | {sig(idn, 1.0/V)} | "
                f"{ms([c['retrieval_overlap_first_decile'] for c in cs])} | "
                f"{ms([c['retrieval_overlap_last_decile'] for c in cs])} |")
    return "\n".join(out) + "\n"


def section_beta(g, V: int) -> str:
    out = ["### Table 4 — Plasticity strength is what stops recruitment "
           "(TopK, feedforward)\n",
           "beta=0 is the random-tiling control: no Hebbian bias toward "
           "neurons that already fired. `V learned` is how many words the run "
           "got through before the area ran out of never-fired neurons and "
           "training aborted; consequence metrics are quoted at `V meas`, the "
           "last checkpoint every seed of that cell reached.\n",
           "| n | beta | V learned | V meas | V* | final w/n | "
           "pairwise overlap | ident acc | retr early |",
           "|---|---|---|---|---|---|---|---|---|"]
    for (n, pol, mode, beta), runs in sorted(g.items()):
        if pol != "topk" or mode != "ff":
            continue
        Vm = last_common_checkpoint(runs) or V
        cs = [c for c in (cp(r, Vm) for r in runs) if c]
        out.append(
            f"| {n} | {beta} | "
            f"{ms([r['vocab_learned'] for r in runs], '{:.0f}')} | {Vm} | "
            f"{ms([recruit_horizon(r['per_word']) for r in runs], '{:.0f}')} | "
            f"{ms([final_wn(r) for r in runs])} | "
            f"{ms([c['pairwise_overlap_mean'] for c in cs])} | "
            f"{ms([c['identification_acc'] for c in cs])} | "
            f"{ms([c['retrieval_overlap_first_decile'] for c in cs])} |")
    return "\n".join(out) + "\n"


def section_scaling(g) -> str:
    """Log-log fit of the recruitment horizon against area size."""
    out = ["### Table 4b — How capacity scales with area size\n",
           "Least-squares slope of `log V*` against `log n` over "
           "n = 1000 / 3000 / 10000 (TopK, feedforward). An area whose "
           "capacity were simply its number of assembly-sized slots would "
           "give an exponent of 1.\n",
           "| beta | V* @ n=1000 | V* @ n=3000 | V* @ n=10000 | "
           "scaling exponent |",
           "|---|---|---|---|---|"]
    for beta in (0.0, 0.01, 0.05, 0.2):
        ns, vs, cols = [], [], []
        for n in (1000, 3000, 10000):
            runs = g.get((n, "topk", "ff", beta))
            if not runs:
                cols.append("n/a")
                continue
            v = [recruit_horizon(r["per_word"]) for r in runs]
            ns.append(n)
            vs.append(float(np.mean(v)))
            cols.append(ms(v, "{:.0f}"))
        exp = (f"{np.polyfit(np.log(ns), np.log(vs), 1)[0]:.2f}"
               if len(ns) == 3 else "n/a")
        out.append(f"| {beta} | " + " | ".join(cols) + f" | {exp} |")
    return "\n".join(out) + "\n"


def section_category(g) -> str:
    out = ["### Table 4c — Category structure in the input does not survive "
           "into the assemblies\n",
           "Words in the same category share 10 of their 30 PHON input "
           "neurons; words in different categories share only what the "
           "random draw gives them. If that structure were carried into LEX, "
           "within-category assembly overlap would exceed between-category "
           "overlap. TopK, feedforward, at the last checkpoint of each cell. "
           "H0: separation = 0.\n",
           "| n | beta | V | within-cat overlap | between-cat overlap | "
           "separation | test |",
           "|---|---|---|---|---|---|---|"]
    for (n, pol, mode, beta), runs in sorted(g.items()):
        if pol != "topk" or mode != "ff":
            continue
        Vm = last_common_checkpoint(runs)
        cs = [c for c in (cp(r, Vm) for r in runs) if c]
        if not cs:
            continue
        sep = [c["category_separation"] for c in cs]
        out.append(f"| {n} | {beta} | {Vm} | "
                   f"{ms([c['within_cat_overlap'] for c in cs])} | "
                   f"{ms([c['between_cat_overlap'] for c in cs])} | "
                   f"{ms(sep, '{:.4f}')} | {sig(sep, 0.0)} |")
    return "\n".join(out) + "\n"


def section_policy(g, V: int) -> str:
    out = [f"### Table 5 — Fixed-k TopK vs E%-WTA (feedforward, beta=0.05, V={V})\n",
           "`size` is the emergent assembly size averaged over the whole "
           "vocabulary; TopK pins it at k=30 by construction.\n",
           "| n | policy | assembly size | V* | final w/n | pairwise overlap "
           "(chance) | ident acc (chance) | ident test |",
           "|---|---|---|---|---|---|---|---|"]
    for (n, pol, mode, beta), runs in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if mode != "ff" or beta != 0.05:
            continue
        cs = [c for c in (cp(r, V) for r in runs) if c]
        if not cs:
            continue
        sizes = [float(np.mean([d["size"] for d in r["per_word"]])) for r in runs]
        idn = [c["identification_acc"] for c in cs]
        out.append(
            f"| {n} | {pol} | {ms(sizes, '{:.1f}')} | "
            f"{ms([recruit_horizon(r['per_word']) for r in runs], '{:.0f}')} | "
            f"{ms([final_wn(r) for r in runs])} | "
            f"{ms([c['pairwise_overlap_mean'] for c in cs])} "
            f"({cs[0]['chance_overlap']:.3f}) | "
            f"{ms(idn)} ({1.0/V:.3f}) | {sig(idn, 1.0/V)} |")
    return "\n".join(out) + "\n"


def section_recurrent(g, V: int) -> str:
    out = [f"### Table 6 — Control: engaging the plastic recurrent fiber "
           f"(TopK, beta=0.05, V={V})\n",
           "| n | mode | V* | final w/n | pairwise overlap (chance) | "
           "ident acc (chance) | ident test |",
           "|---|---|---|---|---|---|---|"]
    for (n, pol, mode, beta), runs in sorted(g.items()):
        if pol != "topk" or beta != 0.05:
            continue
        cs = [c for c in (cp(r, V) for r in runs) if c]
        if not cs:
            continue
        idn = [c["identification_acc"] for c in cs]
        out.append(
            f"| {n} | {mode} | "
            f"{ms([recruit_horizon(r['per_word']) for r in runs], '{:.0f}')} | "
            f"{ms([final_wn(r) for r in runs])} | "
            f"{ms([c['pairwise_overlap_mean'] for c in cs])} "
            f"({cs[0]['chance_overlap']:.3f}) | "
            f"{ms(idn)} ({1.0/V:.3f}) | {sig(idn, 1.0/V)} |")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------
# parser section
# ----------------------------------------------------------------------

def section_parser(rows: List[Dict], areas_of_interest=("NOUN_CORE", "VERB_CORE",
                                                        "ADJ_CORE", "DET_CORE")) -> str:
    out = ["### Table 7 — Real EmergentParser core lexical areas\n",
           "Whole vocabulary trained into the core areas via "
           "`train_lexicon()`. `tiling` is `w / (words * k)`: 1.0 means the "
           "area is tiled with perfectly disjoint assemblies, below 1.0 "
           "means words share neurons, above 1.0 means multi-round training "
           "materialised transient winners that no stored assembly kept.\n",
           "| n | vocab | area | words | w/n | tiling | recruit Q1 | "
           "recruit Q4 | null Q4 | pairwise (chance) | ident (chance) | "
           "retr Q1 | retr Q4 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in rows:
        if "error" in r or r.get("depth") != "LEXICON":
            continue
        for a in r["areas"]:
            if a["area"] in areas_of_interest:
                by[(r["n"], r["vocab_size"], a["area"])].append(a)
    for (n, vs, area), aa in sorted(by.items()):
        out.append(
            f"| {n} | {vs} | {area} | {ms([a['n_words'] for a in aa], '{:.0f}')} | "
            f"{ms([a['w_over_n'] for a in aa])} | "
            f"{ms([a['tiling_ratio'] for a in aa], '{:.2f}')} | "
            f"{ms([a['recruit_frac_q'][0] for a in aa], '{:.2f}')} | "
            f"{ms([a['recruit_frac_q'][3] for a in aa], '{:.2f}')} | "
            f"{ms([a['recruit_null_q'][3] for a in aa], '{:.2f}')} | "
            f"{ms([a['pairwise_overlap_mean'] for a in aa])} "
            f"({aa[0]['chance_overlap']:.3f}) | "
            f"{ms([a['identification_acc'] for a in aa], '{:.2f}')} "
            f"({aa[0]['identification_chance']:.3f}) | "
            f"{ms([a['retrieval_overlap_q'][0] for a in aa], '{:.2f}')} | "
            f"{ms([a['retrieval_overlap_q'][3] for a in aa], '{:.2f}')} |")

    out.append("")
    out.append("Identification accuracy against its null (chance = 1/words), "
               "per cell:\n")
    out.append("| n | vocab | area | ident acc | test |")
    out.append("|---|---|---|---|---|")
    for (n, vs, area), aa in sorted(by.items()):
        idn = [a["identification_acc"] for a in aa]
        out.append(f"| {n} | {vs} | {area} | {ms(idn, '{:.2f}')} | "
                   f"{sig(idn, aa[0]['identification_chance'])} |")

    fails = [(r["n"], r["vocab_size"], a["area"], a["probe_failures"])
             for r in rows if "error" not in r and r.get("depth") == "LEXICON"
             for a in r["areas"] if a.get("probe_failures")]
    if fails:
        out.append("")
        out.append("Cells where retrieval itself could not run because the "
                   "area had fewer than k never-fired neurons left "
                   "(n, vocab, area, failed probes): " +
                   ", ".join(str(f) for f in fails))
    return "\n".join(out) + "\n"


def section_parser_twoword(rows: List[Dict]) -> str:
    tw = [r for r in rows if r.get("depth") == "TWO_WORD" and "error" not in r]
    if not tw:
        return ""
    out = ["### Table 8 — The motivating measurement, reproduced "
           "(TWO_WORD curriculum, n=3000, k=30)\n",
           "| area | words | w | w/n | words*k/n | tiling | pairwise (chance) "
           "| ident (chance) |",
           "|---|---|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in tw:
        for a in r["areas"]:
            by[a["area"]].append(a)
    for area, aa in sorted(by.items()):
        out.append(
            f"| {area} | {ms([a['n_words'] for a in aa], '{:.0f}')} | "
            f"{ms([a['w'] for a in aa], '{:.0f}')} | "
            f"{ms([a['w_over_n'] for a in aa])} | "
            f"{ms([a['words_times_k_over_n'] for a in aa], '{:.2f}')} | "
            f"{ms([a['tiling_ratio'] for a in aa], '{:.2f}')} | "
            f"{ms([a['pairwise_overlap_mean'] for a in aa])} "
            f"({aa[0]['chance_overlap']:.3f}) | "
            f"{ms([a['identification_acc'] for a in aa], '{:.2f}')} "
            f"({aa[0]['identification_chance']:.3f}) |")
    return "\n".join(out) + "\n"


def main() -> int:
    lex_path = HERE / "results_lexicon_capacity.json"
    par_path = HERE / "results_parser_recruitment.json"
    runs = json.loads(lex_path.read_text()) if lex_path.exists() else []
    prows = json.loads(par_path.read_text()) if par_path.exists() else []

    parts = []
    if runs:
        g = group(runs)
        cps = [c["V"] for c in runs[0]["checkpoints"]]
        vlast = cps[-1]
        parts += [section_capacity_curve(g), section_recruit_vs_null(g),
                  section_consequences(g, cps), section_beta(g, vlast),
                  section_scaling(g), section_category(g),
                  section_policy(g, vlast), section_recurrent(g, vlast)]
    if prows:
        parts += [section_parser(prows), section_parser_twoword(prows)]

    body = "\n".join(parts)
    (HERE / "TABLES.md").write_text(body, encoding="utf-8")

    prose_path = HERE / "PROSE.md"
    if prose_path.exists():
        (HERE / "REPORT.md").write_text(
            prose_path.read_text(encoding="utf-8") + body, encoding="utf-8")
        print(f"wrote {HERE / 'REPORT.md'}")
    print(f"wrote {HERE / 'TABLES.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

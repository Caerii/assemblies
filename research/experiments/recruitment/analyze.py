"""Aggregate `results_recruitment.json` (+ `results_beta_pass.json`) into REPORT.md.

    python -m research.experiments.recruitment.analyze
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.experiments.base import summarize, ttest_vs_null  # noqa: E402
from research.experiments.capacity.analyze import (  # noqa: E402
    RECRUIT_FLOOR,
    WINDOW,
    final_wn,
    recruit_horizon,
)

HERE = Path(__file__).parent


# ---------------------------------------------------------------- helpers

def ms(values: Sequence[float], fmt: str = "{:.3f}") -> str:
    v = [x for x in values
         if x is not None and not (isinstance(x, float) and np.isnan(x))]
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
    return f"p={t['p']:.2g} d={t['d']:.2f} {'*' if t['significant'] else 'ns'}"


def cp_at(run: Dict, V: int):
    for c in run["checkpoints"]:
        if c["V"] == V:
            return c
    return None


def last_common_cp(runs: List[Dict]):
    sets = [{c["V"] for c in r["checkpoints"]} for r in runs]
    common = set.intersection(*sets) if sets else set()
    return max(common) if common else None


def cps_of(runs: List[Dict]):
    """(V, cp-per-run) at the last checkpoint SHARED by every run, or (None, []).

    Cells that end by pool exhaustion stop at a seed-dependent V, so there may
    be no common checkpoint at all; those cells are reported without
    checkpoint-derived columns rather than silently compared at mismatched V.
    """
    V = last_common_cp(runs)
    if V is None:
        return None, []
    cps = [cp_at(r, V) for r in runs]
    if any(c is None for c in cps):
        return None, []
    return V, cps


def exponent(vstar_by_n: Dict[int, List[float]]) -> float:
    ns = sorted(vstar_by_n)
    x = np.log(np.array(ns, dtype=float))
    y = np.log(np.array([float(np.mean(vstar_by_n[n])) for n in ns]))
    return float(np.polyfit(x, y, 1)[0])


def exponent_per_seed(vstar_by_n: Dict[int, List[float]]) -> List[float]:
    """One slope per seed, so the exponent gets an honest sd."""
    ns = sorted(vstar_by_n)
    m = min(len(vstar_by_n[n]) for n in ns)
    x = np.log(np.array(ns, dtype=float))
    out = []
    for s in range(m):
        y = np.log(np.array([max(1.0, float(vstar_by_n[n][s])) for n in ns]))
        out.append(float(np.polyfit(x, y, 1)[0]))
    return out


def cell_key(run: Dict) -> tuple:
    c = run["config"]
    return (c["n"], c["beta"], c["mode"], round(c["refracted_strength"], 4),
            c["lri_period"], c["lri_strength"], bool(c["synaptic_scaling"]))


def group(runs: List[Dict]) -> Dict[tuple, List[Dict]]:
    g: Dict[tuple, List[Dict]] = {}
    for r in runs:
        g.setdefault(cell_key(r), []).append(r)
    return g


def tail_excess(run: Dict, frac: float = 0.5) -> float:
    """Mean (recruit_frac - null) over the LAST `frac` of the learned words."""
    pw = run["per_word"]
    if not pw:
        return float("nan")
    start = int(len(pw) * (1 - frac))
    seg = [d["recruit_excess"] for d in pw[start:]]
    return float(np.mean(seg)) if seg else float("nan")


def tail_recruit(run: Dict, frac: float = 0.5) -> float:
    pw = run["per_word"]
    if not pw:
        return float("nan")
    start = int(len(pw) * (1 - frac))
    seg = [d["recruit_frac"] for d in pw[start:]]
    return float(np.mean(seg)) if seg else float("nan")


def tail_null(run: Dict, frac: float = 0.5) -> float:
    pw = run["per_word"]
    if not pw:
        return float("nan")
    start = int(len(pw) * (1 - frac))
    seg = [d["recruit_frac_null"] for d in pw[start:]]
    return float(np.mean(seg)) if seg else float("nan")


def churn(run: Dict) -> float:
    """Neurons MATERIALISED per neuron that ends up in the stored assembly.

    `w_after - w_before` counts every neuron that fired for the first time at
    any point during the word's `train_rounds` projection rounds; `recruited`
    counts only those still present in the assembly that gets stored. The
    ratio is > 1 exactly when the assembly failed to settle -- each round
    threw off a different set of first-time winners. It is the cost side of
    any mechanism that penalises recently-fired neurons, because such a
    mechanism penalises the current word's own assembly between its rounds.
    """
    num = sum(d["w_after"] - d["w_before"] for d in run["per_word"])
    den = sum(d["recruited"] for d in run["per_word"])
    return num / den if den else float("nan")


def n_exhausted(runs: List[Dict]) -> int:
    return sum(1 for r in runs if r.get("exhausted_at") is not None)


# ---------------------------------------------------------------- sections

def sec_refracted_capacity(g) -> str:
    keys = sorted(k for k in g
                  if k[1] == 0.05 and k[2] == "ff" and k[4] == 0 and not k[6])
    out = ["### Table 1 -- Refracted suppression: capacity and recruitment",
           "",
           "`rs` is `refracted_strength`, the per-firing increment to a "
           "neuron's cumulative input penalty. `V*` is the recruitment horizon "
           f"(no {WINDOW}-word window past it has mean recruit fraction >= "
           f"{RECRUIT_FLOOR}). `V learned` is how many words trained before the "
           "sparse engine ran out of never-fired neurons; `exh` counts seeds "
           "that hit that hard wall (where V* is right-censored). "
           "`tail recruit` / `tail null` are the mean recruit fraction and the "
           "`1 - w/n` coupon-collector null over the LAST HALF of the words "
           "each run got through.", "",
           "`churn` is neurons materialised per neuron that survives into the "
           "stored assembly, summed over the run: 1.0 means every first-time "
           "winner stayed in the assembly, >1 means the assembly never settled "
           "within its 6 training rounds.", "",
           "`delta V*` is paired by seed against the rs=0 cell at the same n.",
           "",
           "| n | rs | V learned | exh/5 | V* | delta V* vs rs=0 | final w/n | "
           "churn | tail recruit | tail null | tail excess | excess test |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        runs = g[k]
        n, rs = k[0], k[3]
        ex = [tail_excess(r) for r in runs]
        vs = [recruit_horizon(r["per_word"]) for r in runs]
        base = g.get((n, 0.05, "ff", 0.0, 0, 0.0, False))
        if base and rs:
            b = [recruit_horizon(r["per_word"]) for r in base]
            d = [x - y for x, y in zip(vs, b)]
            dcol = f"{ms(d, '{:+.1f}')} {sig(d, 0.0)}"
        else:
            dcol = "--"
        out.append(
            f"| {n} | {rs} | {ms([r['vocab_learned'] for r in runs], '{:.0f}')} "
            f"| {n_exhausted(runs)} "
            f"| {ms(vs, '{:.0f}')} | {dcol} "
            f"| {ms([final_wn(r) for r in runs])} "
            f"| {ms([churn(r) for r in runs], '{:.2f}')} "
            f"| {ms([tail_recruit(r) for r in runs], '{:.2f}')} "
            f"| {ms([tail_null(r) for r in runs], '{:.2f}')} "
            f"| {ms(ex)} | {sig(ex, 0.0)} |")
    return "\n".join(out)


def sec_exponent(g, betas=(0.05,)) -> str:
    out = ["### Table 2 -- The headline: capacity exponent `d log V* / d log n`",
           "",
           "Least-squares slope of `log V*` against `log n` over "
           "n = 1000 / 3000 / 10000. Prior work (`capacity/REPORT.md` Table 4b) "
           "measured 0.84 at beta=0 and 0.29 at beta=0.05; the question is "
           "whether a mechanism recovers the beta=0 exponent while keeping "
           "beta > 0. `exp (per-seed)` refits the slope separately for each "
           "seed, so the sd is real rather than a fit residual. Rows where "
           "some seed hit the hard pool wall are marked `censored` -- V* there "
           "is a lower bound and the exponent is not interpretable.", "",
           "`V learned` is words trained before the hard pool wall (400 = the "
           "vocabulary ran out first, i.e. right-censored by the protocol "
           "rather than by the model), and `exp(V learned)` is the same slope "
           "fitted to it. When a mechanism drives the area to the wall, `V*` "
           "and `V learned` coincide and BOTH are the honest capacity; when it "
           "does not, `V learned` is pinned at 400 and only `V*` means "
           "anything. Read the pair, not either alone.", "",
           "`exp 3k->10k` is the two-point slope over the uncensored part of "
           "the range: at n=1000 a strong mechanism hits the pool wall and V* "
           "is a lower bound, which drags the three-point fit. Where the "
           "three-point and two-point slopes disagree, the two-point one is "
           "the safer read.", "",
           "| condition | V* n=1000 | V* n=3000 | V* n=10000 | exponent | "
           "exp (per-seed) | exp 3k->10k | V learned 1k/3k/10k | "
           "exp(V learned) | wall |",
           "|---|---|---|---|---|---|---|---|---|---|"]

    def emit(label, sel):
        by_n, vl, cens = {}, {}, 0
        for n in (1000, 3000, 10000):
            k = sel(n)
            if k not in g:
                return
            by_n[n] = [recruit_horizon(r["per_word"]) for r in g[k]]
            vl[n] = [r["vocab_learned"] for r in g[k]]
            cens += n_exhausted(g[k])
        per = exponent_per_seed(by_n)
        out.append(
            f"| {label} | {ms(by_n[1000], '{:.0f}')} | {ms(by_n[3000], '{:.0f}')}"
            f" | {ms(by_n[10000], '{:.0f}')} | {exponent(by_n):.2f} | "
            f"{ms(per, '{:.2f}')} | "
            f"{exponent({k2: by_n[k2] for k2 in (3000, 10000)}):.2f} | "
            f"{ms(vl[1000], '{:.0f}')} / {ms(vl[3000], '{:.0f}')} / "
            f"{ms(vl[10000], '{:.0f}')} | {exponent(vl):.2f} | "
            f"{cens}/15 |")

    for beta in betas:
        rss = sorted({k[3] for k in g if k[1] == beta and k[2] == "ff"
                      and k[4] == 0 and not k[6]})
        for rs in rss:
            emit(f"beta={beta} refracted rs={rs}",
                 lambda n, b=beta, r=rs: (n, b, "ff", r, 0, 0.0, False))
    # scaling + lri + rec
    emit("beta=0.05 synaptic_scaling (ff)",
         lambda n: (n, 0.05, "ff", 0.0, 0, 0.0, True))
    emit("beta=0.05 recurrent (rec), no scaling",
         lambda n: (n, 0.05, "rec", 0.0, 0, 0.0, False))
    emit("beta=0.05 recurrent + synaptic_scaling",
         lambda n: (n, 0.05, "rec", 0.0, 0, 0.0, True))
    emit("beta=0.05 LRI period=20 strength=0.5",
         lambda n: (n, 0.05, "ff", 0.0, 20, 0.5, False))
    return "\n".join(out)


def sec_consequences(g) -> str:
    out = ["### Table 3 -- What it costs: overlap, identification, forgetting",
           "",
           "Measured at the last checkpoint every seed of the cell reached "
           "(`V meas`). `pairwise` is mean overlap between stored assemblies "
           "(chance = k/n). `ident` is the fraction of words retrieved nearest "
           "to their OWN stored assembly (chance = 1/V). `retr early` / "
           "`retr late` are retrieval self-overlap for the first and last "
           "tenth of the vocabulary: early << late is retrograde forgetting.",
           "",
           "`probes ok` is how many of the V words could be probed at all: "
           "retrieval needs the area to still have k never-fired neurons "
           "available, so a mechanism that fills the area destroys its own "
           "read-out. 0 there means every metric to its right is undefined.",
           "",
           "| n | condition | V meas | probes ok | pairwise (chance) | "
           "pairwise test | ident (chance) | ident test | retr early | "
           "retr late |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(g):
        n, beta, mode, rs, lp, lsr, ss = k
        if beta != 0.05:
            continue
        runs = g[k]
        V, cps = cps_of(runs)
        if V is None:
            continue
        pw = [c["pairwise_overlap_mean"] for c in cps]
        ide = [c["identification_acc"] for c in cps]
        chance_pw = 30.0 / n
        cond = (f"rs={rs}" if rs else
                ("synaptic_scaling" if ss and mode == "ff" else
                 ("rec+scaling" if ss else
                  ("rec" if mode == "rec" else
                   (f"LRI {lp}/{lsr}" if lp else "baseline")))))
        out.append(
            f"| {n} | {cond} | {V} | "
            f"{ms([c['probes_ok'] for c in cps], '{:.0f}')}/{V} | "
            f"{ms(pw)} ({chance_pw:.3f}) | "
            f"{sig(pw, chance_pw)} | {ms(ide)} ({1.0/V:.3f}) | "
            f"{sig(ide, 1.0/V)} | "
            f"{ms([c['retrieval_overlap_first_decile'] for c in cps])} | "
            f"{ms([c['retrieval_overlap_last_decile'] for c in cps])} |")
    return "\n".join(out)


def sec_nobias(g) -> str:
    out = ["### Table 4 -- Refracted bias as a read-out distortion",
           "",
           "The cumulative bias is not cleared for retrieval by default: the "
           "clone inherits it, and an assembly learned early carries more "
           "accumulated suppression than one learned late. This table repeats "
           "the final probe on a clone whose bias has been zeroed "
           "(`Brain.clear_refracted_bias`), which separates the mechanism's "
           "effect on STORAGE from its effect on READ-OUT.", "",
           "`n/a` rows are ones where EVERY probe failed: the mechanism had "
           "filled the area past the point where a retrieval drive can still "
           "find k never-fired neurons.", "",
           "| n | rs | V meas | probes ok | ident (bias on) | "
           "ident (bias cleared) | retr early on | retr early cleared | "
           "retr late on | retr late cleared |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(g):
        n, beta, mode, rs, lp, lsr, ss = k
        if beta != 0.05 or mode != "ff" or lp or ss or not rs:
            continue
        runs = [r for r in g[k] if r.get("probe_nobias")
                and "error" not in r["probe_nobias"]]
        if not runs:
            continue
        Vs = {r["probe_nobias"]["V"] for r in runs}
        on = [cp_at(r, r["probe_nobias"]["V"]) for r in runs]
        off = [r["probe_nobias"] for r in runs]
        if any(o is None for o in on):
            continue
        out.append(
            f"| {n} | {rs} | {sorted(Vs)} "
            f"| {ms([c['probes_ok'] for c in on], '{:.0f}')} "
            f"| {ms([c['identification_acc'] for c in on])} "
            f"| {ms([c['identification_acc'] for c in off])} "
            f"| {ms([c['retrieval_overlap_first_decile'] for c in on])} "
            f"| {ms([c['retrieval_overlap_first_decile'] for c in off])} "
            f"| {ms([c['retrieval_overlap_last_decile'] for c in on])} "
            f"| {ms([c['retrieval_overlap_last_decile'] for c in off])} |")
    return "\n".join(out)


def sec_scaling(g) -> str:
    out = ["### Table 5 -- Synaptic scaling (paired against its own control)",
           "",
           "`synaptic_scaling=True` renormalises area->area fibers. In `ff` "
           "mode the only area->area fiber is PHON->LEX; in `rec` mode LEX->LEX "
           "is added. Paired by seed against the matching scaling-off cell.",
           "",
           "| n | mode | V* off | V* on | delta V* (test vs 0) | "
           "final w/n off | final w/n on | pairwise off | pairwise on | "
           "ident off | ident on |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in (1000, 3000, 10000):
        for mode in ("ff", "rec"):
            koff = (n, 0.05, mode, 0.0, 0, 0.0, False)
            kon = (n, 0.05, mode, 0.0, 0, 0.0, True)
            if koff not in g or kon not in g:
                continue
            ro, rn = g[koff], g[kon]
            Vo, co = cps_of(ro)
            Vn, cn = cps_of(rn)
            vo = [recruit_horizon(r["per_word"]) for r in ro]
            vn = [recruit_horizon(r["per_word"]) for r in rn]
            d = [b - a for a, b in zip(vo, vn)]
            same = Vo is not None and Vo == Vn
            out.append(
                f"| {n} | {mode} | {ms(vo, '{:.0f}')} | {ms(vn, '{:.0f}')} | "
                f"{ms(d, '{:+.1f}')} {sig(d, 0.0)} | "
                f"{ms([final_wn(r) for r in ro])} | "
                f"{ms([final_wn(r) for r in rn])} | "
                + (f"{ms([c['pairwise_overlap_mean'] for c in co])} | "
                   f"{ms([c['pairwise_overlap_mean'] for c in cn])} | "
                   f"{ms([c['identification_acc'] for c in co])} | "
                   f"{ms([c['identification_acc'] for c in cn])} |"
                   if same else "n/a | n/a | n/a | n/a |"))
    return "\n".join(out)


def sec_lri(g) -> str:
    out = ["### Table 6 -- LRI (finite-memory refractory suppression)",
           "",
           "`refractory_period` steps of linearly decaying penalty "
           "`inhibition_strength` on neurons that fired within the window. "
           "Unlike `refracted` this forgets, so it cannot accumulate an "
           "unbounded barrier -- but it also penalises the CURRENT word's own "
           "assembly during its 6 training rounds.", "",
           "| n | period/strength | V learned | exh/5 | V* | final w/n | "
           "churn | tail recruit | tail null | pairwise | ident |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(g):
        n, beta, mode, rs, lp, lsr, ss = k
        if beta != 0.05 or mode != "ff" or ss or rs:
            continue
        if lp == 0 and not (n and True):
            pass
        if lp == 0:
            label = "0/0.0 (baseline)"
        else:
            label = f"{lp}/{lsr}"
        runs = g[k]
        V, cps = cps_of(runs)
        pw_s = ms([c["pairwise_overlap_mean"] for c in cps]) if V else "n/a"
        id_s = ms([c["identification_acc"] for c in cps]) if V else "n/a"
        out.append(
            f"| {n} | {label} | {ms([r['vocab_learned'] for r in runs], '{:.0f}')}"
            f" | {n_exhausted(runs)} "
            f"| {ms([recruit_horizon(r['per_word']) for r in runs], '{:.0f}')} "
            f"| {ms([final_wn(r) for r in runs])} "
            f"| {ms([churn(r) for r in runs], '{:.2f}')} "
            f"| {ms([tail_recruit(r) for r in runs], '{:.2f}')} "
            f"| {ms([tail_null(r) for r in runs], '{:.2f}')} "
            f"| {pw_s} | {id_s} |")
    return "\n".join(out)


def sec_beta(gb) -> str:
    if not gb:
        return ""
    out = ["### Table 7 -- Does the mechanism survive a change of beta?", "",
           "Second pass: the same refracted strength at beta = 0.01 and 0.2, "
           "with matched rs=0 controls.", "",
           "| beta | rs | V* n=1000 | V* n=3000 | V* n=10000 | exponent | "
           "exp (per-seed) | V learned 1k/3k/10k | exp(V learned) | exh/15 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    combos = sorted({(k[1], k[3]) for k in gb})
    for beta, rs in combos:
        by_n, vl, cens = {}, {}, 0
        ok = True
        for n in (1000, 3000, 10000):
            k = (n, beta, "ff", rs, 0, 0.0, False)
            if k not in gb:
                ok = False
                break
            by_n[n] = [recruit_horizon(r["per_word"]) for r in gb[k]]
            vl[n] = [r["vocab_learned"] for r in gb[k]]
            cens += n_exhausted(gb[k])
        if not ok:
            continue
        out.append(
            f"| {beta} | {rs} | {ms(by_n[1000], '{:.0f}')} | "
            f"{ms(by_n[3000], '{:.0f}')} | {ms(by_n[10000], '{:.0f}')} | "
            f"{exponent(by_n):.2f} | {ms(exponent_per_seed(by_n), '{:.2f}')} | "
            f"{ms(vl[1000], '{:.0f}')} / {ms(vl[3000], '{:.0f}')} / "
            f"{ms(vl[10000], '{:.0f}')} | {exponent(vl):.2f} | {cens} |")
    return "\n".join(out)


def sec_beta_cost(gb) -> str:
    if not gb:
        return ""
    out = ["", "Consequence metrics for the beta pass, at the last common "
           "checkpoint of each cell:", "",
           "| n | beta | rs | V meas | probes ok | final w/n | pairwise | "
           "ident | retr early | retr late |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(gb):
        n, beta, mode, rs, lp, lsr, ss = k
        runs = gb[k]
        V, cps = cps_of(runs)
        if V is None:
            out.append(f"| {n} | {beta} | {rs} | (no common checkpoint: "
                       f"every seed hit the pool wall at a different V) | "
                       f"{ms([final_wn(r) for r in runs])} | | | | |")
            continue
        out.append(
            f"| {n} | {beta} | {rs} | {V} | "
            f"{ms([c['probes_ok'] for c in cps], '{:.0f}')}/{V} | "
            f"{ms([final_wn(r) for r in runs])} "
            f"| {ms([c['pairwise_overlap_mean'] for c in cps])} "
            f"| {ms([c['identification_acc'] for c in cps])} "
            f"| {ms([c['retrieval_overlap_first_decile'] for c in cps])} "
            f"| {ms([c['retrieval_overlap_last_decile'] for c in cps])} |")
    return "\n".join(out)


def sec_recruit_curve(g) -> str:
    out = ["### Table 8 -- Recruit fraction against the `1 - w/n` null, in "
           "eighths of the vocabulary", "",
           "Only cells that trained all 400 words are shown (an exhausted run "
           "has no late eighths). beta=0.05, TopK, feedforward.", "",
           "| n | condition | source | 1-50 | 51-100 | 101-150 | 151-200 | "
           "201-250 | 251-300 | 301-350 | 351-400 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(g):
        n, beta, mode, rs, lp, lsr, ss = k
        if beta != 0.05 or mode != "ff":
            continue
        runs = [r for r in g[k] if len(r["per_word"]) == 400]
        if len(runs) < 3:
            continue
        cond = (f"rs={rs}" if rs else ("synaptic_scaling" if ss else
                (f"LRI {lp}/{lsr}" if lp else "baseline")))
        for label, key in (("observed", "recruit_frac"),
                           ("null", "recruit_frac_null")):
            bins = np.mean([[float(np.mean(b)) for b in np.array_split(
                np.array([d[key] for d in r["per_word"]]), 8)]
                for r in runs], axis=0)
            out.append(f"| {n} | {cond} | {label} | "
                       + " | ".join(f"{v:.2f}" for v in bins) + " |")
    return "\n".join(out)


def main() -> int:
    # The sweep is sharded across processes (contended host); merge whatever
    # shard files exist, plus a monolithic file if one was produced.
    runs: List[Dict] = []
    files = sorted(HERE.glob("results_recruitment*.json"))
    for f in files:
        runs.extend(json.loads(f.read_text()))
    print(f"merged {len(runs)} runs from {[f.name for f in files]}")
    g = group(runs)
    gb: Dict[tuple, List[Dict]] = {}
    for f in sorted(HERE.glob("results_beta_pass*.json")):
        for r in json.loads(f.read_text()):
            gb.setdefault(cell_key(r), []).append(r)

    prose_path = HERE / "PROSE.md"
    prose = (prose_path.read_text(encoding="utf-8")
             if prose_path.exists() else "")

    parts = [prose.rstrip(), "", "---", "", "## Tables", "",
             sec_refracted_capacity(g), "", sec_exponent(g), "",
             sec_consequences(g), "", sec_nobias(g), "", sec_scaling(g), "",
             sec_lri(g), "", sec_recruit_curve(g), ""]
    if gb:
        parts += [sec_beta(gb), sec_beta_cost(gb), ""]
    text = "\n".join(parts)
    (HERE / "REPORT.md").write_text(text, encoding="utf-8")
    print(f"wrote {HERE / 'REPORT.md'} ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

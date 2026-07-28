"""Do the headline claims survive content-addressed initialisation?

WHY THIS EXISTS
---------------
Addressing synapses by (row, col) instead of by draw order moves every seeded
weight in the repository. It does not move their DISTRIBUTION -- each synapse is
still Bernoulli(p) -- but every number derived from one seed is now a different
number. Any result pinned as a point estimate is therefore unverified until it
is re-derived, and the honest form of the re-derivation is a distribution, not a
replacement point estimate.

That distinction already cost something. When a conformance test went red after
the switch it looked like a regression; measured over twelve seeds, both
disciplines passed seven, and what the failure had actually exposed was a test
that had been passing on a lucky seed since it was written. A single re-run
under the new discipline would have replaced one lucky number with another.

SO THE ACCEPTANCE TEST IS NOT "the number is the same"
-------------------------------------------------------
It is: the CLAIM survives, with its uncertainty stated. Each claim below is
reduced to a scalar with a direction, run over matched seeds under both
disciplines, and reported as mean +/- 95% CI. A claim passes when its interval
excludes the null under BOTH disciplines. A claim whose interval straddles the
null under both was never established, and saying so is the point.

Paired where the design allows it: the same seed under two disciplines is not
the same brain, so these are unpaired between disciplines, but within a
discipline the bin contrasts are per-seed differences and keep that structure.

RESULT (2026-07-28), seeds 42/7/123/2024/5
-------------------------------------------
    claim                        CONTENT (current)        STREAM (legacy)   verdict
    vp_shares_subject           +0.2492 +/- 0.0449      +0.2307 +/- 0.0360  SURVIVES
    vp_shares_verb              +0.1213 +/- 0.0152      +0.1145 +/- 0.0148  SURVIVES
    vp_shares_subject_distant   +0.1696 +/- 0.0514      +0.1626 +/- 0.0427  SURVIVES
    unseen_gradient             +0.2298 +/- 0.0264      +0.2436 +/- 0.0149  SURVIVES
    unseen_minus_seen           +0.0208 +/- 0.0068      +0.0168 +/- 0.0171  report

The STREAM column reproduces the numbers pinned in constituent_structure.py and
held_out_recombination.py (+0.231, +0.114, +0.163; unseen 0.250 vs seen 0.224).
That agreement is what licenses reading the CONTENT column at all -- a harness
that could not reproduce the old result under the old discipline would tell us
nothing about the new one.

Every structural claim survives, and the two columns overlap everywhere. So the
initialisation change moved the numbers without moving the science, which is
what "moves every seeded weight but not their distribution" is supposed to mean
and is now measured rather than asserted.

`unseen_minus_seen` is worth reading carefully. It is POSITIVE, and under the
current discipline its interval excludes zero (+0.0208 +/- 0.0068): never-seen
(subject, verb) combinations get a slightly LARGER compositional gradient than
trained ones. The pre-registered prediction was only that unseen would not be
worse. It is not worse; if anything training a pair specifically adds nothing
and costs a little, which is what you would expect if the constituent's code is
genuinely a function of its parts rather than a memorised whole.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from typing import Callable, Dict, List, NamedTuple, Sequence

# t(0.975) by degrees of freedom; index 0 unused. Small-sample CIs with a
# normal quantile would be too narrow to be honest at n=5.
_T95 = [0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
        2.228, 2.201, 2.179, 2.160, 2.145, 2.131]


class Interval(NamedTuple):
    mean: float
    half_width: float
    n: int

    def excludes_zero(self) -> bool:
        return abs(self.mean) > self.half_width

    def __str__(self) -> str:
        return f"{self.mean:+.4f} +/- {self.half_width:.4f} (n={self.n})"


def ci(values: Sequence[float]) -> Interval:
    n = len(values)
    if n == 0:
        return Interval(float("nan"), float("nan"), 0)
    mean = sum(values) / n
    if n == 1:
        return Interval(mean, float("inf"), 1)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    t = _T95[min(n - 1, len(_T95) - 1)]
    return Interval(mean, t * math.sqrt(var / n), n)


# --- the claims -----------------------------------------------------------
#
# Each returns one scalar per seed whose SIGN is the claim. Keep them small and
# keep the direction in the name: a claim that needs a paragraph to say which
# way it should point is not ready to be re-baselined.

def claims_vp_structure(seed: int) -> Dict[str, float]:
    """VP composition: does a constituent's code track its parts?

    Three scalars from ONE training run, because training dominates the cost
    and re-training per scalar would make a five-seed sweep pointlessly slow:

      shares_subject  overlap(share subject) - overlap(share nothing) > 0
      shares_verb     overlap(share verb)    - overlap(share nothing) > 0
      shares_subject_distant
                      the same contrast on training-DISTANT pairs only. VPs
                      formed close together share connectome state, so the
                      first two can be produced by training proximity alone;
                      this is the contrast that cannot be.
    """
    import statistics

    from constituent_structure import analyse
    from lesion_aphasia import train_parser

    res = analyse(train_parser(seed, n=1000, k=50))

    def m(bins, name):
        vals = bins.get(name) or []
        return statistics.mean(vals) if vals else float("nan")

    nothing = m(res.by_bin, "share nothing")
    d_nothing = m(res.distant, "share nothing")
    return {
        "vp_shares_subject": m(res.by_bin, "share subject") - nothing,
        "vp_shares_verb": m(res.by_bin, "share verb") - nothing,
        "vp_shares_subject_distant": m(res.distant, "share subject") - d_nothing,
    }


def claims_productivity(seed: int) -> Dict[str, float]:
    """Do NEVER-SEEN combinations get the same compositional profile as seen ones?

      unseen_gradient   shared-parent minus share-nothing overlap for held-out
                        (subject, verb) pairs. Composition => > 0.
      unseen_minus_seen the same gradient for unseen MINUS for trained
                        controls. Productivity is the claim that this is NOT
                        negative: if a constituent's code is a function of its
                        parts, having seen the pair should add little.

    The second is the one worth stating carefully. Its null is "unseen is worse
    than seen", so a value near zero SUPPORTS productivity -- which means an
    interval straddling zero is not a failure here, and the harness's blanket
    excludes-zero rule would read it backwards. Reported, not scored.
    """
    import statistics

    from held_out_recombination import (
        HELD_OUT, build_filtered_corpus, form_vp, profile,
    )
    from neural_assemblies.assembly_calculus.emergent.parser import EmergentParser

    held = set(HELD_OUT)
    corpus, _ = build_filtered_corpus(held)
    parser = EmergentParser(n=1000, k=50, p=0.05, beta=0.1, seed=seed,
                            rounds=10)
    parser.train(corpus)
    trained = {k: v for k, v in parser.vp_assemblies.items()
               if k.count("_") == 1}

    subjects = {s for s, _ in held}
    verbs = {v for _, v in held}
    controls = [tuple(k.split("_", 1)) for k in trained
                if k.split("_", 1)[0] in subjects
                and k.split("_", 1)[1] in verbs]

    def gradient(pairs) -> float:
        per_pair = []
        for subj, verb in pairs:
            if subj not in parser.stim_map or verb not in parser.stim_map:
                continue
            per_pair.append(profile(form_vp(parser, subj, verb), trained,
                                    subj, verb))
        if not per_pair:
            return float("nan")
        agg = {b: statistics.mean(p[b] for p in per_pair if p[b] == p[b])
               for b in ("subject", "verb", "nothing")}
        return statistics.mean([agg["subject"], agg["verb"]]) - agg["nothing"]

    unseen = gradient(sorted(held))
    seen = gradient(sorted(controls))
    return {"unseen_gradient": unseen, "unseen_minus_seen": unseen - seen}


def claims_lesion_dissociation(seed: int) -> Dict[str, float]:
    """The double dissociation: two routes that fail on DIFFERENT sentences.

    A single dissociation proves little -- one lesion being worse than another
    can just mean it removed more. The claim is that the two lesions hurt
    OPPOSITE item types, so each scalar is a difference of differences on ONE
    brain, which is what makes them comparable at all:

      lex_hurts_irreversible  how much more the LEXICAL lesion costs on
                              irreversible items than on reversible ones
      pos_hurts_reversible    the mirror for the POSITIONAL lesion

    Both positive is the dissociation. Mind the direction -- I named these
    backwards first and got -1.0 on both, which is the right arithmetic on an
    inverted claim. IRREVERSIBLE items ("lion chases deer") are solvable from
    world knowledge, so the LEXICAL route carries them; REVERSIBLE items ("boy
    chases girl") have no semantic cue, so only the POSITIONAL route can. Raw
    arms at seed 42, which is the clearest statement of the result:

        intact             irrev=1.000  rev=1.000
        lesion LEXICAL     irrev=0.000  rev=1.000
        lesion POSITIONAL  irrev=1.000  rev=0.000

    Both must be measured on the SAME brain per seed; unpairing them would let
    between-brain variance swamp the contrast.
    """
    from lesion_aphasia import (
        clear_role_lexicons, lesion_positional, score, train_parser,
    )
    import copy as _copy

    base = train_parser(seed)

    def arm(fn):
        q = _copy.deepcopy(base)
        if fn is not None:
            fn(q)
        return score(q, "irreversible"), score(q, "reversible")

    irr0, rev0 = arm(None)
    irr_lex, rev_lex = arm(clear_role_lexicons)
    irr_pos, rev_pos = arm(lesion_positional)
    return {
        "lex_hurts_irreversible": (irr0 - irr_lex) - (rev0 - rev_lex),
        "pos_hurts_reversible": (rev0 - rev_pos) - (irr0 - irr_pos),
        "intact_reversible": rev0,
    }


def claims_typology(seed: int) -> Dict[str, float]:
    """Can the true word order be recovered, across the six orders?

    Reduced to ONE scalar per seed -- the fraction of the six orders correctly
    induced, minus the 1/6 a constant answer would score. A method that always
    says SVO gets 0.0 here by construction, which is the comparison that makes
    the number mean anything.
    """
    from word_order_typology import trial

    # trial returns (transitive_ok, intransitive_ok), NOT an order name.
    # Unpacking matters: `trial(...) in (o, True)` compares a TUPLE against a
    # string and a bool, is always False, and would have scored a working model
    # at exactly -1/6 on every seed while looking like a real measurement.
    orders = ("SVO", "SOV", "VSO", "VOS", "OSV", "OVS")
    correct = sum(1 for o in orders if trial(o, 1.0, 300, seed)[0])
    return {"typology_above_constant": correct / len(orders) - 1.0 / len(orders)}


#: name -> group. Grouping means one expensive training run feeds every scalar
#: derived from it, which is what makes a multi-seed sweep affordable.
GROUPS: Dict[str, Callable[[int], Dict[str, float]]] = {
    "vp_structure": claims_vp_structure,
    "productivity": claims_productivity,
    "lesion": claims_lesion_dissociation,
    "typology": claims_typology,
}
CLAIMS = {
    "vp_shares_subject": "vp_structure",
    "vp_shares_verb": "vp_structure",
    "vp_shares_subject_distant": "vp_structure",
    "unseen_gradient": "productivity",
    "unseen_minus_seen": "productivity",
    "lex_hurts_irreversible": "lesion",
    "pos_hurts_reversible": "lesion",
    "intact_reversible": "lesion",
    "typology_above_constant": "typology",
}
#: Claims whose null is NOT "effect is zero", so the blanket excludes-zero rule
#: would score them backwards. `unseen_minus_seen` supports productivity when it
#: is near zero; `intact_reversible` is a level, reported so the lesion
#: contrasts can be read against the headroom they had to work with.
REPORT_ONLY = {"unseen_minus_seen", "intact_reversible"}


def run(seeds: Sequence[int] = (42, 7, 123, 2024, 5),
        claims: Sequence[str] = ()) -> Dict[str, Interval]:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
    os.environ.setdefault("TRAIN_PROGRESS", "0")

    wanted = [c for c in (claims or CLAIMS) if c in CLAIMS]
    groups = sorted({CLAIMS[c] for c in wanted})

    collected: Dict[str, List[float]] = {c: [] for c in wanted}
    for group in groups:
        for seed in seeds:
            try:
                scalars = GROUPS[group](seed)
            except Exception as exc:                      # noqa: BLE001
                print(f"    {group} seed={seed} FAILED: {exc!r}",
                      file=sys.stderr)
                continue
            for name in wanted:
                if CLAIMS[name] == group and name in scalars:
                    value = scalars[name]
                    # A NaN means a bin was empty for this seed, which is a
                    # missing observation and not a zero effect. Dropping it
                    # keeps n honest; silently coercing it to 0 would drag
                    # every interval toward the null.
                    if value == value:
                        collected[name].append(float(value))
    return {name: ci(vals) for name, vals in collected.items()}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    seeds = (42, 7, 123, 2024, 5)

    if os.environ.get("REBASELINE_CHILD"):
        results = run(seeds)
        for name, interval in results.items():
            print(f"RESULT\t{name}\t{interval.mean}\t{interval.half_width}"
                  f"\t{interval.n}")
        return

    # Both disciplines, one subprocess each: the switch is read once at engine
    # construction, so they cannot share a process.
    print(f"\n  seeds {list(seeds)}\n")
    print(f"  {'claim':<26}{'CONTENT (current)':>26}{'STREAM (legacy)':>26}"
          f"{'verdict':>16}")
    both: Dict[str, Dict[str, Interval]] = {}
    for label, env in (("content", {}), ("stream", {"ASSEMBLIES_STREAM_INIT": "1"})):
        child = dict(os.environ, REBASELINE_CHILD="1", **env)
        proc = subprocess.run([sys.executable, os.path.abspath(__file__)],
                              capture_output=True, text=True, env=child)
        parsed = {}
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT\t"):
                _, name, m, h, n = line.split("\t")
                parsed[name] = Interval(float(m), float(h), int(n))
        if not parsed:
            print(f"  [{label}] produced nothing:\n{proc.stdout[-2000:]}"
                  f"\n{proc.stderr[-2000:]}")
        both[label] = parsed

    for name in CLAIMS:
        c, s = both["content"].get(name), both["stream"].get(name)
        if c is None or s is None:
            print(f"  {name:<26}{'MISSING':>26}{'MISSING':>26}{'--':>16}")
            continue
        if name in REPORT_ONLY:
            verdict = "report"
        else:
            verdict = ("SURVIVES" if c.excludes_zero() and s.excludes_zero()
                       else "NOT ESTABLISHED" if not c.excludes_zero()
                       and not s.excludes_zero() else "DISAGREE")
        print(f"  {name:<26}{str(c):>26}{str(s):>26}{verdict:>16}")

    print("\n  SURVIVES        = interval excludes 0 under both disciplines.")
    print("  NOT ESTABLISHED = straddles 0 under both; the claim was never")
    print("                    supported, and the old point estimate hid that.")
    print("  DISAGREE        = the init discipline decides it. Investigate;")
    print("                    do not pick the discipline that agrees with you.")


if __name__ == "__main__":
    main()

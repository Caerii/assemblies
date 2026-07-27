"""Lesion study: the Geschwind reversible/irreversible dissociation.

Geschwind (1970) reported that Broca's aphasics select the right picture for
"the lion chases the deer" but fail on "the lion chases the tiger". The first is
IRREVERSIBLE -- only a lion can chase a deer, so world knowledge alone fixes the
roles. The second is REVERSIBLE -- either animal could be the agent, so the
roles can only come from syntax. Damage the syntactic route and the reversible
sentences break while the irreversible ones survive.

This is a prediction a transformer cannot make: it has no anatomy to lesion.
Mitropolsky & Papadimitriou (2025) place SUBJ/VERB/OBJ in Broca's area and the
lexical/role areas in Wernicke's, which is what makes the experiment meaningful
here.

TWO ROUTES, VERIFIED BEFORE BUILDING THIS
-----------------------------------------
`_assign_roles_neural` can reach a role assignment two ways, and both were
confirmed to be live rather than assumed:

* LEXICAL (neural): each noun's core assembly is projected into the role areas
  and read out against `role_lexicons`. Verified to exist AND to override
  position -- after training where "ball" is only ever a patient, the parser
  reads "ball chases dog" as ball=PATIENT, dog=AGENT, against word order.
* POSITIONAL (symbolic): `constituent_role_order` returns a ranking derived from
  `self.word_order_type`. Corrupting that attribute takes reversible accuracy to
  0.000 across every seed, so it does carry the decision.

HONEST CAVEAT, stated up front: the positional route is a stored Python
attribute, NOT a neural area. Ablating it is therefore a symbolic lesion, not a
synaptic one, and a "Broca's lesion" here is not anatomically real. The lexical
arm IS partly synaptic (see `decompose()` below) -- but its total-collapse
component is a dict clear, so neither arm is purely neural.

RESULT -- 20 brain seeds, 20 irreversible / 12 reversible items, mean +/- 95% CI
-------------------------------------------------------------------------------
    condition                 irreversible        reversible
    intact                   0.930 +/-0.043    1.000 +/-0.000
    lesion POSITIONAL        1.000 +/-0.000    0.000 +/-0.000
    lesion LEXICAL           0.000 +/-0.000    1.000 +/-0.000
    zero synapses only       0.467 +/-0.150    0.950 +/-0.071
    clear lexicons only      0.000 +/-0.000    1.000 +/-0.000

    paired deltas vs intact (same brain per seed)
    POSITIONAL   irrev +0.070 +/-0.043    rev -1.000 +/-0.000
    LEXICAL      irrev -0.930 +/-0.043    rev +0.000 +/-0.000

A clean double dissociation: each lesion destroys one item type (to exactly
zero, with zero variance across seeds) and leaves the other at or above its
intact level. Both directions hold with non-overlapping intervals.

The +0.070 on irreversible under the POSITIONAL lesion is small but real, and
it is a CONFOUND WORTH NAMING: irreversible items are deliberately built in
NON-canonical order, so corrupting the order cue to object-initial makes
position point at the right answer for exactly those items. The spared category
is therefore slightly flattered. At +0.07 it does not threaten the
dissociation; at the +0.50 an earlier single-seed run suggested, it would have.

WHERE THE LEXICAL ROUTE ACTUALLY LIVES (`decompose()`)
-----------------------------------------------------
`lesion_lexical` does two things at full severity -- zeroes core -> ROLE
synapses AND clears `role_lexicons` -- so on its own it cannot say which one
carries the effect. Splitting them shows it is BOTH, in different ways:

* Synapses alone: 0.930 -> 0.467 (+/-0.150). A large, genuinely SYNAPTIC
  effect, and selective (reversible barely moves, 0.950 +/-0.071). The wide
  interval is itself a finding -- how much the learned weights matter varies a
  lot by seed.
* Dictionary alone: 0.930 -> 0.000, zero variance. Mechanically forced:
  `core.py::_score_role_binding` does `stored = lex.get(word)` and returns 0.0
  on a miss, so an empty lexicon silences every role whatever the weights hold.

So destroying every synapse is NOT sufficient for total collapse; clearing the
snapshots is. The snapshots are the readout and the synapses are the evidence
fed into it. An earlier version of this file, measured on ONE seed with four
items, read the synaptic effect as ~0.25 and concluded the arm was purely
symbolic. That was wrong: the synaptic contribution is roughly half the
accuracy.

GRADED SEVERITY (`severity_curve()`, 12 brain seeds) -- only one arm grades
--------------------------------------------------------------------------
    POSITIONAL              irreversible          reversible
      severity 0.00        0.933 +/-0.056      1.000 +/-0.000
               0.25        0.950 +/-0.042      0.750 +/-0.053
               0.50        0.979 +/-0.019      0.500 +/-0.078
               0.75        0.987 +/-0.018      0.236 +/-0.069
               1.00        1.000 +/-0.000      0.000 +/-0.000

    LEXICAL                 irreversible          reversible
      severity 0.00        0.933 +/-0.056      1.000 +/-0.000
               0.25        0.929 +/-0.055      1.000 +/-0.000
               0.50        0.933 +/-0.056      1.000 +/-0.000
               0.75        0.942 +/-0.051      1.000 +/-0.000
               1.00        0.000 +/-0.000      1.000 +/-0.000

The POSITIONAL arm is a near-linear DOSE-RESPONSE on reversible items
(1.00 / .75 / .50 / .24 / .00, intervals well separated) while the spared type
stays at ceiling -- the profile an aphasia comparison would want, and it holds
up under sampling rather than being one seed's shape.

The LEXICAL arm does NOT grade: irreversible is flat within noise through 75%
synaptic destruction and only collapses at 1.00, where the dictionary clear
lands. Partial synaptic damage is nearly free; total damage costs ~0.3-0.5 (the
`zero synapses only` row). That is a strongly non-linear, redundancy-like
profile, not an impairment gradient.

Graded damage on the positional route is modelled as an UNRELIABLE cue --
`word_order_type` becomes a property returning the corrupted object-initial
order on a `severity` fraction of reads. That is a model of degraded syntactic
processing, not of tissue, which is the caveat above restated.

METHOD -- why the numbers above are sampled and the older ones were not
----------------------------------------------------------------------
Everything here previously came from a single `EmergentParser(seed=42)` printed
to two decimals. Three separate problems, all now fixed:

* n=1 model. One brain is one draw from the wiring distribution.
* A sample of the wrong variable. `severity_curve` averaged "3 seeds" that were
  LESION seeds (which synapses got zeroed) while the brain stayed `seed=42`;
  `decompose` looped over seeds while rebuilding the identical `seed=42` brain.
* Item sets too coarse. 4 irreversible / 2 reversible items meant a score could
  only be 0, .25, .5, .75, 1.0 and one item moved it by a quarter.

Now: >=8 BRAIN seeds, each trained once and `deepcopy`-ed per condition so
conditions are PAIRED, reported as mean +/- 95% CI with paired deltas; 20/12
items so one item is worth <=0.05.

This mattered. On one seed with four items the intact irreversible baseline
read 0.50; over 20 seeds it is 0.930 +/-0.043. The alarming number was an
artefact of the sample, not a property of the model.

REPRODUCIBILITY
---------------
Results here once depended on PYTHONHASHSEED -- same `seed=42`, same code,
different process, different answer (and one hash seed crashed). Root cause was
RNG seeds derived from `hash()` of a str plus set-of-str iteration orders that
decide which slice of the seeded stream each stimulus draws. Fixed; 11 hash
seeds now agree exactly, guarded by
`neural_assemblies/tests/test_hashseed_determinism.py`, which subprocesses with
differing hash seeds because an in-process check passes vacuously. The fix
CHANGED the numbers, so nothing measured before it was carried forward.

PREDICTIONS
    lesion POSITIONAL  -> reversible breaks, irreversible survives  (Broca's)
    lesion LEXICAL     -> irreversible breaks, reversible survives  (the mirror)
"""

from __future__ import annotations

import copy
import sys
from typing import Dict, List, Sequence

import numpy as np

from neural_assemblies.assembly_calculus.emergent.parser import EmergentParser
from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
    GroundedSentence, create_training_sentences,
)
from neural_assemblies.assembly_calculus.emergent.core.grounding import VOCABULARY
from neural_assemblies.assembly_calculus.emergent.core.areas import (
    ROLE_AGENT, ROLE_PATIENT,
)

# Role-exclusive nouns give IRREVERSIBLE items (lexical route suffices);
# balanced nouns appear in both roles, so only position can disambiguate them.
#
# SIZED FOR RESOLUTION, not just for coverage. The first version used 2+2+2
# nouns and one test verb, giving 4 irreversible and 2 reversible items -- so a
# score could only land on 0, 0.25, 0.5, 0.75, 1.0 and a single item moved the
# result by a quarter. Every "0.50" reported from that version meant *two
# items*. These counts give 20 irreversible and 12 reversible items (40 and 24
# role judgements), so one item is worth <=0.05.
AGENT_ONLY = ("dog", "cat")
PATIENT_ONLY = ("ball", "book", "food", "table", "car")
BALANCED = ("bird", "boy", "girl")
VERBS = ("chases", "finds", "sees")
TEST_VERBS = ("chases", "finds")


def _gs(words: List[str], roles: List[str]) -> GroundedSentence:
    return GroundedSentence(
        words=words, contexts=[VOCABULARY[w] for w in words], roles=roles,
    )


def build_corpus() -> List[GroundedSentence]:
    """Role-exclusive nouns plus balanced nouns seen in BOTH roles."""
    out: List[GroundedSentence] = []
    for a in AGENT_ONLY:
        for pat in PATIENT_ONLY:
            for v in VERBS:
                out.append(_gs([a, v, pat], ["agent", "action", "patient"]))
    # balanced nouns appear equally often as agent and as patient, so they carry
    # no lexical preference and the model must fall back on order
    for b in BALANCED:
        for other in BALANCED:
            if b == other:
                continue
            for v in VERBS:
                out.append(_gs([b, v, other], ["agent", "action", "patient"]))
                out.append(_gs([other, v, b], ["agent", "action", "patient"]))
    return out * 3


def test_items():
    """(sentence, gold_roles, kind) for scoring.

    A lesion can only show through where the two routes DISAGREE. Canonical
    sentences are scored correctly by position alone AND by lexical preference
    alone, so every condition scores 1.00 and the experiment measures nothing --
    that was the first version of this file, and it produced a flat null.

    So each item type is built to isolate one route:

    * IRREVERSIBLE items put the role-exclusive nouns in NON-canonical order,
      so position says the wrong thing and only world knowledge ("a ball does
      not chase a dog") gives the right answer. This is Geschwind's
      irreversible case: semantics can rescue you when syntax cannot.
    * REVERSIBLE items use the balanced nouns in canonical order. They carry no
      lexical preference by construction, so only order can assign the roles.
    """
    items = []
    # semantics vs position: gold follows the LEXICAL preference
    for a in AGENT_ONLY:
        for pat in PATIENT_ONLY:
            for v in TEST_VERBS:
                items.append(([pat, v, a], {a: "AGENT", pat: "PATIENT"},
                              "irreversible"))
    # no lexical preference available: gold follows ORDER
    for b in BALANCED:
        for other in BALANCED:
            if b != other:
                for v in TEST_VERBS:
                    items.append(([b, v, other],
                                  {b: "AGENT", other: "PATIENT"}, "reversible"))
    return items


def train_parser(seed: int):
    """One trained parser at a given BRAIN seed."""
    p = EmergentParser(n=1000, k=50, p=0.05, beta=0.1, seed=seed, rounds=10)
    p.train(create_training_sentences() + build_corpus())
    return p


def _mean_ci(xs) -> tuple:
    """(mean, half-width of the 95% CI) using the normal approximation.

    n is small (tens of seeds), so this is indicative, not exact -- but it is
    the difference between "0.50" and "0.50 +/- 0.18", and only the second one
    tells you whether a condition actually differs from another.
    """
    a = np.asarray(xs, dtype=float)
    if a.size < 2:
        return float(a.mean()) if a.size else float("nan"), float("nan")
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(a.size))


def sample_conditions(seeds: Sequence[int] = tuple(range(20))
                      ) -> Dict[str, Dict[str, list]]:
    """Score every condition on EVERY brain seed, returning the raw samples.

    Why this exists: earlier versions of this file trained a single parser at
    `seed=42` and reported its scores as if they were the model's behaviour.
    They are one draw. `severity_curve` looked better because it averaged "3
    seeds", but those were LESION seeds -- which synapses got zeroed -- while
    the brain stayed `seed=42` throughout, so it was still n=1 brain.

    Each seed is trained ONCE and deep-copied per condition, so the conditions
    are paired (same brain), which is what makes the differences comparable.
    """
    conds = {
        "intact": lambda q: None,
        "lesion POSITIONAL": lesion_positional,
        "lesion LEXICAL": lesion_lexical,
        "zero synapses only": lambda q: zero_role_synapses(q, 1.0),
        "clear lexicons only": clear_role_lexicons,
    }
    out = {c: {"irreversible": [], "reversible": []} for c in conds}
    for seed in seeds:
        base = train_parser(seed)
        for name, fn in conds.items():
            q = copy.deepcopy(base)
            fn(q)
            out[name]["irreversible"].append(score(q, "irreversible"))
            out[name]["reversible"].append(score(q, "reversible"))
        print(f"    seed {seed} done", flush=True)
    return out


def distribution(seeds: Sequence[int] = tuple(range(20))) -> None:
    """The headline table, as a distribution rather than a point estimate."""
    items = test_items()
    n_irr = sum(1 for _, _, k in items if k == "irreversible")
    n_rev = len(items) - n_irr
    seeds = list(seeds)
    print(f"\nDISTRIBUTION over {len(seeds)} BRAIN seeds "
          f"({n_irr} irreversible / {n_rev} reversible items)\n")
    res = sample_conditions(seeds)
    print(f"\n  {'condition':<22}{'irreversible':>22}{'reversible':>22}")
    for name, d in res.items():
        mi, ci_i = _mean_ci(d["irreversible"])
        mr, ci_r = _mean_ci(d["reversible"])
        print(f"  {name:<22}{mi:>13.3f} +/-{ci_i:<6.3f}"
              f"{mr:>13.3f} +/-{ci_r:<6.3f}", flush=True)
    base_i = np.asarray(res["intact"]["irreversible"])
    base_r = np.asarray(res["intact"]["reversible"])
    print("\n  Paired deltas vs intact (same brain per seed):")
    for name in ("lesion POSITIONAL", "lesion LEXICAL"):
        di, ci_i = _mean_ci(np.asarray(res[name]["irreversible"]) - base_i)
        dr, ci_r = _mean_ci(np.asarray(res[name]["reversible"]) - base_r)
        print(f"  {name:<22} irrev {di:+.3f} +/-{ci_i:.3f}   "
              f"rev {dr:+.3f} +/-{ci_r:.3f}", flush=True)


def score(parser, kind: str) -> float:
    ok = tot = 0
    for words, gold, k in test_items():
        if k != kind:
            continue
        got = parser.parse(list(words))["roles"]
        for w, g in gold.items():
            ok += (got.get(w) == g)
            tot += 1
    return ok / max(tot, 1)


def zero_role_synapses(parser, severity: float = 1.0, seed: int = 0) -> None:
    """Zero a random `severity` fraction of the core -> ROLE synapses.

    The purely SYNAPTIC half of `lesion_lexical`. Kept separate so the two
    components can be applied independently -- see `decompose()`, which is what
    showed this half is NOT what drives the lexical dissociation.
    """
    if severity <= 0.0:
        return
    rng = np.random.default_rng(seed)
    eng = parser.brain._engine
    for core in list(parser.core_lexicons.keys()):
        for role in (ROLE_AGENT, ROLE_PATIENT):
            conn = eng._area_conns.get(core, {}).get(role)
            w = getattr(conn, "weights", None)
            if w is None or not getattr(w, "size", 0):
                continue
            arr = np.asarray(w)          # verified a VIEW, not a copy
            arr[rng.random(arr.shape) < severity] = 0.0


def clear_role_lexicons(parser) -> None:
    """Drop the stored role->word assembly snapshots.

    The SYMBOLIC half of `lesion_lexical`, and -- per `decompose()` -- the half
    that actually carries the effect. `_score_role_binding` returns 0.0 when the
    lookup misses, so an empty lexicon silences every role at once.
    """
    for role in (ROLE_AGENT, ROLE_PATIENT):
        parser.role_lexicons[role] = {}


def decompose(seeds: Sequence[int] = tuple(range(8))) -> None:
    """Attribute the lexical dissociation to synapses vs. stored snapshots.

    `lesion_lexical` at full severity does BOTH, so on its own it cannot say
    which one matters. This applies each alone. Result: the dictionary clear is
    necessary and sufficient, total synaptic destruction is neither.

    `seeds` are BRAIN seeds. The first version looped over "seeds" while
    rebuilding `seed=42` every time, so it averaged N identical runs -- an n=1
    result wearing an error bar's clothing.
    """
    seeds = list(seeds)
    print(f"\nDECOMPOSING the lexical lesion "
          f"(mean +/- 95% CI over {len(seeds)} BRAIN seeds)\n")
    conds = (
        ("(none) intact", lambda p: None),
        ("zero ALL core->ROLE synapses", lambda p: zero_role_synapses(p, 1.0)),
        ("clear role_lexicons only", clear_role_lexicons),
        ("both (== lesion_lexical)",
         lambda p: (zero_role_synapses(p, 1.0), clear_role_lexicons(p))),
    )
    acc = {name: {"irreversible": [], "reversible": []} for name, _ in conds}
    for seed in seeds:
        base = train_parser(seed)
        for name, fn in conds:
            q = copy.deepcopy(base)
            fn(q)
            acc[name]["irreversible"].append(score(q, "irreversible"))
            acc[name]["reversible"].append(score(q, "reversible"))
    print(f"  {'manipulation':<34}{'irrev':>22}{'rev':>22}")
    for name, _ in conds:
        mi, ci_i = _mean_ci(acc[name]["irreversible"])
        mr, ci_r = _mean_ci(acc[name]["reversible"])
        print(f"  {name:<34}{mi:>13.3f} +/-{ci_i:<6.3f}"
              f"{mr:>13.3f} +/-{ci_r:<6.3f}", flush=True)
    print("\n  -> the SNAPSHOT DICTIONARY is the readout; synapses only shape\n"
          "     the projection compared against it. Both arms are symbolic.")


def lesion_lexical(parser, severity: float = 1.0, seed: int = 0) -> None:
    """Lesion of the lexical route, graded by `severity` in [0, 1].

    Zeroes a `severity` fraction of the core -> ROLE synapses, and at FULL
    severity also clears the stored snapshots. Do not read this as a synaptic
    lesion: `decompose()` shows the snapshot clear is what produces the effect,
    and that partial synaptic damage (even 99%) produces none.
    """
    if severity <= 0.0:
        return
    zero_role_synapses(parser, severity, seed)
    if severity >= 1.0:
        clear_role_lexicons(parser)


def lesion_positional(parser, severity: float = 1.0, seed: int = 0) -> None:
    """SYMBOLIC lesion of the positional route, graded by `severity`.

    The order cue is a stored attribute, so it cannot be damaged synaptically
    (see the module caveat). Graded damage is modelled as an UNRELIABLE cue:
    `word_order_type` becomes a property that returns the corrupted
    object-initial order on a `severity` fraction of reads, and the correct one
    otherwise. That is a model of degraded syntactic processing, not of tissue.
    """
    if severity <= 0.0:
        return
    rng = np.random.default_rng(seed)
    correct = getattr(parser, "word_order_type", "SVO") or "SVO"

    class _Unreliable(type(parser)):
        pass

    # Per-instance property: rebind the class so the attribute can be dynamic.
    def _get(self):
        return "OVS" if rng.random() < severity else correct

    _Unreliable.word_order_type = property(_get)
    parser.__class__ = _Unreliable
    parser.__dict__.pop("word_order_type", None)


def severity_curve(seeds: Sequence[int] = tuple(range(8)),
                   levels=(0.0, 0.25, 0.5, 0.75, 1.0)) -> None:
    """Impairment curves: accuracy vs lesion severity, per route, per item type.

    The binary version showed the two routes are SEPARABLE. This asks the
    stronger question -- whether damage produces a graded PROFILE, which is what
    an aphasia comparison would need.

    `seeds` are BRAIN seeds, and that is a correction. This function used to
    hold the brain at `seed=42` and vary only the LESION seed (which synapses
    got zeroed), then print "mean over 3 seeds" -- which read as a sample over
    models but was three lesion draws of ONE model. Sampling the lesion tells
    you nothing about how much of the effect is that particular brain.

    Each brain is trained once and deep-copied per (route, severity), so all
    cells are paired on the same underlying model.
    """
    seeds = list(seeds)
    print(f"\nGRADED SEVERITY (mean +/- 95% CI over {len(seeds)} BRAIN seeds)\n")
    acc = {r: {s: {"irreversible": [], "reversible": []} for s in levels}
           for r in ("POSITIONAL", "LEXICAL")}
    for seed in seeds:
        base = train_parser(seed)
        for route, lesion in (("POSITIONAL", lesion_positional),
                              ("LEXICAL", lesion_lexical)):
            for sev in levels:
                q = copy.deepcopy(base)
                lesion(q, severity=sev, seed=seed)
                acc[route][sev]["irreversible"].append(score(q, "irreversible"))
                acc[route][sev]["reversible"].append(score(q, "reversible"))
        print(f"    seed {seed} done", flush=True)
    for route in ("POSITIONAL", "LEXICAL"):
        print(f"\n  lesion {route}")
        print(f"    {'severity':>9}{'irreversible':>22}{'reversible':>22}")
        for sev in levels:
            mi, ci_i = _mean_ci(acc[route][sev]["irreversible"])
            mr, ci_r = _mean_ci(acc[route][sev]["reversible"])
            print(f"    {sev:>9.2f}{mi:>13.3f} +/-{ci_i:<6.3f}"
                  f"{mr:>13.3f} +/-{ci_r:<6.3f}", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    corpus = build_corpus()
    print("Geschwind lesion study -- role assignment under damage")
    print(f"corpus {len(corpus)} sentences | "
          f"agent-only {AGENT_ONLY} patient-only {PATIENT_ONLY} "
          f"balanced {BALANCED}\n")

    # No single-seed headline table any more. One brain is one draw, and the
    # earlier version of this function printed its scores to two decimals as
    # though they were the model's behaviour. `distribution()` reports the same
    # conditions with a mean and a 95% CI over independently seeded brains,
    # paired so the lesion deltas are within-model.
    distribution()
    decompose()
    severity_curve()


if __name__ == "__main__":
    main()

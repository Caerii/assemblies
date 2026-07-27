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
  `self.word_order_type`. Flipping that attribute drops role accuracy 1.00 ->
  0.33, so it does carry the decision.

HONEST CAVEAT, stated up front: the positional route is a stored Python
attribute, NOT a neural area. Ablating it is therefore a symbolic lesion, not a
synaptic one, and a "Broca's lesion" here is not anatomically real.

BOTH ARMS ARE SYMBOLIC -- a correction, measured by `decompose()` below
--------------------------------------------------------------------
This file previously claimed the LEXICAL arm was "genuinely synaptic" because
`lesion_lexical` zeroes core -> ROLE weights. That claim was WRONG, and
`decompose()` is the experiment that falsified it. `lesion_lexical` does two
things at full severity -- zeroes the synapses AND clears `role_lexicons` -- so
its clean 1.00 -> 0.00 could not be attributed. Separating them:

    manipulation                      irreversible   reversible
    intact                                    1.00         1.00
    zero ALL core -> ROLE synapses            0.75         1.00
    clear role_lexicons only                  0.00         1.00
    both                                      0.00         1.00

Clearing the dictionary is NECESSARY AND SUFFICIENT for the collapse; destroying
every synapse is neither. `core.py::_score_role_binding` shows the mechanism --
it reads `stored = lex.get(word)` and returns 0.0 when that misses, so an empty
`role_lexicons` yields no signal for ANY role no matter what the weights hold.
The synapses only shape the projected assembly that is then compared against the
stored snapshots; the snapshots are the readout.

So the published double dissociation is symbolic on BOTH sides, not one-neural /
one-symbolic. What survives as a genuinely synaptic effect is smaller and in the
predicted direction: zeroing every core -> ROLE weight costs irreversible items
~0.25 while leaving reversible items untouched -- selective, but a long way from
the engineered 1.00 -> 0.00.

GRADED SEVERITY (`severity_curve()`), and which arm actually grades
------------------------------------------------------------------
    POSITIONAL   severity 0 / .25 / .5 / .75 / 1
                 reversible    1.00  0.83  0.33  0.00  0.00
                 irreversible  1.00  1.00  1.00  1.00  1.00
    LEXICAL      irreversible  1.00  0.92  1.00  1.00  0.00

The positional arm gives a genuinely GRADED, monotonic impairment with the
spared type flat at ceiling -- the shape an aphasia comparison would want. The
lexical arm does NOT grade: it is flat until severity 1.0 and then falls off a
cliff, and per the decomposition above that cliff is the dictionary clear, not
accumulated synaptic damage (zeroing 99% of the weights changes nothing). The
0.92 at severity 0.25 is a single item, i.e. noise, not a dose effect.

REPRODUCIBILITY CAVEAT -- READ BEFORE QUOTING ANY NUMBER HERE
-------------------------------------------------------------
Chasing the decomposition above turned up a separate defect: results here
depended on PYTHONHASHSEED. Same `seed=42`, same code, different process ->
irreversible scored 1.00 under hash seeds 0/1/2/9/13 and 0.50 under 6/8/42,
and one hash seed crashed. Within a process it was perfectly stable across
rebuilds, which is exactly why it went unnoticed: the test suite runs in one
process, and `Brain(seed=)` had already been "verified" reproducible there.

Root cause found and fixed for the core engine (see `_sparse.stable_seed` --
`hash()` of a str is per-process randomized, so lazy connectomes were seeded
differently every run), plus three set-iteration sites whose order allocates
neurons. A plain Brain is now hash-seed stable; the EmergentParser layer is
NOT yet -- divergence is isolated to `train_lexicon`.

So the 1.00 entries below carry roughly +/-0.5 of run-to-run uncertainty on
the irreversible column. The 0.00 entries are mechanically forced (an empty
`role_lexicons` makes `_score_role_binding` return 0.0 for every role) and are
robust. Treat the DIRECTION of the dissociation as the result and the exact
magnitudes as provisional until the parser layer is deterministic too.

PREDICTIONS
    lesion POSITIONAL  -> reversible breaks, irreversible survives  (Broca's)
    lesion LEXICAL     -> irreversible breaks, reversible survives  (the mirror)
"""

from __future__ import annotations

import sys
from typing import Dict, List

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
AGENT_ONLY = ("dog", "cat")
PATIENT_ONLY = ("ball", "book")
BALANCED = ("bird", "boy")
VERBS = ("chases", "finds", "sees")


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
            items.append(([pat, "chases", a], {a: "AGENT", pat: "PATIENT"},
                          "irreversible"))
    # no lexical preference available: gold follows ORDER
    for b in BALANCED:
        for other in BALANCED:
            if b != other:
                items.append(([b, "chases", other],
                              {b: "AGENT", other: "PATIENT"}, "reversible"))
    return items


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


def decompose(seeds=(1, 2, 3)) -> None:
    """Attribute the lexical dissociation to synapses vs. stored snapshots.

    `lesion_lexical` at full severity does BOTH, so on its own it cannot say
    which one matters. This applies each alone. Result: the dictionary clear is
    necessary and sufficient, total synaptic destruction is neither.
    """
    print("\nDECOMPOSING the lexical lesion (which component carries it?)\n")
    print(f"  {'manipulation':<34}{'irrev':>7}{'rev':>7}")
    conds = (
        ("(none) intact", lambda p: None),
        ("zero ALL core->ROLE synapses", lambda p: zero_role_synapses(p, 1.0)),
        ("clear role_lexicons only", clear_role_lexicons),
        ("both (== lesion_lexical)",
         lambda p: (zero_role_synapses(p, 1.0), clear_role_lexicons(p))),
    )
    for name, fn in conds:
        irr, rev = [], []
        for sd in seeds:
            p = EmergentParser(n=1000, k=50, p=0.05, beta=0.1, seed=42,
                               rounds=10)
            p.train(create_training_sentences() + build_corpus())
            fn(p)
            irr.append(score(p, "irreversible"))
            rev.append(score(p, "reversible"))
        print(f"  {name:<34}{np.mean(irr):>7.2f}{np.mean(rev):>7.2f}",
              flush=True)
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


def severity_curve(seeds=(1, 2, 3),
                   levels=(0.0, 0.25, 0.5, 0.75, 1.0)) -> None:
    """Impairment curves: accuracy vs lesion severity, per route, per item type.

    The binary version showed the two routes are SEPARABLE. This asks the
    stronger question -- whether damage produces a graded PROFILE, which is what
    an aphasia comparison would need.
    """
    print(f"\nGRADED SEVERITY (mean over {len(seeds)} seeds)\n")
    for route, lesion in (("POSITIONAL (symbolic)", lesion_positional),
                          ("LEXICAL (synaptic)", lesion_lexical)):
        print(f"  lesion {route}")
        print(f"    {'severity':>9} {'irreversible':>13} {'reversible':>12}")
        for sev in levels:
            irr_s, rev_s = [], []
            for sd in seeds:
                p_ = EmergentParser(n=1000, k=50, p=0.05, beta=0.1,
                                    seed=42, rounds=10)
                p_.train(create_training_sentences() + build_corpus())
                lesion(p_, severity=sev, seed=sd)
                irr_s.append(score(p_, "irreversible"))
                rev_s.append(score(p_, "reversible"))
            print(f"    {sev:>9.2f} {np.mean(irr_s):>13.2f} "
                  f"{np.mean(rev_s):>12.2f}", flush=True)
        print()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    corpus = build_corpus()
    print("Geschwind lesion study -- role assignment under damage")
    print(f"corpus {len(corpus)} sentences | "
          f"agent-only {AGENT_ONLY} patient-only {PATIENT_ONLY} "
          f"balanced {BALANCED}\n")

    print(f"{'condition':<22} {'irreversible':>13} {'reversible':>12}")
    print("-" * 50)
    rows: Dict[str, tuple] = {}
    for label, lesion in (("intact", None),
                          ("lesion POSITIONAL", lesion_positional),
                          ("lesion LEXICAL", lesion_lexical)):
        p = EmergentParser(n=1000, k=50, p=0.05, beta=0.1, seed=42, rounds=10)
        p.train(create_training_sentences() + corpus)
        if lesion is not None:
            lesion(p)
        irr, rev = score(p, "irreversible"), score(p, "reversible")
        rows[label] = (irr, rev)
        print(f"{label:<22} {irr:>13.2f} {rev:>12.2f}", flush=True)

    base_irr, base_rev = rows["intact"]
    p_irr, p_rev = rows["lesion POSITIONAL"]
    l_irr, l_rev = rows["lesion LEXICAL"]
    print("\nDissociation check (drop from intact):")
    print(f"  POSITIONAL lesion: irreversible {base_irr - p_irr:+.2f}  "
          f"reversible {base_rev - p_rev:+.2f}   "
          f"(Geschwind predicts reversible drops MORE)")
    print(f"  LEXICAL lesion:    irreversible {base_irr - l_irr:+.2f}  "
          f"reversible {base_rev - l_rev:+.2f}   "
          f"(mirror predicts irreversible drops MORE)")

    decompose()
    severity_curve()


if __name__ == "__main__":
    main()

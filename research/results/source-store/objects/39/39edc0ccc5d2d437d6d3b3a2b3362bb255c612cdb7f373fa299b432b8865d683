"""Which branch does `_role_binding_margin` actually take in a real parse?

THE DEFECT (task #110). `_role_binding_margin` returns TWO DIFFERENT QUANTITIES
under one name:

    if not others:
        return own                                        # RAW OVERLAP
    base = sum(others) / len(others)
    return max(0.0, (own - base) / max(1e-6, 1.0 - base))  # RESIDUAL

The raw overlap has no baseline subtracted, so for the same underlying binding
quality it is systematically LARGER than the residual. The function's own
docstring says the raw form "returns ~0.9 for EVERY candidate role" and "cannot
discriminate" -- that is the reason the residual exists.

WHY IT WOULD MATTER. `_assign_roles_neural` (roles.py:271-284) calls this for
every candidate role area and normalizes the results against each other:

    total = sum(margins.values()) + eps * len(margins)

So the two quantities COMPETE ON THE SAME SCALE. A role area whose lexicon holds
exactly one filler contributes the larger raw number and takes more of the
probability mass than an equally-well-bound role area with several fillers --
making LEXICON SIZE, not binding strength, move the role decision.

THIS SCRIPT ONLY COUNTS. It changes nothing. If the `not others` branch never
fires in a real parse, the fix is free and needs no A/B. If it fires, we know
which parses are affected before touching anything.

Also counted: the `stored is None` branch, which returns a bare 0.0 for "this
word was never trained in this role" -- a ⊥-invention, though a defensible one
here given the additive smoothing downstream.

Run: python role_margin_branch_census.py [seed]
"""
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (  # noqa: E402
    get_parser_cache,
)

def _sentences_from_lexicons(parser, n=8):
    """Build probe sentences from words the parser ACTUALLY learned.

    The first version of this script used hand-written sentences ("the dog
    chases the cat"). None of those words are in the role lexicons -- the
    trained corpus is a different vocabulary entirely -- so 78% of calls took
    the `stored is None` branch and the census measured MY corpus, not the
    parser. Derive the probes from the lexicons and that artefact cannot recur.
    """
    agents = sorted(parser.role_lexicons.get("ROLE_AGENT", {}))
    actions = sorted(parser.role_lexicons.get("ROLE_ACTION", {}))
    patients = sorted(parser.role_lexicons.get("ROLE_PATIENT", {}))
    if not (agents and actions and patients):
        raise SystemExit(
            f"cannot build probes: |AGENT|={len(agents)} |ACTION|={len(actions)} "
            f"|PATIENT|={len(patients)} -- at least one role lexicon is empty")
    return [[agents[i % len(agents)],
             actions[i % len(actions)],
             patients[i % len(patients)]] for i in range(n)]


def main(seed=11):
    parser = get_parser_cache().fork("SENTENCES", seed=seed)

    print("role lexicon sizes: " + "  ".join(
        f"{r}={len(l)}" for r, l in parser.role_lexicons.items()))
    empty = [r for r, l in parser.role_lexicons.items() if not l]
    if empty:
        print(f"EMPTY role lexicons (every margin there is branch A): {empty}")
    SENTENCES = _sentences_from_lexicons(parser)
    print(f"probes built from learned vocabulary: {SENTENCES[:3]} ...\n")

    branches = Counter()
    lex_sizes = Counter()          # len(lex) at each call
    per_role = defaultdict(Counter)
    values = defaultdict(list)     # branch -> returned values

    original = type(parser)._role_binding_margin

    def instrumented(self, word, core_area, role_area):
        lex = self.role_lexicons.get(role_area, {})
        stored = lex.get(word)
        lex_sizes[len(lex)] += 1
        if stored is None:
            branch = "A: no stored binding -> bare 0.0"
        elif len([w for w in lex if w != word]) == 0:
            branch = "B: no other fillers -> RAW OVERLAP"
        else:
            branch = "C: residual (the intended quantity)"
        branches[branch] += 1
        per_role[role_area][branch] += 1
        out = original(self, word, core_area, role_area)
        # `_role_binding_margin` now returns a `Measured`. Record what the
        # CALLER sees -- `.or_else(0.0)` -- so this census stays comparable
        # with the pre-fix numbers it was written to produce.
        values[branch].append(
            out.or_else(0.0) if hasattr(out, "or_else") else out)
        return out

    type(parser)._role_binding_margin = instrumented
    try:
        for words in SENTENCES:
            parser.parse(list(words))
    finally:
        type(parser)._role_binding_margin = original

    total = sum(branches.values())
    print(f"seed={seed}   {len(SENTENCES)} sentences, "
          f"{total} calls to _role_binding_margin\n")

    print(f"{'branch':44s} {'calls':>6s} {'share':>7s} {'mean':>8s} {'max':>8s}")
    for branch, count in branches.most_common():
        vals = values[branch]
        mean = sum(vals) / len(vals) if vals else float("nan")
        print(f"{branch:44s} {count:6d} {count / total:7.1%} "
              f"{mean:8.4f} {max(vals) if vals else float('nan'):8.4f}")

    print("\nrole lexicon size at call time:")
    for size in sorted(lex_sizes):
        print(f"  |lex| = {size:3d}   {lex_sizes[size]:5d} calls")

    print("\nper role area:")
    for role in sorted(per_role):
        counts = per_role[role]
        tag = "".join(b[0] for b in sorted(counts))
        print(f"  {role:22s} {dict(counts)}   [{tag}]")

    raw = branches["B: no other fillers -> RAW OVERLAP"]
    residual = branches["C: residual (the intended quantity)"]
    print("\n" + "=" * 68)
    if raw == 0:
        print("BRANCH B IS DEAD in this corpus: every scored role area held at")
        print("least one other filler, so every margin returned is the residual.")
        print("The two-scale confound does NOT fire here -- the fix is free,")
        print("and should be made anyway so it cannot start firing silently.")
    elif residual == 0:
        print("BRANCH C IS DEAD: every margin is a RAW OVERLAP, which the")
        print("docstring says cannot discriminate (~0.9 for every role). The")
        print("lexical route is then contributing almost nothing to the")
        print("decision, and the structural prior is doing all the work.")
    else:
        print(f"BOTH BRANCHES FIRE: {raw} raw vs {residual} residual.")
        print("These are normalized against each other, so lexicon size moves")
        print("the role decision. Every role assignment mixing the two is")
        print("confounded, and the fix needs a paired A/B (research/harness.py).")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 11)

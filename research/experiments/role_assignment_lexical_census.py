"""Is the LEXICAL half of role assignment doing anything, or is it all prior?

WHERE THIS COMES FROM. `nemo2025_curriculum`'s role-probe metric is
substrate-invariant: beta=0.05, phon_weight=6, and both together give results
BYTE-IDENTICAL to baseline, while the same knobs move direct role retrieval from
0.806/0.733 to 0.974/0.964. A metric that cannot respond to the substrate is not
measuring the substrate.

`_assign_roles_neural` shows how that is possible:

    score   = prior + lexical
    prior   = _STRUCTURAL_PRIOR * (0.5 ** rank)          # position in role_order
    lexical = (margins[ra] + eps) / total  if any(margins.values()) else 0.0

`margins[ra]` is `_role_binding_margin(...).or_else(0.0)`. If every margin comes
back UNDEFINED, `any(margins.values())` is False, `lexical` is 0.0 for every
candidate, and the winner is decided entirely by a positional prior that no
substrate parameter can touch. The parse would then be "neural" in name only.

THIS FILE COUNTS IT RATHER THAN INFERRING IT. Three questions, in order:

  1. HOW OFTEN IS A MARGIN DEFINED? `_role_binding_margin` returns
     `Measured.undefined` for non-residual cases, and four of seven role
     lexicons were empty as recently as #113. An undefined margin is not a
     small margin -- it is no evidence at all.

  2. WHEN DEFINED, IS IT EVER DECISIVE? A lexical term that exists but never
     outranks the prior's gap is a mechanism that runs and changes nothing --
     the dormant-mechanism shape this repo has found twice (mutual inhibition
     with zero co-targeting fibers; `update_beta_by_area` writing to a dict the
     engine does not read). Decisiveness is measured by re-scoring each
     decision with `lexical` forced to 0 and asking whether the WINNER changes.

  3. DOES RAISING THE SUBSTRATE'S QUALITY CHANGE EITHER? The phon_weight=6 /
     beta=0.05 arm is included because it demonstrably improves role retrieval
     elsewhere. If margins stay undefined there too, the readout -- not the
     binding -- is the broken part, which is #121.

WHAT WOULD MAKE THIS FILE'S ANSWER WRONG: wrapping the wrong function. The
margin is read inside `_assign_roles_neural`, so the wrapper goes on the
INSTANCE's bound method and the count is asserted non-zero before anything is
concluded -- a census that captured nothing looks exactly like a mechanism that
never fires. That check killed an earlier experiment of mine that patched two
of three namespaces and silently observed nothing.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.pop("EMERGENT_DEV_CURRICULUM", None)

import json                                                            # noqa: E402

GOLDEN = (Path(__file__).resolve().parents[2] / "research" / "literature"
          / "parity" / "golden" / "nemo2025_curriculum.json")

PROBES = [
    {"words": ["the", "dog", "runs"], "expected_roles": {"dog": "AGENT"}},
    {"words": ["the", "cat", "chases", "the", "bird"],
     "expected_roles": {"cat": "AGENT", "bird": "PATIENT"}},
]

ARMS = [
    ("golden as recorded", {}),
    ("beta=0.05 + phon_weight=6", {"beta": 0.05, "phon_weight": 6.0}),
]


def build(params, overrides):
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        CurriculumTrainer,
    )
    kw = dict(n=params["n"], k=params["k"], p=params["p"],
              beta=params["beta"], seed=42, rounds=params["rounds"])
    kw.update(overrides)
    parser = EmergentParser(**kw)
    trainer = CurriculumTrainer(parser)
    for stage in params["stages"]:
        trainer.train_stage(stage)
    return parser


def census(parser):
    """Wrap the INSTANCE's margin call and record every read."""
    calls = []
    orig = parser._role_binding_margin

    def wrapped(word, core_area, role_area, *a, **kw):
        m = orig(word, core_area, role_area, *a, **kw)
        try:
            defined = m.is_defined
        except AttributeError:
            defined = m is not None
        try:
            val = m.or_else(0.0)
        except AttributeError:
            val = float(m) if m is not None else 0.0
        calls.append({"word": word, "role_area": role_area,
                      "defined": bool(defined), "value": float(val)})
        return m

    parser._role_binding_margin = wrapped
    try:
        for item in PROBES:
            parser.parse(item["words"])
    finally:
        parser._role_binding_margin = orig
    return calls


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    g = json.loads(GOLDEN.read_text(encoding="utf-8"))
    params = g["parameters"]
    print("Is the LEXICAL half of role assignment doing anything?")
    print(f"parameters: n={params['n']} k={params['k']} "
          f"beta={params['beta']} rounds={params['rounds']}")
    print()

    for label, ov in ARMS:
        parser = build(params, ov)
        calls = census(parser)
        print(f"=== {label} ===")
        if not calls:
            print("  ** CAPTURED NOTHING. The wrapper did not intercept the "
                  "margin call, so this arm measured nothing and must NOT be "
                  "read as 'the mechanism never fires'. **")
            print()
            continue
        n = len(calls)
        defined = [c for c in calls if c["defined"]]
        nonzero = [c for c in calls if c["value"] != 0.0]
        print(f"  margin reads: {n}")
        print(f"  DEFINED:      {len(defined)}/{n} "
              f"({len(defined) / n:.0%})")
        print(f"  non-zero:     {len(nonzero)}/{n} "
              f"({len(nonzero) / n:.0%})")
        if nonzero:
            vals = sorted(c["value"] for c in nonzero)
            print(f"  value range:  {vals[0]:.4f} .. {vals[-1]:.4f}")
            for c in nonzero[:8]:
                print(f"     {c['word']:<8} {c['role_area']:<14} "
                      f"{c['value']:.4f}")
        else:
            print("  -> EVERY margin is zero, so `any(margins.values())` is")
            print("     False and the lexical term contributes 0.0 to every")
            print("     candidate. The winner is the STRUCTURAL PRIOR alone,")
            print("     which no substrate parameter can move. That is why the")
            print("     golden's role accuracy is byte-identical across arms.")
        print()

    print("=" * 70)
    print("If the lexical term is uniformly zero, the role-probe golden is a")
    print("test of a positional rule wearing the name `_assign_roles_neural`.")
    print("The fix is #121 (read the role lexicons via recall/bind_strength),")
    print("NOT re-recording the threshold.")


if __name__ == "__main__":
    main()

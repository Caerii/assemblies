"""#91 sequential extension: does MORE EXPOSURE rescue the noun/verb split?

CONTEXT. `task91_emergent_noun_verb.py`'s corrected (unsaturated) probe found
a genuine null at its default budget -- 120 sentences total, ~10/word: E1
wrong sign, E2 exact chance (0.500), E3 non-monotonic. Two independently
built probes (stability, recall) agreed. That rules out "wrong instrument"
for THIS budget, but not for the mechanism overall -- #52/#121/#151 all found
toy-scale substrates blind to effects the production recipe shows cleanly,
and this toy trains at ~10 sentences/word against the paper's own claim
(Fig 3d) that the effect is a co-occurrence-diversity phenomenon.

THIS SCRIPT holds n=10^4, k=50 and the same 8-noun/4-verb vocabulary fixed
(so it is the SAME substrate, not a bigger one) and varies ONLY exposure:
sentences/word at 10x (the original budget, replicated for continuity),
40x and 160x. BOTH arms run at every level, because a control that rises
with exposure would mean the increase is coming from "more training" in
general, not from the architectural asymmetry -- exactly the ablation
`task91_emergent_noun_verb.py` already uses, extended along a new axis.

REGISTERED PREDICTIONS (before any level beyond 10x, which is already known):
  P-RESCUE   E2 accuracy (asymmetric arm) rises with exposure and clears
             the majority baseline (0.667) by 160x. This is the claim that
             would overturn the toy-scale null and license testing the
             mechanism at production scale.
  P-NULL     Accuracy stays flat within noise across all three levels
             (asymmetric arm). This would mean the toy substrate's null is
             about the MECHANISM as implemented here (this architecture,
             this vocabulary size, this k), not merely under-trained --
             closing #91 at this design rather than deferring it again.
  CONTROL    the flat-fiber control's accuracy does NOT trend upward with
             exposure. A rising control at 160x would VOID any P-RESCUE
             reading at that level (the gain would not be attributable to
             the asymmetry).
  DECISION RULE (pre-stated): production-scale testing of this mechanism is
             warranted ONLY if P-RESCUE holds AND the control stays flat.
             Otherwise #91 stays closed on this design and the paper-fidelity
             deviations (n=10^4 not 10^5, no C_i context areas, m=0) become
             the next candidates to investigate, not exposure.

SEEDS: 3 (matching the parent script's convention; numpy_exact's O(M^2) cost
bounds how far this can be pushed -- see [[numpy-exact-95x-slower]]).
"""
from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TRAIN_PROGRESS", "0")

from task91_emergent_noun_verb import NOUNS, VERBS, arm  # noqa: E402

SEEDS = (1, 2, 3)
WORDS = len(NOUNS) + len(VERBS)
MULTIPLIERS = (10, 40, 160)
BASELINE = max(len(NOUNS), len(VERBS)) / WORDS


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"\n  #91 exposure sweep -- does the split appear with more "
          f"training, same n={10000}/k={50}/vocab?")
    print(f"  majority baseline {BASELINE:.3f}, {len(SEEDS)} seeds\n")

    print(f"  {'sentences/word':>15} {'arm':>12} {'noun gap':>10} "
          f"{'verb gap':>10} {'accuracy':>10} {'stability':>10}")
    rows = {}
    for mult in MULTIPLIERS:
        sentences = mult * WORDS
        for label, asym in (("asymmetric", True), ("control", False)):
            r = [arm(asym, s, sentences) for s in SEEDS]
            ng = statistics.mean(x[0] for x in r)
            vg = statistics.mean(x[1] for x in r)
            acc = statistics.mean(x[2] for x in r)
            lvl = statistics.mean(x[3] for x in r)
            rows[(mult, label)] = (ng, vg, acc, lvl)
            print(f"  {mult:>15} {label:>12} {ng:>10.3f} {vg:>10.3f} "
                  f"{acc:>10.3f} {lvl:>10.3f}", flush=True)

    print("\n  READING\n")
    asym_acc = [rows[(m, "asymmetric")][2] for m in MULTIPLIERS]
    ctrl_acc = [rows[(m, "control")][2] for m in MULTIPLIERS]
    rescue = asym_acc[-1] > BASELINE and asym_acc[-1] > asym_acc[0] + 0.05
    ctrl_flat = ctrl_acc[-1] <= ctrl_acc[0] + 0.05
    print(f"    P-RESCUE  accuracy rises and clears baseline by {MULTIPLIERS[-1]}x: "
          f"{str(rescue):>5}   {asym_acc[0]:.3f} -> {asym_acc[-1]:.3f} "
          f"(baseline {BASELINE:.3f})")
    print(f"    CONTROL   flat arm stays flat across levels:      "
          f"{str(ctrl_flat):>5}   {ctrl_acc[0]:.3f} -> {ctrl_acc[-1]:.3f}")

    print()
    if rescue and ctrl_flat:
        print("    P-RESCUE HOLDS, control stays flat: the toy null was an")
        print("    under-training artifact, not a mechanism failure.")
        print("    Production-scale testing is warranted per the decision rule.")
    elif not ctrl_flat:
        print("    THE CONTROL ROSE WITH EXPOSURE. Any asymmetric-arm gain is")
        print("    confounded with 'more training in general' and cannot be")
        print("    attributed to the architectural asymmetry. VOID.")
    else:
        print("    P-NULL: accuracy stays flat within noise across all three")
        print("    exposure levels. #91 closes on THIS design -- the null is")
        print("    about the mechanism as implemented (architecture, vocab")
        print("    size, k), not about under-training. Paper-fidelity")
        print("    deviations (n=10^4, no C_i areas) are the next candidates.")


if __name__ == "__main__":
    main()

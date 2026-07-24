"""Assembly instability metrics — P600 analogue kernels.

THE BRIDGE BEING BUILT.  The P600 is a late positive ERP component elicited by
syntactic anomaly and by structures that force reanalysis.  Two influential
process accounts read it as a COST rather than a detection signal: Vosse &
Kempen (2000) as the time a unification network takes to settle, Brouwer &
Crocker (2017) as the magnitude of the update needed to integrate the incoming
word.  Both are statements about how much the representation has to move.

NEMO gives that quantity directly.  A projection is an iterated map on winner
sets; a well-formed input lands the area near a fixed point immediately and
subsequent rounds barely change the winners, while an ill-formed one leaves it
churning.  Summing how much the winner set changes per round is therefore a
literal implementation of "settling cost", not an analogy fitted after the
fact.  That sum is what this module computes.

WHY JACCARD RATHER THAN ``assembly.overlap``.  ``overlap`` normalises by
``min(|A|, |B|)``, which is right for "is this the assembly I stored?" -- a
subset scores 1.0.  It is wrong here: an area that keeps its old winners and
recruits fifty more has moved a lot, and min-normalisation would report no
change at all.  Jaccard normalises by the union, so both losing and gaining
neurons count against stability.  The two measures disagree exactly on the
cases this metric exists to catch.

Scale note: ``compute_jaccard_instability`` returns a SUM, so it grows with
the number of rounds and is only comparable across runs of equal length.
``mean_jaccard_instability`` divides by the number of transitions and is the
one to use when round counts differ.

Pure functions on winner sets and ``Brain`` projection. No parser state.

  compute_jaccard_instability: sum of (1 - Jaccard) across consecutive rounds.
  mean_jaccard_instability: per-transition average (normalized P600 scale).
  compute_anchored_instability: fresh-stimulus anchored binding probe.

References:
  - Vosse & Kempen 2000: P600 = settling time
  - Brouwer & Crocker 2017: P600 = integration update cost
  - research/results/primitives/RESULTS_composed_erp.md
"""

from __future__ import annotations

from typing import Any, Dict, List, Set


def compute_jaccard_instability(round_winners: List[Set[int]]) -> float:
    """Sum of (1 - Jaccard) across consecutive winner sets."""
    instability = 0.0
    for i in range(1, len(round_winners)):
        prev_set = round_winners[i - 1]
        curr_set = round_winners[i]
        union = prev_set | curr_set
        if len(union) > 0:
            jaccard = len(prev_set & curr_set) / len(union)
        else:
            jaccard = 1.0
        instability += 1.0 - jaccard
    return instability


def mean_jaccard_instability(round_winners: List[Set[int]]) -> float:
    """Per-transition mean instability (0 when fewer than 2 rounds)."""
    if len(round_winners) <= 1:
        return 0.0
    return compute_jaccard_instability(round_winners) / (len(round_winners) - 1)


def compute_anchored_instability(
    brain,
    word: str,
    core_area: str,
    role_area: str,
    n_settling_rounds: int = 10,
    activate_rounds: int = 3,
) -> Dict[str, Any]:
    """Anchored P600 via fresh-stimulus co-projection then area-only settling.

    Phase A: stimulus co-projection creates anchored pattern in role area.
    Phase B: area-to-area settling without stimulus; instability over all rounds.

    WHY "ANCHORED".  Letting the role area settle from whatever it happened to
    be holding conflates two things: how hard THIS word is to integrate, and
    where the area started.  Phase A removes the second by inhibiting both
    areas and driving the role area from the word's own fresh PHON stimulus,
    so every word is measured from a comparable, word-determined starting
    point.  Only then is the stimulus withdrawn and the area allowed to settle
    on its recurrent and cross-area weights alone -- Phase B is the part that
    is actually scored.

    PRECONDITION, NOT ENFORCED.  Plasticity must already be off
    (``brain.disable_plasticity = True``).  With it on, each settling round
    potentiates the winners it just produced, which drives the area to a fixed
    point faster than it otherwise would and systematically UNDERSTATES
    instability -- i.e. the metric quietly moves toward zero for every word,
    compressing exactly the contrast being measured.  This function does not
    check; the caller is responsible.

    Winner sets are read as raw compact indices rather than
    :class:`~..assembly.Assembly` neuron IDs.  That is sound here only because
    the comparison is within one call and compact indices are append-only, so
    an index means the same neuron across the rounds being compared.  Do not
    compare these sets against sets captured in another call.
    """
    brain.inhibit_areas([core_area])
    for _ in range(activate_rounds):
        brain.project({f"PHON_{word}": [core_area]}, {core_area: [core_area]})

    brain.inhibit_areas([role_area])
    brain.project(
        {f"PHON_{word}": [core_area, role_area]},
        {core_area: [role_area]},
    )
    all_round_winners = [set(brain.areas[role_area].winners.tolist())]

    for _ in range(n_settling_rounds - 1):
        brain.project(
            {},
            {core_area: [role_area], role_area: [role_area, core_area]},
        )
        all_round_winners.append(
            set(brain.areas[role_area].winners.tolist()),
        )

    instability = compute_jaccard_instability(all_round_winners)

    return {
        "instability": instability,
        "round_winners": all_round_winners,
    }

"""Ratchet against measuring things the wrong way.

THE FAILURE THIS EXISTS FOR, and it is recent and mine. In one evening I:

  * concluded an engine comparison from a criterion written as `exact > sparse`
    -- no threshold at all, so a single extra success "confirmed" it. It duly
    did: 18/24 against 20/24, a difference of exactly ONE standard error, was
    printed under a verdict claiming the sampler was a large part of the
    problem;
  * drew a conclusion from a single seed, twice, and had to withdraw both;
  * published a "conserved budget" law computed with `statistics.mean` over
    seeds and no interval, then retracted it.

Every one of those was already prevented by something in `diagnostics`:

    ensemble(run, seeds)     RAISES below 3 seeds -- "a single-seed
                             before/after is not a measurement"
    Ensemble.beats(thresh)   judges by the CONFIDENCE BOUND, not the mean
    paired_delta(a, b)       per-seed difference, which is the test an A/B
                             actually asks; independent CIs are weaker and
                             understate the spread by ~sqrt(2)
    compare_arms(strict=1)   refuses arms that produced bit-identical values,
                             i.e. a dead manipulation reported as "no effect"
    verify_probe(hi, lo)     refuses a probe until it has produced BOTH
                             answers on cases where it must

The tooling was not missing. It was BYPASSED, because `statistics.mean` is one
import away and the sanctioned path is not. Vigilance did not fix that and will
not; a ratchet might. This is the same instrument that caught the compact-index
defect the same evening -- the only mechanism that has actually worked here.

WHAT IS CHECKED
---------------
1. Seed statistics computed by hand in `research/`: a file that mentions seeds
   and calls `statistics.mean` / `np.mean` without going through `ensemble`,
   `paired_delta` or `compare_arms`. A mean over seeds with no interval cannot
   support a comparison, and the comparison gets made anyway.
2. `Brain(...)` constructed in `research/` without an explicit `engine=`. The
   default is `numpy_sparse`, whose candidate sampler INVENTS the drive for
   neurons that have not fired. `WordOrderLearner` had no engine parameter at
   all, so every word-order result in this repository is a sampler measurement
   -- discovered only after an evening of work built on top of them.

WHY A RATCHET AND NOT A BAN. The baseline is 129 files and ~1196 hand-rolled
sites; 79 files and 118 unpinned constructions. Most predate the tooling and
some are legitimately fine -- a mean over conditions is not a mean over seeds,
and plenty of scripts are demos rather than measurements. Banning outright
would mean a flag day nobody will take. Freezing the counts stops the pattern
SPREADING and points new code at the sanctioned path, which is the achievable
goal.

The baseline lives in `methodology_baseline.json`, generated rather than
hand-typed. Lower an entry when a file is fixed. RAISING one, or adding a new
file, needs a comment here saying why that site is legitimate -- which is the
whole point of the friction.

RAISED 2026-08-10, for the sequence-organ scripts (`seq_*`). Two of them judged
BARS on a bare mean over seeds and were FIXED rather than baselined:
`seq_a1_fsm_parity.py` and `seq_a2_word_order_fsm.py` now build a
`diagnostics.ensemble` for every overlap statistic and test the CONFIDENCE
BOUND against the threshold, so P-CONJ/P-CONJ2/P-PRE survive seed variation
rather than resting on a point estimate.

The eight files added to the baseline judge NOTHING on their aggregates. Their
decision bars are counts of seeds (x/10 correct), which need no interval, and
every one of them prints the per-seed value on its own line -- the mean is a
reading aid over a distribution already reported in full, which is the
"mean over CONDITIONS rather than over seeds" case this ratchet exempts.
`seq_a1_step_accuracy.py` additionally reports mean +/- sd explicitly.
`task91_exposure_sweep.py` predates this work and is unchanged.

RAISED 2026-08-10 for `seq_state_refraction.py` (2 sites). Both are inside
`measure()` and average over SENTENCE PAIRS within a single seed -- the mean
cross-prefix overlap and the mean same-prefix overlap that together form that
seed's separation and determinism. They ARE the per-seed statistic, and each
one is then handed to `ensemble_from_values` across seeds, which is where the
interval is taken and where every bar is judged. Rewriting them as `ensemble`
would put a confidence interval over sentence pairs inside one brain, which is
a different and wrong claim -- the "mean over CONDITIONS rather than over
seeds" case `_SEED_ADVICE` names.

RAISED 2026-08-10 for `seq_s5_word_problem.py` (1 site). `np.mean` there
averages the solvable GROUPS' accuracies (Z60 and A4xZ5) into the reference
that the non-solvable arms are compared against -- a mean over ARMS, not over
seeds. Each arm's own accuracy is an `ensemble_from_values` over seeds, and
every bar is judged on those intervals.

RAISED 2026-08-10 for `seq_s5_cliff_anatomy.py` (2 sites). `pre_onblock_mean`
averages the live state's on-block overlap over the STEPS of one trajectory,
and `ok_margin_mean` averages readout margins over the CENSUSED PAIRS of one
brain -- both within a single seed, forming that seed's record. Every bar in
that study (C1-C4) is a per-seed exact condition ANDed across seeds, with each
seed's value printed on its own line; no aggregate over seeds is judged at
all.

RAISED (1 -> 6) 2026-08-10 for `seq_s5_word_problem.py`, the Amendment 2
per-step readout. `np.mean(correct[:L])` in `evaluate` averages TRANSITION
correctness over the STEPS of one trajectory -- the per-seed statistic, which
`ensemble_from_values` then aggregates across seeds where every bar is judged.
The solvable/non-solvable summary averages the four ARMS' ensemble means for a
printed gap; no bar reads it, and the amendment's S5a/S5b verdicts were judged
on the per-arm ensembles.

RAISED 2026-09-03 for `alignment_load.py` (2 sites) and `unaligned_scenes.py`
(1 site). `head_tail` averages per-word accuracy over the WORDS of one seed's
head and tail halves -- the per-seed statistic, which `judge` then hands to
`ensemble_from_values` across seeds; the bundles-per-scene figure averages
over SCENES of one corpus and is descriptive. Everything judged in
`alignment_load.judge` -- L1, L2, the L3 paired gap, Z1 -- is an ensemble over
seeds read on its bound. The ratchet's catch on the first draft was real: L3
compared two bare seed means, and was rewritten as a paired per-seed gap.

RAISED 2026-09-03 for `seq_s5_bar_tie.py` (1 site). `np.mean(js)` averages
Jaccard similarities over ORGANS x noisy readouts of one trained brain each;
the verdict is a SET comparison (are the soft pairs identical under noise?)
and its bars are thresholds on that descriptive mean plus exact counts (zero
hard defects, count ratio). No seed-level interval is claimed or judged.
"""

from __future__ import annotations

import ast
import json
import os
import re

from ._source_scan import blank_prose, python_sources

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "methodology_baseline.json")

_SEEDY = re.compile(r"\b(statistics\.mean|np\.mean|numpy\.mean)\b")
_SEEDLOOP = re.compile(r"\bfor\s+\w*seed\w*\s+in\b|\bseeds\b")
_SANCTIONED = re.compile(r"\b(ensemble|paired_delta|compare_arms)\s*\(")
_BRAIN = re.compile(r"\bBrain\s*\(")


_SEED_ADVICE = (
    "\n\n  A mean over seeds with no interval cannot support a comparison, and"
    "\n  the comparison gets made anyway. Use `diagnostics.ensemble(run, seeds)`"
    "\n  -- it refuses fewer than 3 seeds -- and `paired_delta` for an A/B."
    "\n  Judge with `Ensemble.beats(threshold)`, which reads the confidence"
    "\n  bound rather than the mean. If the site is a mean over CONDITIONS"
    "\n  rather than over seeds, raise its entry in methodology_baseline.json"
    "\n  with a comment in test_methodology_ratchet.py saying so."
)

#: JUSTIFICATION FOR THE 2026-08-07 BASELINE ADDITIONS (role-binding arc).
#: Four experiment files were flagged by `hand_rolled_seed_stats`. Every one of
#: their `np.mean` sites averages over ITEMS, not over seeds, so an interval
#: over seeds is not the missing thing:
#:
#:   retrieval_guided_rebinding.py    mean over assembly PAIRS (spread)
#:   role_area_state.py               mean over PAIRS, and over per-word MARGINS
#:   stored_assemblies_still_current.py  mean over WORDS (round-trip, freshness)
#:   supervised_vs_unsupervised_roles.py mean over per-word MARGINS, and PAIRS
#:
#: The scanner is a heuristic -- it fires when a file contains `np.mean`
#: ANYWHERE and the token `seeds` ANYWHERE -- and `_SEED_ADVICE` names exactly
#: this case ("if the site is a mean over CONDITIONS rather than over seeds").
#: Rewriting them to `ensemble()` would put a confidence interval on a
#: within-run item average, which is a different and wrong claim.
#:
#: What these files DO lack is seeds: they report 2-3. That is recorded as a
#: limit in each note rather than hidden, and it is a sampling problem, not a
#: statistic-choice problem, so it is not what this ratchet guards.
_ENGINE_ADVICE = (
    "\n\n  `Brain(...)` without `engine=` silently selects `numpy_sparse`,"
    "\n  whose candidate sampler INVENTS drive for neurons that have not fired."
    "\n  Every word-order result in this repo was measured that way because"
    "\n  WordOrderLearner had no engine parameter. Pin the engine explicitly --"
    "\n  `numpy_exact` computes the drive -- or raise the baseline entry with a"
    "\n  comment saying why the sampler is the right instrument here."
)


# Baseline notes (why an entry is above zero):
#   research/experiments/refraction_memory_numpy.py: 1 -- `np.mean(pw)` is a
#   mean over sampled PAIRS of stored assemblies within one brain (a
#   crosstalk statistic), not over seeds; the per-brain values go through
#   the ensemble helpers downstream.


def _load_baseline():
    with open(BASELINE_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["hand_rolled_seed_stats"], data["unpinned_engine"]


def _scan():
    hand, unpinned = {}, {}
    for full in python_sources(REPO):
        rel = os.path.relpath(full, REPO).replace(os.sep, "/")
        if not rel.startswith("research/"):
            continue
        try:
            text = open(full, encoding="utf-8").read()
        except Exception:                                # noqa: BLE001
            continue
        # SCAN CODE, NOT PROSE. This scanned raw text until 2026-08-05,
        # which counted `Brain(seed=)` inside a docstring EXPLAINING the
        # global-RNG hazard as a new unpinned construction -- a guard
        # punishing documentation of the thing it guards. The sibling
        # ratchet (`test_index_space_ratchet`) had the identical defect and
        # was fixed alone; the shared scanner exists so the next one cannot
        # be fixed alone again. See `_source_scan`.
        code = blank_prose(text)
        if (_SEEDY.search(code) and _SEEDLOOP.search(code)
                and not _SANCTIONED.search(code)):
            hand[rel] = len(_SEEDY.findall(code))
        n = sum(1 for line in code.splitlines()
                if _BRAIN.search(line) and "engine=" not in line)
        if n:
            unpinned[rel] = n
    return hand, unpinned


def _grown(found, baseline):
    return {p: (baseline.get(p, 0), n) for p, n in found.items()
            if n > baseline.get(p, 0)}


def _fmt(grew):
    return "\n".join(f"    {p}: {was} -> {now}"
                     for p, (was, now) in sorted(grew.items()))


def _unsafe_xfails(source: str) -> list[int]:
    """Return pytest xfails that do not explicitly make XPASS fail CI."""
    tree = ast.parse(source)
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "xfail"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "mark"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "pytest"
        ):
            continue
        strict = next((
            kw.value
            for kw in node.keywords
            if kw.arg == "strict"
        ), None)
        if not (
            isinstance(strict, ast.Constant)
            and strict.value is True
        ):
            lines.append(node.lineno)
    return lines


def test_no_new_hand_rolled_seed_statistics():
    base, _ = _load_baseline()
    found, _ = _scan()
    grew = _grown(found, base)
    assert not grew, (
        "new or increased hand-rolled seed statistics:\n" + _fmt(grew)
        + _SEED_ADVICE)


def test_no_new_unpinned_engine_constructions():
    _, base = _load_baseline()
    _, found = _scan()
    grew = _grown(found, base)
    assert not grew, (
        "new or increased `Brain()` without an explicit engine:\n" + _fmt(grew)
        + _ENGINE_ADVICE)


def test_non_strict_xfail_scanner_has_a_true_negative():
    source = """
import pytest
@pytest.mark.xfail(strict=False, reason='unstable')
def test_claim(): ...
@pytest.mark.xfail(reason='defaults are also non-strict')
def test_other_claim(): ...
"""
    assert _unsafe_xfails(source) == [3, 5]


def test_no_non_strict_expected_failures():
    """An unexpected pass is a changed result and must stop for review."""
    found = {}
    for full in python_sources(REPO):
        rel = os.path.relpath(full, REPO).replace(os.sep, "/")
        if not rel.startswith("neural_assemblies/tests/"):
            continue
        lines = _unsafe_xfails(open(full, encoding="utf-8").read())
        if lines:
            found[rel] = lines
    assert not found, (
        "non-strict expected failures let changed scientific outcomes pass CI; "
        f"use strict=True or replace the unstable instrument: {found}"
    )


def test_baselines_are_not_stale():
    """Entries above reality should be lowered, or the ratchet loses its grip.

    The other direction matters as much: a baseline that drifts above the real
    count silently stops catching anything, which is how a guard becomes
    decoration.
    """
    hb, eb = _load_baseline()
    hf, ef = _scan()
    stale = {f"seed:{p}": (c, hf.get(p, 0)) for p, c in hb.items()
             if hf.get(p, 0) < c}
    stale.update({f"engine:{p}": (c, ef.get(p, 0)) for p, c in eb.items()
                  if ef.get(p, 0) < c})
    assert not stale, (
        "baseline is above the real count -- lower these:\n"
        + "\n".join(f"    {p}: baseline {was}, actual {now}"
                    for p, (was, now) in sorted(stale.items())))

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
"""

from __future__ import annotations

import json
import os
import re

from ._source_scan import blank_prose

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "methodology_baseline.json")

_SEEDY = re.compile(r"\b(statistics\.mean|np\.mean|numpy\.mean)\b")
_SEEDLOOP = re.compile(r"\bfor\s+\w*seed\w*\s+in\b|\bseeds\b")
_SANCTIONED = re.compile(r"\b(ensemble|paired_delta|compare_arms)\s*\(")
_BRAIN = re.compile(r"\bBrain\s*\(")

_SKIP_DIRS = (".git", "__pycache__", ".venv", "node_modules", ".reference",
              ".pytest_cache", "legacy")

_SEED_ADVICE = (
    "\n\n  A mean over seeds with no interval cannot support a comparison, and"
    "\n  the comparison gets made anyway. Use `diagnostics.ensemble(run, seeds)`"
    "\n  -- it refuses fewer than 3 seeds -- and `paired_delta` for an A/B."
    "\n  Judge with `Ensemble.beats(threshold)`, which reads the confidence"
    "\n  bound rather than the mean. If the site is a mean over CONDITIONS"
    "\n  rather than over seeds, raise its entry in methodology_baseline.json"
    "\n  with a comment in test_methodology_ratchet.py saying so."
)

_ENGINE_ADVICE = (
    "\n\n  `Brain(...)` without `engine=` silently selects `numpy_sparse`,"
    "\n  whose candidate sampler INVENTS drive for neurons that have not fired."
    "\n  Every word-order result in this repo was measured that way because"
    "\n  WordOrderLearner had no engine parameter. Pin the engine explicitly --"
    "\n  `numpy_exact` computes the drive -- or raise the baseline entry with a"
    "\n  comment saying why the sampler is the right instrument here."
)


def _load_baseline():
    with open(BASELINE_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["hand_rolled_seed_stats"], data["unpinned_engine"]


def _scan():
    hand, unpinned = {}, {}
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(root, fn)
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

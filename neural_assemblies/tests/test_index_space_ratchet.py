"""Ratchet against the compact-index / neuron-ID confusion.

THE BUG THIS EXISTS FOR. An area has two coordinate systems. ``area.winners``
holds COMPACT engine indices -- positions in the engine's materialised arrays,
renumbered as the area recruits. ``Assembly.winners`` holds NEURON IDs -- stable
identities in the notional population of ``n``. ``ops._snap`` is the one-way door
between them.

Intersecting one with the other compares unrelated integer sets, so the overlap
is whatever two arbitrary sets happen to share. That reads as EXACTLY CHANCE and
is INVARIANT TO EVERY PARAMETER, which is indistinguishable from a real negative
result and is much more convincing than one. It has cost this project three
results:

  * the merge line, silently voided
  * "role retrieval is at chance in every configuration", which survived a 25x
    sweep of beta and produced a whole retracted theory about the mechanism
    lacking an addressable key. With the index spaces matched, the same code in
    the same arms retrieves at 1.000.
  * a LEX-reproduction precondition gate that read 0.020 -- the same bug, inside
    the check written to catch this class of bug.

WHY A RATCHET AND NOT A BAN. Most of the 58 existing sites are self-consistent:
they compare ``area.winners`` to ``area.winners`` from the same run, which is
fine because both are compact. The defect is MIXING the two spaces, which needs
dataflow analysis to detect properly. So this test does the cheap, honest thing:
it freezes the current count per file and fails if it grows, or if a new file
starts doing it. That does not fix history and does not pretend to -- it stops
the pattern spreading, and points new code at the sanctioned readout.

THE SANCTIONED READOUT is ``diagnostics.read_assembly(brain, area)``, which
returns stable neuron IDs and is directly comparable with a stored
``Assembly.winners``. Use ``diagnostics.assembly_overlap`` to compare them.

To update the baseline after legitimately removing sites, lower the number.
Raising a number requires a comment here saying why the new site is
space-consistent.
"""

from __future__ import annotations

import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

#: `area.winners` accessed on the same line as a set/overlap operation.
_ACCESS = re.compile(r"areas\[[^\]]*\]\.winners")
_COMPARE = re.compile(r"set\(|&|overlap|intersection")

#: Engine internals legitimately work in compact space throughout.
_EXEMPT = ("/core/", "engine")

#: Frozen baseline: path -> number of risky lines. Verified 2026-07-29.
BASELINE = {
    "research/experiments/run_quick_validation.py": 9,
    "legacy/root_modules/simulations.py": 6,
    "research/experiments/applications/test_language_syntax.py": 5,
    "neural_assemblies/simulation/advanced_simulations.py": 4,
    "research/experiments/primitives/test_bidirectional_association.py": 4,
    "research/experiments/primitives/test_projection.py": 4,
    "research/experiments/primitives/diagnose_erp_dynamics.py": 3,
    "research/experiments/primitives/test_association.py": 3,
    "research/experiments/primitives/test_inhibition.py": 3,
    "neural_assemblies/assembly_calculus/emergent/evaluation/erp/adapters.py": 2,
    "neural_assemblies/assembly_calculus/metrics/instability.py": 2,
    "research/experiments/metrics/measurement.py": 2,
    "research/experiments/primitives/test_merge.py": 2,
    "research/experiments/stability/test_noise_robustness.py": 2,
    "legacy/scripts/simulations/overlap_sim.py": 1,
    "neural_assemblies/nemo/language/emergent/tests/generation/"
    "test_original_pattern_completion.py": 1,
    "neural_assemblies/simulation/association_simulator.py": 1,
    "neural_assemblies/simulation/pattern_completion.py": 1,
    "neural_assemblies/tests/test_ac_conformance.py": 1,
    "research/experiments/distinctiveness/test_competition_mechanisms.py": 1,
    "research/experiments/stability/test_phase_diagram.py": 1,
}

_ADVICE = (
    "\n\n  `area.winners` is COMPACT ENGINE INDICES; `Assembly.winners` is "
    "NEURON IDs.\n  Comparing them reads exactly chance, invariant to every "
    "parameter, and looks\n  like a real negative result. Use "
    "`diagnostics.read_assembly(brain, area)` for\n  the live side and "
    "`diagnostics.assembly_overlap` to compare.\n  If the new site really is "
    "compact-vs-compact, raise its BASELINE entry in\n  "
    "neural_assemblies/tests/test_index_space_ratchet.py with a comment "
    "saying why."
)


def _scan():
    found = {}
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in
                   (".git", "__pycache__", ".venv", "node_modules",
                    ".reference", ".pytest_cache")]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, REPO).replace(os.sep, "/")
            if any(x in rel for x in _EXEMPT):
                continue
            try:
                text = open(full, encoding="utf-8").read()
            except Exception:                                # noqa: BLE001
                continue
            n = sum(1 for line in text.splitlines()
                    if _ACCESS.search(line) and _COMPARE.search(line))
            if n:
                found[rel] = n
    return found


def test_no_new_index_space_comparisons():
    found = _scan()
    grew = {p: (BASELINE.get(p, 0), n) for p, n in found.items()
            if n > BASELINE.get(p, 0)}
    assert not grew, (
        "new or increased raw compact-index comparisons:\n"
        + "\n".join(f"    {p}: {was} -> {now}"
                    for p, (was, now) in sorted(grew.items()))
        + _ADVICE
    )


def test_baseline_is_not_stale():
    """Baseline entries that no longer exist should be lowered.

    Keeps the ratchet honest in the other direction: a baseline that drifts
    above reality stops catching anything.
    """
    found = _scan()
    stale = {p: (c, found.get(p, 0)) for p, c in BASELINE.items()
             if found.get(p, 0) < c}
    assert not stale, (
        "BASELINE is above the real count -- lower these so the ratchet keeps "
        "its grip:\n"
        + "\n".join(f"    {p}: baseline {was}, actual {now}"
                    for p, (was, now) in sorted(stale.items()))
    )

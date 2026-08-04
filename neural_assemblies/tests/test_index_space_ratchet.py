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


# ---------------------------------------------------------------------------
# `.w` -- the SAME defect class, a second quantity pair
#
# TIER A + C of research/plans/TYPE_SAFETY_PROGRAM.md. Added here rather than in
# a new file BECAUSE it is the same class: one attribute name, two quantities,
# wrong read returns a plausible number. A separate ratchet would be the fifth
# mechanism for a class that already has one.
#
# THE PAIR:
#     sparse engine    `w` = neurons MATERIALIZED
#     explicit engine  `w` = len(winners), i.e. k -- not an extent at all
#
# MEASURED on one area at one instant: 247 against 50. Slicing a fiber by the
# wrong one kept only the ~k^2/n assembly neurons whose GLOBAL id happened to
# fall below k, and the quantity being measured through the three candidate
# extents (`w`, `_log_cols`, `weights.shape[1]` = 247/213/367) read
# 1.87 / 1.65 / 2.84. I made that mistake myself writing the harness for #69.
#
# SANCTIONED READS, which name what they return and give None for "not
# applicable" rather than 0:
#     engine.materialized_count(area)    neurons that exist
#     engine.fiber_extent(src, dst)      columns of one fiber
#
# Same exemptions as above: `core/` and `engine` own the space by construction.
# ---------------------------------------------------------------------------

#: `.w` as a whole attribute -- not `.weights`, `.winners`, `.w_max`.
_W_ACCESS = re.compile(r"\.w\b(?!_)")

#: Frozen baseline: path -> COUNT OF `.w` ATTRIBUTE READS (tokenized, so
#: strings and comments do not count). Re-frozen 2026-08-04 when the scan
#: moved from a line regex to tokenize: 38 files / 115 reads outside core,
#: down from "44 files / 136 lines" -- 21 of those were PROSE, and six files
#: had nothing but prose. The old number was part code and part commentary,
#: so deleting a real read while adding a comment about it netted to zero.
W_BASELINE = {
    "legacy/root_modules/simulations.py": 13,
    "neural_assemblies/simulation/advanced_simulations.py": 12,
    "legacy/root_modules/image_learner.py": 8,
    "neural_assemblies/assembly_calculus/emergent/parser_mixins/incremental.py": 7,
    "neural_assemblies/assembly_calculus/emergent/training/linker.py": 6,
    "research/experiments/capacity/lexicon_capacity.py": 6,
    "research/experiments/metrics/measurement.py": 4,
    "research/experiments/recruitment/recruitment_mechanisms.py": 4,
    "tests/test_brain_core.py": 4,
    "legacy/scripts/simulations/turing_sim.py": 3,
    "neural_assemblies/assembly_calculus/emergent/parser_mixins/state_prediction.py": 3,
    "neural_assemblies/assembly_calculus/emergent/parser_mixins/unsupervised.py": 3,
    "neural_assemblies/simulation/turing_simulations.py": 3,
    "research/experiments/_substrate.py": 3,
    "research/experiments/p600_metric_comparison.py": 3,
    "research/experiments/primitives/diagnose_erp_dynamics.py": 3,
    "research/experiments/recruitment/smoke.py": 3,
    "neural_assemblies/assembly_calculus/consolidation.py": 2,
    "neural_assemblies/assembly_calculus/emergent/training/compiler.py": 2,
    "neural_assemblies/programs/colt_mnist_tier_a.py": 2,
    "neural_assemblies/programs/patch_merge.py": 2,
    "research/experiments/distinctiveness/test_competition_mechanisms.py": 2,
    "research/experiments/recurrent_assembly_decay.py": 2,
    "legacy/root_modules/parser.py": 1,
    "legacy/root_modules/recursive_parser.py": 1,
    "neural_assemblies/assembly_calculus/binding.py": 1,
    "neural_assemblies/assembly_calculus/emergent/evaluation/erp/adapters.py": 1,
    "neural_assemblies/assembly_calculus/tracing/operations.py": 1,
    "neural_assemblies/language/debugger.py": 1,
    "neural_assemblies/language/parser.py": 1,
    "neural_assemblies/programs/colt_mnist_hierarchical_brain.py": 1,
    "neural_assemblies/programs/colt_mnist_visual_advanced_brain.py": 1,
    "neural_assemblies/simulation/density_simulator.py": 1,
    "research/experiments/erp_p600_probe_contamination.py": 1,
    "research/experiments/metrics/instability.py": 1,
    "research/experiments/metrics/settling.py": 1,
    "research/experiments/recruitment/diagnose_synaptic_scaling.py": 1,
    "research/experiments/worker_divergence_probe.py": 1,
}

_W_ADVICE = (
    "\n\n  `.w` means TWO different quantities: neurons MATERIALIZED on the"
    "\n  sparse engine, but len(winners) == k on the explicit engine. Measured"
    "\n  247 vs 50 for one area at one instant. Prefer a read that names what"
    "\n  it returns, and that gives None for 'not applicable' rather than 0:"
    "\n"
    "\n      engine.materialized_count(area)   neurons that exist"
    "\n      engine.fiber_extent(src, dst)     columns of one fiber"
    "\n"
    "\n  If this site is genuinely engine-internal, raise its W_BASELINE entry"
    "\n  with a comment saying which quantity it means and why."
)


def _count_w_reads(text: str) -> int:
    """Count `.w` ATTRIBUTE READS, ignoring strings and comments.

    WHY TOKENIZE AND NOT A REGEX ON LINES. The line scan this replaces counted
    prose: a docstring warning that `.w` is the wrong divisor read as a NEW
    ambiguous access and failed the ratchet, which punishes documenting the
    hazard the ratchet exists to track. Worse, it means the frozen baselines
    were part code and part commentary, so an edit that deleted a real read and
    added a comment about it netted to zero.

    Tokenizing keeps the guard strictly stronger: only an OP `.` followed by
    NAME `w` counts, which is an attribute access and nothing else.

    Falls back to the old line scan when a file will not tokenize -- some
    research scripts do not parse -- because silently counting zero there would
    be a hole in a guard whose whole job is to have no holes.
    """
    import io
    import tokenize

    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return sum(1 for line in text.splitlines() if _W_ACCESS.search(line))

    n = 0
    prev_op_dot = False
    for tok in toks:
        if tok.type == tokenize.OP and tok.string == ".":
            prev_op_dot = True
            continue
        if prev_op_dot and tok.type == tokenize.NAME and tok.string == "w":
            n += 1
        prev_op_dot = False
    return n


def _scan_w():
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
            # `/tests/` excluded as well as the engine: a test that pins the
            # ambiguity deliberately (there are several) must not be flagged.
            if any(x in rel for x in _EXEMPT) or "/tests/" in rel:
                continue
            try:
                text = open(full, encoding="utf-8").read()
            except Exception:                                # noqa: BLE001
                continue
            n = _count_w_reads(text)
            if n:
                found[rel] = n
    return found


def test_no_new_ambiguous_w_reads():
    found = _scan_w()
    grew = {p: (W_BASELINE.get(p, 0), n) for p, n in found.items()
            if n > W_BASELINE.get(p, 0)}
    assert not grew, (
        "new or increased ambiguous `.w` reads outside the engine:\n"
        + "\n".join(f"    {p}: {was} -> {now}"
                    for p, (was, now) in sorted(grew.items()))
        + _W_ADVICE
    )


def test_w_baseline_is_not_stale():
    found = _scan_w()
    stale = {p: (c, found.get(p, 0)) for p, c in W_BASELINE.items()
             if found.get(p, 0) < c}
    assert not stale, (
        "W_BASELINE is above the real count -- lower these:\n"
        + "\n".join(f"    {p}: baseline {was}, actual {now}"
                    for p, (was, now) in sorted(stale.items()))
    )


def test_both_scanners_still_see_something():
    """A ratchet whose regex has stopped matching passes forever and protects
    nothing. Pin that each scanner is still looking at real code."""
    assert _scan(), "compact-index scanner matched nothing -- check _ACCESS"
    w = _scan_w()
    assert sum(w.values()) > 50, (
        f"`.w` scanner found only {sum(w.values())} lines; the baseline was "
        f"built at 136, so it has probably stopped matching")


def test_the_w_scan_counts_CODE_and_not_PROSE():
    """The guard on the guard, and it caught a real own-goal.

    A docstring warning that `.w` is the wrong divisor used to read as a NEW
    ambiguous access and fail the ratchet -- punishing documentation of the
    exact hazard the ratchet tracks. It also meant the frozen numbers were part
    code and part commentary, so removing a real read while adding a comment
    about it netted to zero. Re-freezing found 21 of 136 tracked "reads" were
    prose, and six files had NOTHING but prose.
    """
    src = (
        '"""A docstring mentioning area.w and brain.w and obj.w."""\n'
        "# a comment about foo.w\n"
        "x = 'a string with bar.w in it'\n"
        "def f(area):\n"
        "    return area.w\n"                       # the ONLY real read
    )
    assert _count_w_reads(src) == 1, (
        "the `.w` scan is counting strings or comments again; the baseline is "
        "no longer a count of attribute reads")


def test_the_w_scan_still_counts_real_reads_in_several_forms():
    """The positive control. A tokenizer that returned 0 would pass the test
    above and silently disable the whole ratchet."""
    src = (
        "a = area.w\n"
        "b = brain.areas['X'].w + 1\n"
        "c = self.w\n"
        "d = obj.w_max\n"                           # NOT a match: `w_max`
        "e = w\n"                                   # NOT a match: bare name
    )
    assert _count_w_reads(src) == 3


def test_the_w_scan_falls_back_when_a_file_will_not_tokenize():
    """Unparseable research scripts must not silently count zero.

    A guard whose failure mode is "sees nothing" is worse than no guard, since
    it reports success. On a tokenize error the scan reverts to the old line
    regex, which over-counts rather than under-counts.
    """
    broken = "def f(:\n    return area.w\n"
    assert _count_w_reads(broken) >= 1

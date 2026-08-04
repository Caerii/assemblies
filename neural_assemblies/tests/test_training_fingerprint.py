"""The backbone cache fingerprint must cover everything training executes.

WHY THIS FILE EXISTS.  `ParserCache` pickles trained backbones to disk and keys
the filename on a hash of "the sources that determine a trained backbone". When
that list is incomplete the cache serves a backbone trained by OLD code, and
the result is not an error -- it is a plausible number. It has already produced
two false conclusions in this repo:

  * a role-training change whose A/B returned byte-identical numbers in both
    arms, because neither arm trained anything;
  * a k-WTA sampler rewrite that appeared to move `test_erp_calibration`'s
    p600 Cohen's d below its threshold, reproducibly, while being verifiably
    bit-identical on all 2013 calls of a live run. The backbone had been
    trained under the old sampler and calibrated under the new one.

Reproducibility is no defence: a stale pickle is perfectly deterministic.

So the list is checked against a MEASUREMENT rather than against memory. The
test traces a real projection run and asserts every module that actually
executed is covered by the fingerprint.
"""

import os
import sys
from pathlib import Path

import pytest

import neural_assemblies
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (
    fingerprint_source_files,
    training_code_fingerprint,
)
from neural_assemblies.core.brain import Brain

PKG = Path(neural_assemblies.__file__).resolve().parent


def _run_a_little_training():
    """The smallest thing that is unambiguously training.

    Stimulus drive, area-to-area projection, recurrence, plasticity, growth,
    materialisation and a frozen read -- enough to pull in the seeding, pricing,
    sampling and winner-selection paths. Deliberately tiny: this runs under a
    tracer.
    """
    b = Brain(p=0.05, save_winners=True, seed=1, engine="numpy_sparse")
    b.add_stimulus("s", 20)
    b.add_area("A", 500, 20, 0.1)
    b.add_area("B", 500, 20, 0.1)
    for _ in range(4):
        b.project({"s": ["A"]}, {"A": ["B"], "B": ["A"]})
    b._engine.materialize_area("A")
    with b.frozen():
        b.project({}, {"A": ["A"]})
    return b


def _train_a_little_parser():
    """A REAL parser build, which is what actually produces a backbone.

    WHY THIS EXISTS. The tracer used to see only `_run_a_little_training()` --
    a bare `Brain`. That proves coverage of `core/` and `compute/`, which are
    hashed wholesale anyway, and proves NOTHING about the language layer where
    every cached backbone is actually trained. The enumeration of that layer
    then missed `acquisition/pos_inference.py` and `parser_mixins/prediction.py`
    -- the two files whose edits fixed #80 -- and the guard written to catch
    exactly this could not see them.

    Third time one shape has bitten: a guard whose exercised path is not the
    path that produces the artifact. #21's hashseed guard exercised
    `p.train(...)` and never the curriculum; my own replacement guard ran at
    n=1000 and passed with the fix reverted; this one traced a bare Brain.

    Kept tiny (n=300, k=10, a few sentences) because it runs under `settrace`.
    It does not need to train WELL -- it needs to EXECUTE the modules.
    """
    from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
        create_training_sentences,
    )
    from neural_assemblies.assembly_calculus.emergent.parser import (
        EmergentParser,
    )

    p = EmergentParser(n=300, k=10, seed=1, fast_training=True)
    p.train(create_training_sentences()[:4])
    return p


def _executed_modules():
    """Package-relative paths of every neural_assemblies file that ran code.

    BOTH workloads are traced: the bare-Brain one for the engine internals and
    a real parser build for the language layer. Running only the first is the
    hole this guard had.
    """
    _run_a_little_training()          # warm imports; import-time code is not
    _train_a_little_parser()          # what we are asking about
    executed = set()

    def tracer(frame, event, _arg):
        if event == "call":
            fn = frame.f_code.co_filename
            if fn.startswith(str(PKG)):
                executed.add(fn)
        return None

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        _run_a_little_training()
        _train_a_little_parser()
    finally:
        sys.settrace(old)

    out = set()
    for f in executed:
        try:
            rel = Path(f).resolve().relative_to(PKG).as_posix()
        except ValueError:            # pragma: no cover - outside the package
            continue
        # This file's own frames are on the stack while it traces. Test code is
        # not shipped and cannot train a backbone.
        if rel.startswith("tests/"):
            continue
        out.add(rel)
    return out


#: Modules that execute during training but genuinely cannot change a trained
#: backbone. Each needs a REASON, because "it's probably fine" is how the list
#: became wrong in the first place. Empty today -- `core/` and `compute/` are
#: hashed wholesale, so nothing under them needs excusing.
ALLOWED_UNCOVERED: dict = {}


def test_everything_training_executes_is_fingerprinted():
    executed = _executed_modules()
    assert executed, "tracer captured nothing -- the test is vacuous"

    covered = set(fingerprint_source_files())
    uncovered = {m for m in executed
                 if m not in covered and m not in ALLOWED_UNCOVERED}

    assert not uncovered, (
        "these modules run during training but are NOT covered by the backbone "
        "cache fingerprint, so editing them serves a stale pickle instead of "
        "retraining:\n  " + "\n  ".join(sorted(uncovered))
        + "\n\nAdd them to _TRAINING_SOURCES (or their package to "
          "_TRAINING_SOURCE_DIRS), or list them in ALLOWED_UNCOVERED with a "
          "reason they cannot affect a trained backbone.")


def test_the_engine_internals_are_covered():
    """Names the specific misses that have already cost time.

    The test above is the general guard; this one fails with a recognisable
    message if the packages stop being hashed wholesale. `_seeding.py` decides
    every synapse's initial weight and `_pricing.py` decides who wins -- both
    were outside the list until 2026-07-31.
    """
    covered = set(fingerprint_source_files())
    for critical in ("core/numpy_engine/_seeding.py",
                     "core/numpy_engine/_sparse.py",
                     "core/_pricing.py",
                     "compute/sparse_simulation.py",
                     "compute/winner_selection.py"):
        assert critical in covered, f"{critical} is not fingerprinted"


def test_fingerprint_is_order_independent(monkeypatch):
    """Two enumerations of the same tree must give the same hash.

    `rglob` order is filesystem-dependent, so an unsorted fingerprint would
    give the same code two different cache keys on two machines -- silently
    halving the cache hit rate and, worse, making a cross-machine comparison
    look like a code change.
    """
    files = fingerprint_source_files()
    assert list(files) == sorted(files)


def test_fingerprint_changes_when_a_covered_file_changes(monkeypatch):
    """The mechanism, not just the list: touching a covered file must move it.

    Guards against the fingerprint silently degrading to a constant -- which
    would make every check above pass while restoring the original bug.
    """
    import neural_assemblies.assembly_calculus.emergent.evaluation.sweep as SW

    monkeypatch.setattr(SW, "_CODE_FINGERPRINT", None)
    before = training_code_fingerprint()

    target = PKG / "core" / "numpy_engine" / "_seeding.py"
    st = target.stat()
    monkeypatch.setattr(SW, "_CODE_FINGERPRINT", None)
    try:
        # Move mtime forward; the fingerprint hashes size and mtime.
        os.utime(target, (st.st_atime, st.st_mtime + 120))
        after = training_code_fingerprint()
    finally:
        os.utime(target, (st.st_atime, st.st_mtime))
        monkeypatch.setattr(SW, "_CODE_FINGERPRINT", None)

    assert before != after, (
        "touching _seeding.py did not change the fingerprint -- the cache "
        "would serve a backbone trained before the edit")


def test_ignore_switch_still_works(monkeypatch):
    """The documented escape hatch must survive the rewrite."""
    import neural_assemblies.assembly_calculus.emergent.evaluation.sweep as SW

    monkeypatch.setenv("ASSEMBLIES_IGNORE_CODE_FINGERPRINT", "1")
    monkeypatch.setattr(SW, "_CODE_FINGERPRINT", None)
    try:
        assert training_code_fingerprint() == "ignored"
    finally:
        monkeypatch.setattr(SW, "_CODE_FINGERPRINT", None)


@pytest.mark.parametrize("pkg_dir", ["core", "compute"])
def test_the_hashed_packages_are_not_empty(pkg_dir):
    """A typo'd directory name would hash nothing and fail silently open."""
    covered = fingerprint_source_files()
    assert any(f.startswith(pkg_dir + "/") for f in covered), (
        f"no files under {pkg_dir}/ are fingerprinted -- "
        f"_TRAINING_SOURCE_DIRS may be misspelled")

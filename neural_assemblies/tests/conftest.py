"""Shared pytest fixtures — amortize parser training across test session."""

from __future__ import annotations

import os

import pytest

# xdist parallelizes at the test-process level, so native thread-pool tuning
# is machine-dependent.  Leave library defaults untouched unless a benchmark
# explicitly supplies ASSEMBLIES_TEST_NATIVE_THREADS; then apply that value
# consistently before NumPy/Torch import.  A whole-suite measurement showed
# that forcing one thread regresses parser/example tests even when it helps a
# narrow CUDA test file.
_NATIVE_THREADS = os.environ.get("ASSEMBLIES_TEST_NATIVE_THREADS")
if _NATIVE_THREADS:
    for _native_var in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS", "TORCH_NUM_INTEROP_THREADS",
    ):
        os.environ.setdefault(_native_var, _NATIVE_THREADS)

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (
    DEFAULT_K,
    DEFAULT_N,
    ParserCache,
    get_parser_cache,
)

N, K = DEFAULT_N, DEFAULT_K


# --- Test tiering -----------------------------------------------------------
# The fast dev loop is `pytest -m "not slow" -n auto --dist loadfile`. Profiling
# the full suite showed a small set of heavy end-to-end research / diagnostic /
# stress / MNIST files dominate the wall-clock (e.g. test_novel_chat ~746s,
# test_h9_diagnostic ~633s, test_wobbly_stress ~214s per run) -- validation that
# belongs in CI, not the dev iteration loop. They are marked slow CENTRALLY here
# rather than scattering pytestmark across the files (two of which don't even
# import pytest), reusing the `slow` marker already registered in
# pyproject.toml. Run the full suite with `pytest -m slow` or no marker filter.
# (The 59-min test_evidence_suite_full_ladder / test_ventral_evidence module is
# already marked slow at its own source.)
_SLOW_FILE_STEMS = frozenset({
    "test_novel_chat",
    "test_h9_diagnostic",
    "test_patch_merge",
    "test_colt_mnist_protocol",
    "test_ventral_push",
    "test_wobbly_stress",
})


# MNIST vision-benchmark tests (colt2022 protocols) are heavy CI validation and
# dominate the remaining fast-tier wall-clock (the colt2022_mnist parity
# protocols alone sum ~300s on one xdist worker: 130s+58s+42s+41s+30s). They
# appear across several files (test_parity_infrastructure parametrized ids,
# test_literature_golden Colt2022Mnist* classes), so match them by nodeid.
_SLOW_NODEID_SUBSTRINGS = (
    "colt2022_mnist", "Colt2022Mnist", "mnist",
    # Empirical calibration/learnability gates measured above one minute each
    # in the maintained-suite duration profile. Keep them in the full suite,
    # but out of the fast contract loop.
    "test_dual_metric_learnability_gate",
    "test_calibration_separates_category_violation_from_grammatical",
    "test_calibration_mode_does_not_change_observations",
    "test_calibration_reports_auc_and_span",
    "test_sentences_depth_holdout_bootstrap_floor",
    "test_stage2_expands_vocab",
    "test_session_bootstrap_conversation_smoke",
    "test_unknown_word_registered_on_interact",
    "test_train_for_conversation_registers_prediction",
    "test_dialogue_stage_runs_dialogue_phase",
    "test_scaled_vocabulary_builds",
    "test_same_seed_same_classifications",
    "test_evaluation_on_curriculum_parser",
    "test_predict_returns_ranked_list",
    # Remaining end-to-end ERP/parser probes dominate the fast tier even with
    # EMERGENT_*_FAST enabled. Keep their empirical signal in the slow tier so
    # contract failures surface quickly during local development.
    "test_grammatical_excess_is_crushed_against_the_floor",
    "test_stage3_learns_roles",
    "test_the_control_arm_is_alive",
    "test_calibrate_mine_replay_pipeline",
    "test_a_parse_grows_a_trained_parser",
    "test_scaled_lexicon_trains",
    "test_recall_meets_theorem_3_bound",
    # Class-scoped fixtures train large parsers once but still impose a
    # 60--90s setup cost on the worker that owns the file. Keep the complete
    # empirical class together in the slow tier; splitting one assertion out
    # leaves the expensive fixture in the fast loop.
    "TestCurriculumLearning",
    "TestScaledVocabulary",
    "TestColtMultiAssembly",
    # Exploratory parameter sweeps intentionally retrain the same sequence
    # under many settings; they are scientific evidence, not contract smoke.
    "TestBestParameterDemo",
    # ERP and parser-idempotence diagnostics currently perform full training
    # and replay runs (130--140s each). Keep their complete classes together
    # so class/module fixtures do not leak that cost into the contract loop.
    "TestRawQuantityIsSaturated",
    "TestTheProbedAreasAreReal",
    "TestParseIsNotIdempotent",
    "TestErpCalibration",
)


def pytest_collection_modifyitems(config, items):
    """Mark heavy CI-tier files slow so `-m "not slow"` stays a fast dev loop."""
    for item in items:
        if getattr(item.path, "stem", None) in _SLOW_FILE_STEMS:
            item.add_marker(pytest.mark.slow)
            continue
        nodeid = item.nodeid.lower()
        if any(s.lower() in nodeid for s in _SLOW_NODEID_SUBSTRINGS):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def _restore_array_backend():
    """Undo the global backend flip that constructing a GPU engine performs.

    ``cupy_engine.py`` and ``cuda_engine.py`` call ``set_backend("cupy")`` in
    their constructors. The backend is PROCESS-GLOBAL and never restored, so
    one test that builds a GPU engine hands CuPy arrays to every numpy-engine
    test that runs after it in the same process, which then dies on "Implicit
    conversion to a NumPy array is not allowed".

    The tests it takes down are the ones written to guard materialisation and
    CSR storage -- so they pass in isolation and fail in a full-suite run,
    which is the failure mode least likely to be believed and most likely to be
    dismissed as flakiness. It also only reproduces where CuPy is actually
    installed, so CI never sees it.

    Engine-owned state now pins its array module, so ordinary NumPy/Sparse
    projections are isolated. The fixture remains because legacy adapters and
    optional GPU constructors still mutate the compatibility selector; it keeps
    those boundaries from contaminating unrelated tests until that adapter is
    retired.
    for library users, who can still construct a CuPy engine and find their
    next numpy Brain broken.
    """
    from neural_assemblies.core import backend as _backend

    before = _backend._xp
    try:
        yield
    finally:
        if _backend._xp is not before:
            _backend._xp = before


@pytest.fixture(scope="session")
def parser_cache() -> ParserCache:
    return get_parser_cache()


@pytest.fixture(scope="session")
def sentences_parser(parser_cache):
    """SENTENCES-depth parser (seed=42, default holdouts) — read-mostly."""
    return parser_cache.get("SENTENCES", seed=42, calibrate=False)


@pytest.fixture(scope="session")
def calibrated_sentences_parser(parser_cache):
    """SENTENCES parser with empirical ERP calibration cached."""
    return parser_cache.get("SENTENCES", seed=42, calibrate=True)


@pytest.fixture(scope="session")
def two_word_parser(parser_cache):
    return parser_cache.get("TWO_WORD", seed=42, calibrate=False)


@pytest.fixture(scope="session")
def forked_parser(parser_cache):
    """Factory for INDEPENDENT trained parsers, sharing one training run.

    The ``*_parser`` fixtures above hand back the cache's shared object, so
    they are only safe to read. Any test that mutates its parser -- ERP
    calibration, extra exposure, lexicon writes -- should call this instead::

        def test_x(forked_parser):
            parser = forked_parser("SENTENCES", seed=42)

    Training is still paid once per (depth, seed, holdout, n, k); only the
    much cheaper clone is per-test.
    """
    def _make(depth: str = "SENTENCES", *, wobbly: bool = False, **kwargs):
        return parser_cache.fork(depth, wobbly=wobbly, **kwargs)

    return _make

"""Shared pytest fixtures — amortize parser training across test session."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (
    DEFAULT_K,
    DEFAULT_N,
    ParserCache,
    get_parser_cache,
    reset_parser_cache,
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
_SLOW_NODEID_SUBSTRINGS = ("colt2022_mnist", "Colt2022Mnist", "mnist")


def pytest_collection_modifyitems(config, items):
    """Mark heavy CI-tier files slow so `-m "not slow"` stays a fast dev loop."""
    for item in items:
        if getattr(item.path, "stem", None) in _SLOW_FILE_STEMS:
            item.add_marker(pytest.mark.slow)
            continue
        nodeid = item.nodeid.lower()
        if any(s.lower() in nodeid for s in _SLOW_NODEID_SUBSTRINGS):
            item.add_marker(pytest.mark.slow)


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

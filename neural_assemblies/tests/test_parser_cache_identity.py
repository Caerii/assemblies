"""Changing an instrument request must not reuse another request's state."""
from types import SimpleNamespace

import pytest

from neural_assemblies.assembly_calculus.emergent.evaluation import sweep, generalization


@pytest.fixture
def harness(monkeypatch, tmp_path):
    calls = []

    def train(depth, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(request=kwargs, _category_cache={})

    monkeypatch.setattr(generalization, "train_parser_to_depth", train)
    monkeypatch.setattr(sweep, "backbone_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(sweep, "training_code_fingerprint", lambda: "fixture")
    return calls


def test_engine_change_misses_memory_and_disk_cache(monkeypatch, harness):
    cache = sweep.ParserCache()
    monkeypatch.setenv("ASSEMBLIES_ENGINE", "numpy_sparse")
    first = cache.get("TWO_WORD")
    monkeypatch.setenv("ASSEMBLIES_ENGINE", "numpy_exact")
    second = cache.get("TWO_WORD")
    assert second is not first
    assert [call["engine"] for call in harness] == ["numpy_sparse", "numpy_exact"]
    third = sweep.ParserCache().get("TWO_WORD")
    assert third.request == second.request
    assert len(harness) == 2


def test_fast_training_is_disk_identity(harness):
    sweep.ParserCache().get("TWO_WORD", fast_training=True)
    slow = sweep.ParserCache().get("TWO_WORD", fast_training=False)
    assert slow.request["fast_training"] is False
    assert len(harness) == 2


def test_default_holdout_resolves_before_keying(harness):
    cache = sweep.ParserCache()
    default = cache.get("TWO_WORD")
    explicit = cache.get("TWO_WORD", holdout_words=generalization.default_holdout_set())
    empty = cache.get("TWO_WORD", holdout_words=set())
    assert explicit is default
    assert empty is not default
    assert empty.request["holdout_words"] == set()


def test_engine_environment_changes_are_key_material(monkeypatch, harness):
    cache = sweep.ParserCache()
    monkeypatch.setenv("ASSEMBLIES_STREAM_INIT", "0")
    first = cache.get("TWO_WORD")
    monkeypatch.setenv("ASSEMBLIES_STREAM_INIT", "1")
    second = cache.get("TWO_WORD")
    assert second is not first


def test_calibration_variants_share_training_but_not_calibrated_state(monkeypatch, harness):
    modes = []
    def calibrate(parser, *, fast):
        modes.append(fast)
        assert not hasattr(parser, "threshold_mode")
        parser.threshold_mode = fast
    monkeypatch.setattr(
        "neural_assemblies.assembly_calculus.emergent.evaluation.erp.ensure_parser_erp_calibration",
        calibrate)
    cache = sweep.ParserCache()
    monkeypatch.setattr(sweep, "erp_fast_calibration_enabled", lambda: True)
    fast = cache.get("TWO_WORD", calibrate=True)
    monkeypatch.setattr(sweep, "erp_fast_calibration_enabled", lambda: False)
    full = cache.get("TWO_WORD", calibrate=True)
    raw = cache.get("TWO_WORD", calibrate=False)
    assert fast.threshold_mode is True and full.threshold_mode is False
    assert not hasattr(raw, "threshold_mode")
    assert len(harness) == 1 and modes == [True, False]


def test_disk_entry_with_wrong_identity_is_not_reused(monkeypatch, harness):
    from neural_assemblies.assembly_calculus.emergent.evaluation.checkpoint import ParserCheckpoint
    wrong = ParserCheckpoint(parser=SimpleNamespace(request={"wrong": True}),
                             depth="TWO_WORD", seed=42, n=3000, k=30,
                             holdout_words=frozenset(), meta={"cache_identity": "other"})
    monkeypatch.setattr(
        "neural_assemblies.assembly_calculus.emergent.evaluation.checkpoint.load_backbone_cache",
        lambda path: wrong)
    result = sweep.ParserCache().get("TWO_WORD")
    assert len(harness) == 1 and "wrong" not in result.request


def test_trainer_honors_empty_holdouts_and_explicit_engine(monkeypatch):
    observed = {}
    class Parser:
        def __init__(self, **kwargs):
            observed.update(kwargs)
        def train(self, **kwargs):
            observed.update(kwargs)
    monkeypatch.setattr(
        "neural_assemblies.assembly_calculus.emergent.parser.EmergentParser", Parser)
    monkeypatch.setattr(generalization, "_finish_parser", lambda parser: parser)
    generalization.train_parser_to_depth("FULL_TRAIN", vocabulary={},
                                        holdout_words=set(), engine="numpy_exact")
    assert observed["holdout_words"] == set()
    assert observed["engine"] == "numpy_exact"


def test_dialogue_training_honors_empty_holdouts(monkeypatch):
    observed = []
    monkeypatch.setattr(
        "neural_assemblies.assembly_calculus.emergent.evaluation.parity.run_dialogue_stage",
        lambda parser, schedule: observed.append(schedule.holdout_words))
    generalization.train_dialogue_with_holdouts(None, SimpleNamespace(), set())
    assert observed == [set()]


def test_environment_identity_hashes_exact_values(monkeypatch):
    monkeypatch.setenv("ASSEMBLIES_TEST_OPTION", "CaseSensitiveValue")
    first = sweep._training_env_signature()
    assert "CaseSensitiveValue" not in repr(first)
    assert len(dict(first)["ASSEMBLIES_TEST_OPTION"]) == 64
    monkeypatch.setenv("ASSEMBLIES_TEST_OPTION", "casesensitivevalue")
    assert sweep._training_env_signature() != first

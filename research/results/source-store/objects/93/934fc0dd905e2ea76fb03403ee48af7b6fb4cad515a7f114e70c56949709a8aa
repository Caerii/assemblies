"""Sweep forks must isolate parser metadata as well as neural weights."""
import copy
from types import SimpleNamespace

import pytest

from neural_assemblies.assembly_calculus.emergent import EmergentParser
from neural_assemblies.assembly_calculus.emergent.evaluation.checkpoint import fork_parser_instance
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (
    ParserCache, ParserCacheEntry, _pristine_copy,
)


@pytest.fixture
def parser():
    parser = EmergentParser(n=300, k=8, seed=23, engine="numpy_sparse", fast_training=True)
    parser._bootstrap_categories = {"known": "NOUN"}
    parser._dist_categories = parser._bootstrap_categories
    parser._func_subcategories = {"the": "DET"}
    parser._exposure_log = [["the", "dog"]]
    return parser


@pytest.mark.parametrize("wobbly", [False, True])
@pytest.mark.parametrize("field", [
    "core_lexicons", "role_lexicons", "vp_assemblies", "_bootstrap_categories",
    "_func_subcategories", "_exposure_log",
])
def test_fork_metadata_changes_cannot_reach_source_or_sibling(parser, field, wobbly):
    before = copy.deepcopy(getattr(parser, field))
    fork = fork_parser_instance(parser, wobbly=wobbly)
    sibling = fork_parser_instance(parser, wobbly=wobbly)
    value = getattr(fork, field)
    if isinstance(value, list):
        value[0].append("modified")
    else:
        value["fork-only"] = "modified"
    assert getattr(parser, field) == before
    assert getattr(sibling, field) == before


def test_fork_retains_internal_metadata_alias_without_sharing_parent(parser):
    fork = fork_parser_instance(parser)
    assert fork._dist_categories is fork._bootstrap_categories
    assert fork._dist_categories is not parser._dist_categories


@pytest.mark.parametrize("wobbly", [False, True])
def test_fork_records_wobbly_provenance(parser, wobbly):
    fork = fork_parser_instance(parser, wobbly=wobbly)
    assert fork._wobbly_fork is wobbly


class Uncopyable:
    def __deepcopy__(self, memo):
        raise TypeError("cannot copy this state")


def test_snapshot_failure_is_not_silently_converted_to_live_parser_fallback():
    with pytest.raises(RuntimeError, match="pristine"):
        _pristine_copy(Uncopyable())


def test_fork_refuses_cache_entry_without_pristine_snapshot(monkeypatch, parser):
    cache = ParserCache()
    cache._entries["fixture"] = ParserCacheEntry(parser=parser, train_seconds=0)
    monkeypatch.setattr(cache, "get", lambda *args, **kwargs: parser)
    with pytest.raises(RuntimeError, match="pristine"):
        cache.fork("SENTENCES")


def test_calibration_uses_pristine_state_and_publishes_matching_snapshots(monkeypatch):
    live = SimpleNamespace(value=1)
    entry = ParserCacheEntry(parser=live, pristine=copy.deepcopy(live), train_seconds=0)
    live.value = 99  # Someone previously mutated the shared public object.
    observed = []

    def calibrate(parser, **kwargs):
        observed.append(parser.value)
        parser.thresholds = {"reference": parser.value}

    monkeypatch.setattr(
        "neural_assemblies.assembly_calculus.emergent.evaluation.erp.ensure_parser_erp_calibration",
        calibrate)
    ParserCache()._calibrate(entry)
    assert observed == [1]
    assert entry.calibrated
    assert entry.parser.thresholds == entry.pristine.thresholds == {"reference": 1}
    assert entry.parser.thresholds is not entry.pristine.thresholds
    assert live.value == 99 and not hasattr(live, "thresholds")


@pytest.mark.parametrize("failure", ["calibration", "snapshot"])
def test_failed_calibration_does_not_publish_partial_state(monkeypatch, failure):
    live, pristine = SimpleNamespace(value=1), SimpleNamespace(value=1)
    entry = ParserCacheEntry(parser=live, pristine=pristine, train_seconds=0)

    def calibrate(parser, **kwargs):
        parser.value = 2
        if failure == "calibration":
            raise RuntimeError("failed calibration")
        parser.uncopyable = Uncopyable()

    monkeypatch.setattr(
        "neural_assemblies.assembly_calculus.emergent.evaluation.erp.ensure_parser_erp_calibration",
        calibrate)
    with pytest.raises(RuntimeError):
        ParserCache()._calibrate(entry)
    assert entry.parser is live and entry.pristine is pristine
    assert live.value == pristine.value == 1
    assert not entry.calibrated


def test_first_calibrated_get_returns_the_published_parser(monkeypatch, parser):
    monkeypatch.setattr(
        "neural_assemblies.assembly_calculus.emergent.evaluation.sweep._backbone_disk_path",
        lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "neural_assemblies.assembly_calculus.emergent.evaluation.generalization.train_parser_to_depth",
        lambda *args, **kwargs: parser)
    monkeypatch.setattr(
        "neural_assemblies.assembly_calculus.emergent.evaluation.erp.ensure_parser_erp_calibration",
        lambda target, **kwargs: setattr(target, "calibration_marker", "ready"))
    cache = ParserCache()
    result = cache.get("TWO_WORD", calibrate=True)
    entry = next(entry for entry in cache._entries.values() if entry.calibrated)
    assert result is entry.parser
    assert result.calibration_marker == entry.pristine.calibration_marker == "ready"
    assert not hasattr(parser, "calibration_marker")

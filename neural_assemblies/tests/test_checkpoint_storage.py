"""A failed or concurrent cache save cannot publish a partial checkpoint."""
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from neural_assemblies.assembly_calculus.emergent.evaluation.checkpoint import (
    ParserCheckpoint, load_backbone_cache, save_backbone_cache,
)


def checkpoint(seed):
    return ParserCheckpoint(parser={"seed": seed}, depth="fixture", seed=seed,
                            n=100, k=10, holdout_words=frozenset())


@pytest.mark.parametrize("contents", [b"", b"\x80\x05", b"\x80\xff"])
def test_incomplete_or_unsupported_pickle_is_a_cache_miss(tmp_path, contents):
    path = tmp_path / "checkpoint.pkl"
    path.write_bytes(contents)
    assert load_backbone_cache(path) is None


def test_failed_serialization_preserves_previous_checkpoint_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pkl"
    save_backbone_cache(checkpoint(1), path)
    before = path.read_bytes()

    def fail(obj, stream, **kwargs):
        stream.write(b"partial")
        raise ValueError("serialization failed")

    monkeypatch.setattr(pickle, "dump", fail)
    with pytest.raises(ValueError, match="serialization failed"):
        save_backbone_cache(checkpoint(2), path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_simultaneous_saves_use_independent_temporary_files(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pkl"
    barrier = Barrier(2)
    dump = pickle.dump
    temporary_paths = []

    def synchronized_dump(obj, stream, **kwargs):
        temporary_paths.append(stream.name)
        # Both writers have opened their temporary file before either writes.
        barrier.wait(timeout=10)
        dump(obj, stream, **kwargs)

    monkeypatch.setattr(pickle, "dump", synchronized_dump)
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [executor.submit(save_backbone_cache, checkpoint(seed), path) for seed in (1, 2)]
        for job in jobs:
            job.result(timeout=15)
    assert len(set(temporary_paths)) == 2
    loaded = load_backbone_cache(path)
    assert loaded.seed in (1, 2)
    assert loaded.parser == {"seed": loaded.seed}
    assert list(tmp_path.iterdir()) == [path]


def test_replacement_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pkl"
    save_backbone_cache(checkpoint(1), path)
    before = path.read_bytes()

    def fail(*args):
        raise PermissionError("replace denied")

    monkeypatch.setattr(type(path), "replace", fail)
    with pytest.raises(PermissionError, match="replace denied"):
        save_backbone_cache(checkpoint(2), path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_transient_windows_replacement_contention_is_retried(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pkl"
    replace = type(path).replace
    calls = []

    def busy_once(source, target):
        calls.append(source)
        if len(calls) == 1:
            error = PermissionError("simulated Windows sharing contention")
            error.winerror = 32
            raise error
        return replace(source, target)

    monkeypatch.setattr(type(path), "replace", busy_once)
    save_backbone_cache(checkpoint(2), path)
    assert len(calls) == 2
    assert load_backbone_cache(path).seed == 2
    assert list(tmp_path.iterdir()) == [path]


def test_persistent_windows_denial_is_bounded_and_preserves_old_file(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pkl"
    save_backbone_cache(checkpoint(1), path)
    calls = []

    def denied(*args):
        calls.append(None)
        error = PermissionError("persistent denial")
        error.winerror = 5
        raise error

    monkeypatch.setattr(type(path), "replace", denied)
    with pytest.raises(PermissionError, match="persistent denial"):
        save_backbone_cache(checkpoint(2), path)
    assert len(calls) == 6
    assert load_backbone_cache(path).seed == 1
    assert list(tmp_path.iterdir()) == [path]

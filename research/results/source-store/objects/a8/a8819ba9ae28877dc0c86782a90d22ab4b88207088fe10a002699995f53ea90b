"""research.collect: a finished run comes from its pinned worktree whole,
verified, and only verified copies leave the worktree."""
import hashlib
import json
import subprocess

import pytest

from research import collect as C

RUN = "research/results/runs/memory.example/tag-1"


def _object(root, data):
    digest = hashlib.sha256(data).hexdigest()
    path = root / C.STORE / digest[:2] / digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return digest


def _worktree(tmp_path):
    wt = tmp_path / "wt"
    (wt / RUN).mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(wt)], check=True)
    digests = [_object(wt, b"script source"), _object(wt, b"registration")]
    (wt / RUN / "results.json").write_text('{"observations": 1}')
    (wt / RUN / "source.manifest.json").write_text(json.dumps({
        "format": "assemblies-source-manifest/1", "archive_sha256": "0" * 64,
        "rebuild": {}, "members": [["a.py", digests[0]], ["b.md", digests[1]]]}))
    (wt / "run.log").write_text("wrote it\n")
    return wt, digests


def test_collect_copies_the_run_its_objects_and_log_then_cleans(tmp_path):
    wt, digests = _worktree(tmp_path)
    root = tmp_path / "main"
    (root / C.LOGS).mkdir(parents=True)
    seen = []
    report = C.collect(wt, [RUN], [("run.log", "tag-1.log")], root=root, clean=True,
                       validate=lambda p: seen.append(p) or [])
    assert (root / RUN / "results.json").read_text() == '{"observations": 1}'
    assert sorted(report["objects"]) == sorted(digests)
    assert (root / C.LOGS / "tag-1.log").read_text() == "wrote it\n"
    assert seen == [root / RUN / "results.json"]
    assert not (wt / RUN).exists()
    assert all(not (wt / C.STORE / d[:2] / d).exists() for d in digests)


def test_collect_refuses_a_different_copy_and_a_corrupt_object(tmp_path):
    wt, digests = _worktree(tmp_path)
    root = tmp_path / "main"
    (root / RUN).mkdir(parents=True)
    (root / RUN / "results.json").write_text('{"observations": 2}')
    with pytest.raises(FileExistsError):
        C.collect(wt, [RUN], root=root, validate=lambda p: [])
    root2 = tmp_path / "main2"
    (wt / C.STORE / digests[0][:2] / digests[0]).write_bytes(b"tampered")
    with pytest.raises(ValueError):
        C.collect(wt, [RUN], root=root2, validate=lambda p: [])


def test_an_invalid_record_is_reported_and_nothing_is_cleaned(tmp_path):
    wt, _ = _worktree(tmp_path)
    root = tmp_path / "main"
    report = C.collect(wt, [RUN], root=root, clean=True, validate=lambda p: ["bad"])
    assert report["errors"] == {RUN: ["bad"]}
    assert report["cleaned"] == [] and (wt / RUN).exists()

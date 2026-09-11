"""The legacy results adapter must preserve evidence and JSON semantics."""

import pytest

from research.experiments import _results


def test_write_result_is_exclusive_and_returns_path(tmp_path, monkeypatch):
    monkeypatch.setattr(_results, "RESULTS", str(tmp_path))

    path = _results.write_result("line", "run.json", {"value": 1})
    assert path.read_text(encoding="utf-8").endswith('"value": 1\n}\n')
    with pytest.raises(FileExistsError):
        _results.write_result("line", "run.json", {"value": 2})
    assert '"value": 1' in path.read_text(encoding="utf-8")


def test_write_result_rejects_nonfinite_json_before_creating_file(tmp_path, monkeypatch):
    monkeypatch.setattr(_results, "RESULTS", str(tmp_path))

    with pytest.raises(ValueError, match="not JSON compliant"):
        _results.write_result("line", "bad.json", {"value": float("nan")})
    assert not (tmp_path / "line" / "bad.json").exists()

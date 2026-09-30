"""Operational benchmark output has explicit, reproducible semantics."""

import json

import pytest

from neural_assemblies.benchmarks.throughput import (
    _materialization_storage,
    benchmark,
    main,
)


def test_throughput_selects_backend_supported_materialization():
    assert _materialization_storage("numpy_exact") == "dense"
    assert _materialization_storage("numpy_sparse") == "dense"
    assert _materialization_storage("torch_sparse") == "csr"


def test_throughput_reports_per_seed_quantiles():
    result = benchmark(engine="numpy_exact", sizes=[40], k=5, rounds=1,
                       seeds=[1, 2, 3])
    cell = result["cells"][0]
    assert result["status"] == "diagnostic"
    assert "git_commit" in result["runtime"]
    assert result["storage"] == "materialized"
    assert result["materialization_storage"] == "dense"
    assert result["warmup"] is True
    assert result["model_semantics"]["connectome"] == "fixed-hash-regenerated"
    assert len(cell["per_seed"]) == 3
    assert set(cell["rounds_per_second"]) == {"min", "median", "p90", "max"}
    assert cell["rounds_per_second"]["median"] <= cell["rounds_per_second"]["p90"] <= cell["rounds_per_second"]["max"]


def test_throughput_rejects_underpowered_seed_set():
    with pytest.raises(ValueError, match="at least three"):
        benchmark(engine="numpy_exact", sizes=[40], k=5, rounds=1, seeds=[1, 2])


def test_throughput_output_is_exclusive(tmp_path):
    output = tmp_path / "throughput.json"
    assert main(["--engine", "numpy_exact", "--sizes", "40", "--k", "5",
                 "--rounds", "1", "--seeds", "1", "2", "3",
                 "--output", str(output)]) == 0
    with pytest.raises(FileExistsError):
        main(["--engine", "numpy_exact", "--sizes", "40", "--k", "5",
              "--rounds", "1", "--seeds", "1", "2", "3",
              "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8"))["benchmark"] == "projection-throughput-v1"

"""Operational benchmark output has explicit, reproducible semantics."""

import json

import pytest

from neural_assemblies.benchmarks.throughput import benchmark, main


def test_throughput_reports_per_seed_quantiles():
    result = benchmark(engine="numpy_exact", sizes=[40], k=5, rounds=1,
                       seeds=[1, 2, 3])
    cell = result["cells"][0]
    assert result["status"] == "diagnostic"
    assert result["model_semantics"]["connectome"] == "fixed-hash-regenerated"
    assert len(cell["per_seed"]) == 3
    assert set(cell["rounds_per_second"]) == {"min", "median", "p90", "max"}


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

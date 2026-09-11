"""Negative controls for the legacy quick-suite reporting contract."""
import pytest

from research.experiments.base import ExperimentResult
from research.experiments.run_all_experiments import generate_summary, print_summary


def test_empty_suite_cannot_report_success():
    with pytest.raises(ValueError, match="empty"):
        generate_summary({})


def test_perfect_metrics_cannot_promote_smoke_to_scientific_pass(capsys):
    result = ExperimentResult("noise", metrics={"recovery": 1.0})
    summary = generate_summary({"noise_robustness": result})
    assert summary["scientific_status"] == "VOID"
    assert summary["execution_success"] is True
    assert summary["experiments"]["noise_robustness"]["metrics"] == {"recovery": 1.0}
    assert "max_recoverable_noise" not in summary["experiments"]["noise_robustness"]
    print_summary(summary)
    printed = capsys.readouterr().out
    assert "VOID" in printed
    assert "PASS" not in printed


def test_failed_and_unrecognized_experiments_remain_visible(capsys):
    failed = ExperimentResult("phase", success=False, error_message="no observations")
    unknown = ExperimentResult("new", parameters={"seed": 7}, metrics={})
    summary = generate_summary({"phase_diagram": failed, "new_protocol": unknown})
    assert summary["execution_success"] is False
    assert set(summary["experiments"]) == {"phase_diagram", "new_protocol"}
    assert summary["experiments"]["new_protocol"]["parameters"] == {"seed": 7}
    assert summary["experiments"]["new_protocol"]["metrics"] == {}
    print_summary(summary)
    assert "no observations" in capsys.readouterr().out


def test_summary_is_detached_from_mutable_result():
    result = ExperimentResult("noise", metrics={"values": [0.5]})
    summary = generate_summary({"noise": result})
    result.metrics["values"].append(1.0)
    assert summary["experiments"]["noise"]["metrics"] == {"values": [0.5]}


def test_full_request_never_falls_back_to_quick(monkeypatch, capsys):
    from research.experiments import run_all_experiments as suite

    def forbidden():
        pytest.fail("unsupported full mode launched quick experiments")

    monkeypatch.setattr(suite, "run_quick_suite", forbidden)
    with pytest.raises(SystemExit) as error:
        suite.main(["--full"])
    assert error.value.code == 2
    assert "full suite is not implemented" in capsys.readouterr().err


@pytest.mark.parametrize("index", range(8))
def test_unknown_parameter_refused_before_experiment_body(index):
    from research.experiments.run_all_experiments import QUICK_EXPERIMENTS

    _, factory, _ = QUICK_EXPERIMENTS[index]
    # No constructor, filesystem writes or usable engine: entering the body fails
    # differently, so this checks the actual Python API boundary, not introspection.
    uninitialized = object.__new__(factory)
    with pytest.raises(TypeError, match="unexpected keyword argument 'unconsumed_parameter'"):
        uninitialized.run(unconsumed_parameter=1)


def test_old_suite_reports_all_six_mismatches_without_constructing(monkeypatch):
    from research.experiments import run_all_experiments as suite

    def forbidden(*args, **kwargs):
        pytest.fail("invalid suite constructed an experiment")

    for _, factory, _ in suite.QUICK_EXPERIMENTS:
        monkeypatch.setattr(factory, "__init__", forbidden)
    with pytest.raises(ValueError) as error:
        suite.run_quick_suite()
    message = str(error.value)
    for name in ("projection", "association", "merge", "phase_diagram", "scaling_laws", "noise_robustness"):
        assert f"{name}: unsupported parameters" in message
    assert "coding_capacity:" not in message
    assert "biological:" not in message


def test_valid_suite_configuration_does_not_execute_or_mutate():
    from research.experiments.run_all_experiments import validate_suite

    class Probe:
        def run(self, *, size):
            pytest.fail("preflight executed an experiment")

    parameters = {"size": 7}
    validate_suite((("probe", Probe, parameters),))
    assert parameters == {"size": 7}
    with pytest.raises(ValueError, match="missing a required.*argument"):
        validate_suite((("probe", Probe, {}),))
    with pytest.raises(ValueError, match="duplicate experiment name"):
        validate_suite((("probe", Probe, parameters), ("probe", Probe, parameters)))
    with pytest.raises(ValueError, match="empty"):
        validate_suite(())

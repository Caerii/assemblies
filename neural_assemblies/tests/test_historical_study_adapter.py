"""Historical protocol identity and shared-dispatch controls."""
from dataclasses import replace
from pathlib import Path

import pytest
from research.experiments._historical import HistoricalStudy
from research.experiments.base import ExperimentResult


def specification(factory):
    return HistoricalStudy("memory.fixture", "1", "registration.md", Path("fixture.py"),
                           factory, lambda smoke: {"size": 3 if smoke else 10}, "fixture scope")


def record(**changes):
    return {"protocol": "memory.fixture", "protocol_version": "1", "engine": "numpy_explicit",
            "mode": "smoke", "tag": "fixture", "seeds": [9,2,7], "parameters": {"size": 3}, **changes}


@pytest.mark.parametrize("changes", [
    {"protocol": "memory.other"}, {"protocol_version": "2"},
    {"engine": "numpy_sparse"}, {"mode": "smkoe"},
])
def test_mismatched_record_fails_before_producer_construction(changes):
    def forbidden(**kwargs):
        pytest.fail("mismatched record constructed a producer")
    with pytest.raises(ValueError):
        specification(forbidden).measure(record(**changes))


@pytest.mark.parametrize("mode,verdict", [("smoke","VOID"),("study","UNADOPTED")])
def test_common_adapter_preserves_inputs_scope_and_execution_failure(mode, verdict):
    calls = []
    class Producer:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def run(self, **kwargs):
            calls.append(kwargs)
            return ExperimentResult("fixture", success=False, error_message="failed computation")
    output = specification(Producer).measure(record(mode=mode))
    assert calls[0]["seed"] == 0 and calls[0]["verbose"] is False
    assert calls[0]["results_dir"].parts[-2:] == ("memory.fixture", "fixture")
    assert calls[1] == {"seed_ids": [9,2,7], "size": 3}
    assert output["scope"] == "fixture scope" and output["verdict"] == verdict
    assert output["result"]["success"] is False


def test_cli_uses_specified_protocol_source_registration_and_seed_defaults():
    calls = []
    spec = replace(specification(lambda **kwargs: None), default_seeds=(5,6,7))
    spec.main(["--tag", "fixture", "--quick"], writer=lambda **kwargs: calls.append(kwargs) or "saved")
    sent = calls[0]
    assert sent["script"] == Path("fixture.py")
    assert sent["registration"] == "registration.md"
    assert sent["protocol"] == "memory.fixture" and sent["protocol_version"] == "1"
    assert sent["seeds"] == [5,6,7] and sent["parameters"] == {"size": 3}
    assert sent["smoke"] is True and sent["measure"] == spec.measure

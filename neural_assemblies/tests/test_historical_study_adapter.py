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


@pytest.mark.parametrize('document', [
    '[]', '{"size": 2, "size": 3}', '{"seed_ids": [1, 2, 3]}',
    '{"engine": "numpy_sparse"}', '{"tag": "replacement"}', '{"n_seeds": 3}',
    '{"typo": 7}', '{"size": NaN}', '{"size": 1e999}', '{"size": 1e-999}',
])
def test_parameter_file_errors_stop_before_writer(tmp_path, monkeypatch, document):
    from research.experiments import _historical
    monkeypatch.setattr(_historical, 'ROOT', tmp_path)
    (tmp_path / 'parameters.json').write_text(document)
    with pytest.raises(SystemExit) as exc:
        specification(None).main(['--tag', 'fixture', '--parameters', 'parameters.json'],
                                 writer=lambda **kwargs: pytest.fail('invalid file reached writer'))
    assert exc.value.code == 2


@pytest.mark.parametrize('smoke', [False, True])
def test_parameter_file_replaces_only_named_defaults_and_binds_exact_bytes(tmp_path, monkeypatch, smoke):
    import hashlib
    from research.experiments import _historical
    monkeypatch.setattr(_historical, 'ROOT', tmp_path)
    data = b'{"size": 7}\r\n'
    (tmp_path / 'parameters.json').write_bytes(data)
    defaults = {'size': 3 if smoke else 10, 'rounds': 5}
    spec = replace(specification(None), parameters=lambda mode: defaults)
    calls = []
    spec.main(['--tag', 'fixture', '--parameters', './parameters.json'] + (['--smoke'] if smoke else []),
              writer=lambda **kwargs: calls.append(kwargs))
    sent = calls[0]
    assert sent['parameters'] == {'size': 7, 'rounds': 5}
    assert defaults == {'size': 3 if smoke else 10, 'rounds': 5}
    assert sent['input_artifacts'] == ('parameters.json',)
    assert sent['expected_input_digests'] == {'parameters.json': hashlib.sha256(data).hexdigest()}
    assert sent['smoke'] is smoke


@pytest.mark.parametrize('name', ['missing.json', '../outside.json'])
def test_parameter_file_must_exist_in_repository(tmp_path, monkeypatch, name):
    from research.experiments import _historical
    monkeypatch.setattr(_historical, 'ROOT', tmp_path)
    (tmp_path.parent / 'outside.json').write_text('{}')
    with pytest.raises(SystemExit):
        specification(None).main(['--tag', 'fixture', '--parameters', name],
                                 writer=lambda **kwargs: pytest.fail('invalid path reached writer'))

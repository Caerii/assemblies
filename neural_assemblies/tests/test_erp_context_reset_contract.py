"""The ERP runner executes and records the selected context reset semantics."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from neural_assemblies.assembly_calculus.emergent.evaluation.erp.protocol import ErpProtocol
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.runner import run_incremental_erp_probes


@pytest.mark.parametrize("mode", ["construction", "activity"])
def test_runner_dispatches_and_records_reset(mode):
    calls = []
    parser = SimpleNamespace(
        brain=SimpleNamespace(frozen=nullcontext),
        _reset_context_state=lambda: calls.append("construction"),
        _reset_context_winners=lambda *, preserve_mapping: calls.append(("activity", preserve_mapping)),
        _get_incremental_circuit=lambda *, reset: None,
    )
    result, probes = run_incremental_erp_probes(
        parser, ["word"], protocol=ErpProtocol(context_reset=mode),
        readiness=object(), baseline=object(), thresholds=object(),
        stop_at_position=-1, finalize_parse=False,
    )
    assert calls == (["construction"] if mode == "construction" else [("activity", True)])
    assert result["erp_protocol"]["context_reset"] == mode
    assert not probes


@pytest.mark.parametrize("mode", ["unknown", None, True, 1])
def test_reset_choice_rejects_invalid_values(mode):
    with pytest.raises(ValueError, match="context_reset"):
        ErpProtocol(context_reset=mode)


def test_reset_choice_is_named_in_description_and_derived_explicitly():
    original = ErpProtocol()
    derived = original.with_(context_reset="activity")
    assert original.context_reset == "construction"
    assert derived.context_reset == "activity"
    assert "context_reset=activity" in derived.describe()

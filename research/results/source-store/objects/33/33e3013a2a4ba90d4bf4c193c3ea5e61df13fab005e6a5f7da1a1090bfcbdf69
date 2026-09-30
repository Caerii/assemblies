"""Controls for the registered per-fiber plasticity measurement."""
from copy import deepcopy

import pytest

from research.experiments.per_fiber_plasticity import judge, parameters, run_seed


def test_copied_fibers_follow_their_rates_without_clipping():
    config = parameters()
    row, raw = run_seed(201, config)

    assert all(judge([row], config).values())
    assert row["cells"]["forward"]["initial_block_sha256"] == row["cells"]["swapped"]["initial_block_sha256"]
    assert raw["cells"]["equal"]["final_blocks"]["A"] == raw["cells"]["equal"]["final_blocks"]["B"]


@pytest.mark.parametrize("fault,check", [
    ("equal", "equal-null"),
    ("forward", "forward-ratio"),
    ("history", "winner-history-symmetry"),
    ("clip", "unclipped"),
])
def test_constructed_failures_are_detected(fault, check):
    config = parameters()
    row, _ = run_seed(201, config)
    broken = deepcopy(row)
    if fault == "equal":
        broken["cells"]["equal"]["ratio"] = 2
    elif fault == "forward":
        broken["cells"]["forward"]["ratio"] = 1
    elif fault == "history":
        broken["histories_equal"] = False
    else:
        broken["cells"]["forward"]["maximum_selected_weight"] = config["w_max"]

    assert judge([row], config)[check]
    assert not judge([broken], config)[check]

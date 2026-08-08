"""The P600's violation arm probes an area that does not exist. PARTLY MOVED.

UPDATE (#128), and it is NOT a resolution -- I claimed one and the full suite
refuted it within the hour.

``train_phrases`` picks its subject and verb by ``role == "agent"`` /
``"action"``, and every curriculum sentence carried ``roles=[None]``, so phrase
training HAD NEVER RUN: VP existed only as bootstrap pre-growth plus whatever
calibration recruited. Routing scene-derived roles through the pipeline wakes
it -- 72 VP merges during training, self-fiber extent 2061 -- and the probe
then reads 0.001172 where it read exactly 0.000000 before.

BUT ONLY IN ISOLATION. Run after the sibling ERP tests, the same probe reads
0.000000 again with 1920 self-fiber columns and 30 winners present. Fiber
exists, winners exist, drive is zero -- the ORIGINAL diagnosis's exact
signature. So waking phrase training was NECESSARY and is NOT SUFFICIENT: the
liveness of this arm depends on what ran before it, which is
[[erp-suite-cross-test-leakage]] and means the self-block/recruitment defect
below is still open (#32, #104).

The marker is therefore non-strict: the outcome genuinely varies with test
order, and a strict xfail would flip-flop rather than report anything.

#104 / #32. Measured on a PRISTINE fork, before any parse:

    area           self-fiber extent   materialized
    ROLE_PATIENT   960                 675
    VP             0                   0
    NP             0                   0

and `_self_recurrent_energy(brain, "VP")` returns **exactly 0.000000** -- total
0.0 over 0 candidates -- while VP holds 30 winners. `p600 = 1 - energy`, so the
VP arm reads a constant 1.0 no matter what the sentence was.

THIS IS THE ROOT CAUSE UNDER THE OTHER TWO. Today established that the P600
scale tracks `area.w` (a lazy-materialisation artifact) and that no divisor
fixes it because the candidate pool is too small to express concentration. Both
are real. Neither matters as much as an arm of the contrast being STRUCTURALLY
EMPTY: a fiber with zero columns carries zero drive and the k-WTA still returns
k winners, so nothing raises and the number looks like a measurement
([[silent-no-op-dead-fibers]]).

It also explains the saturation directly. The violation arm sits pinned near
1.0 because its energy is ZERO, not because the metric is compressed.

And it is the missing half of [[erp-p600-sign-inverted]], which recorded that
"the gram/violation arms measure DIFFERENT AREAS (ROLE_PATIENT vs VP)" and that
"the documented untrained-pathway mechanism never runs". This is why it never
ran.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

from neural_assemblies.assembly_calculus.emergent.core.areas import (
    ROLE_PATIENT, VP,
)
from neural_assemblies.assembly_calculus.emergent.evaluation import (
    calibrate_erp_thresholds,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.adapters import (
    MIN_POOL_RATIO, _self_recurrent_energy,
)


@pytest.fixture
def parsed(forked_parser):
    """A parser that has actually PARSED something.

    `_self_recurrent_energy` early-returns 0.0 when the area has no winners, so
    on a pristine fork every probe reads zero and every assertion below would
    pass or fail for the wrong reason. Calibration is what the shipped ERP path
    does before reading, so it is what these tests must do too.
    """
    parser = forked_parser("SENTENCES", seed=11)
    calibrate_erp_thresholds(parser)
    return parser.brain


class TestTheProbedAreasAreReal:

    def test_the_control_arm_is_alive(self, parsed):
        """ROLE_PATIENT: a real fiber, real candidates, nonzero energy.

        The positive control. Without it, every assertion below could be
        satisfied by a parser that simply builds nothing.
        """
        b = parsed
        eng = b._engine_for(b.areas[ROLE_PATIENT])
        assert (eng.fiber_extent(ROLE_PATIENT, ROLE_PATIENT) or 0) > 0
        assert _self_recurrent_energy(b, ROLE_PATIENT) > 0.0

    @pytest.mark.xfail(strict=False, reason=(
        "ORDER-DEPENDENT, which is itself the finding. Waking `train_phrases` "
        "(it had never run -- every curriculum sentence carried roles=[None]) "
        "gives VP 72 merges and self-fiber extent 2061, and the probe reads "
        "0.001172 IN ISOLATION against 0.000000 before. Run after the sibling "
        "ERP tests it reads 0.000000 again, with 1920 columns and 30 winners "
        "-- fiber exists, winners exist, drive zero, the original signature. "
        "So phrase training was NECESSARY and is NOT SUFFICIENT, and the "
        "self-block/recruitment defect (#32, #104) is still open. Non-strict "
        "because the outcome varies with test order; a strict marker would "
        "flip-flop instead of reporting. ROLE_PATIENT is the live control at "
        "0.010442 and passes in both orders."))
    def test_the_violation_arm_is_alive_too(self, parsed):
        b = parsed
        eng = b._engine_for(b.areas[VP])
        assert (eng.fiber_extent(VP, VP) or 0) > 0, (
            f"VP self-fiber has {eng.fiber_extent(VP, VP)} columns and "
            f"{eng.materialized_count(VP)} materialised neurons")
        assert len(b.areas[VP].winners) > 0, "VP never fired at all"
        assert _self_recurrent_energy(b, VP) > 0.0, (
            f"VP has {eng.fiber_extent(VP, VP)} self-fiber columns and "
            f"{len(b.areas[VP].winners)} winners yet delivers ZERO drive -- "
            f"the firing assembly's rows in the self-block are unwired")


class TestPoolRatioIsReported:
    """A number read from a pool barely larger than k is not a measurement."""

    def test_the_probe_records_its_pool_ratio(self, parsed):
        b = parsed
        b.last_erp_pool_ratio = {}
        _self_recurrent_energy(b, ROLE_PATIENT)
        ratios = getattr(b, "last_erp_pool_ratio", {})
        assert ROLE_PATIENT in ratios, (
            "the probe reported no pool ratio, so a caller cannot tell a "
            "measurement from a starved one")

    def test_the_control_arm_clears_the_bar(self, parsed):
        """Measured 22.60 against a bar of 2.0.

        Asserted as a RELATION to MIN_POOL_RATIO rather than pinning 22.6,
        which is a realization that moves with training.
        """
        b = parsed
        b.last_erp_pool_ratio = {}
        _self_recurrent_energy(b, ROLE_PATIENT)
        ratio = b.last_erp_pool_ratio[ROLE_PATIENT]
        assert ratio >= MIN_POOL_RATIO, (
            f"pool/k = {ratio:.2f} is below {MIN_POOL_RATIO}: most of the "
            f"candidate pool IS the assembly, so concentration cannot be read")

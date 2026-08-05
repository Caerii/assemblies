"""The three quantities called "p600" must stay distinguishable.

WHY THIS FILE EXISTS. `sample.p600` (raw deficit), `sample.p600_excess`
(clipped over baseline) and `separation["p600_auc"]` are three DIFFERENT
quantities sharing one word. On 2026-08-05 two of them were compared as though
interchangeable and the apparent contradiction was chased for hours -- the
premise "the AUC ranks the excess" was simply false, and re-reading
`calibration.py` (which passes `catv_p600_raw`) settled it in a minute.

The pinning test here is `test_auc_of_raw_matches_the_separation_key`: it fails
the moment `separation["p600_auc"]` stops ranking the raw deficit. That is the
exact wrong belief that cost the time, so it is now a test rather than a
docstring.
"""
from __future__ import annotations

import math

import pytest

from neural_assemblies.assembly_calculus.emergent.evaluation.erp.calibration import (
    ErpCalibrationReport,
    ErpQuantities,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.frames import (
    PositionErpSample,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.gates import (
    ErpBaseline,
    ErpReadiness,
    ErpThresholds,
)


def _sample(label, p600, excess):
    return PositionErpSample(
        label=label, sentence=("x",), position=0, word="x", category="NOUN",
        n400=0.0, p600=p600, phrase_stability=0.0,
        n400_excess=0.0, p600_excess=excess,
    )


def _report(samples):
    return ErpCalibrationReport(
        readiness=ErpReadiness(n400_ready=True, p600_ready=True),
        baseline=ErpBaseline(),
        thresholds=ErpThresholds(), samples=samples,
    )


class TestTheReaderKeepsThemApart:

    def test_raw_and_excess_are_reported_separately(self):
        """A caller must never have to guess which one they were handed."""
        rep = _report([
            _sample("grammatical", 0.90, 0.00),
            _sample("category_violation", 0.95, 0.05),
        ])
        q = rep.p600_quantities()
        assert q.deficit_raw_grammatical == [0.90]
        assert q.deficit_raw_violation == [0.95]
        assert q.excess_over_baseline_grammatical == [0.00]
        assert q.excess_over_baseline_violation == [0.05]

    def test_auc_is_computed_on_raw_not_on_excess(self):
        """The distinguishing case: the two rank the arms OPPOSITE ways.

        Constructed so raw says violation > grammatical (AUC 1.0) while the
        excess says the reverse. If `auc_of_raw` ever silently switched to the
        excess it would read 0.0 here, and no amount of prose would have caught
        that.
        """
        rep = _report([
            _sample("grammatical", 0.10, 0.90),
            _sample("category_violation", 0.90, 0.10),
        ])
        q = rep.p600_quantities()
        assert q.auc_of_raw == 1.0, (
            "auc_of_raw must rank the RAW deficit; ranking the excess would "
            "give 0.0 on this input"
        )

    def test_granularity_is_reported_so_small_moves_are_not_over_read(self):
        """3x3 frames give AUC steps of 1/9; differences under that are noise."""
        rep = _report(
            [_sample("grammatical", 0.1 * i, 0.0) for i in range(3)]
            + [_sample("category_violation", 0.5 + 0.1 * i, 0.0) for i in range(3)]
        )
        q = rep.p600_quantities()
        assert q.n == 9
        assert math.isclose(1.0 / q.n, 0.1111, abs_tol=1e-3)

    def test_missing_arm_yields_nan_not_a_number(self):
        """An absent condition must not read as a finding.

        Returning 0.0 or 0.5 here would be the totalizing-substrate failure in
        the reporting layer: a number where there is no measurement.
        """
        q = _report([_sample("grammatical", 0.5, 0.0)]).p600_quantities()
        assert math.isnan(q.auc_of_raw)


class TestAgainstTheLiveReport:

    @pytest.mark.slow
    def test_auc_of_raw_matches_the_separation_key(self, forked_parser):
        """`separation["p600_auc"]` ranks the RAW deficit -- pinned here.

        The key is spelled `p600_auc`, which says nothing about WHICH quantity
        it ranks. Believing it ranked the excess produced a whole false
        explanation for a real discrepancy. If this ever fails, the name and the
        computation have diverged and every comparison between the two is
        suspect.
        """
        from neural_assemblies.assembly_calculus.emergent.evaluation import (
            calibrate_erp_thresholds,
        )
        report = calibrate_erp_thresholds(forked_parser("SENTENCES", seed=11))
        q = report.p600_quantities()
        assert math.isclose(
            q.auc_of_raw, report.separation["p600_auc"], abs_tol=1e-9), (
            f"p600_quantities().auc_of_raw={q.auc_of_raw} disagrees with "
            f"separation['p600_auc']={report.separation['p600_auc']}: the "
            f"separation key is no longer computed on the raw deficit"
        )


def test_the_docstring_names_all_three():
    """Cheap guard that the type keeps explaining itself.

    The whole defect was a naming one, so an `ErpQuantities` whose docstring
    stopped distinguishing the three would be the defect returning.
    """
    doc = ErpQuantities.__doc__ or ""
    for term in ("deficit_raw", "excess_over_baseline", "auc_of_raw"):
        assert term in doc, f"{term} is no longer explained on ErpQuantities"

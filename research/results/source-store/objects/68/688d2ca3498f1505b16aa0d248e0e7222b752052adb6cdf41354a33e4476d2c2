"""Where does p600_excess become 0.0 on BOTH calibration arms? (task #32)

Two hypotheses were already killed by measurement, and this file exists so the
third is not guessed at either:

  REFUTED  the adapter floor-clamps. ``anchored_p600_live`` ends in
           ``max(0.0, 1.0 - energy)``, which would report 0.0 if energy
           exceeded 1.0. Measured on a trained n=1000 parser: energy 0.0933
           (grammatical) / 0.0622 (violation), reported 0.8915 / 0.9352. Not
           clamped -- healthy, non-zero, and in the right DIRECTION.

  REFUTED  the readiness gate. ``measure_live_integration`` returns early with
           ``(0.0, VP, 1.0)`` when ``core is None`` or ``not p600_ready``, and
           the observed triple matched that literal. But readiness came back
           ready=True with both cores present.

So the zero is introduced between a healthy per-word p600 and the p600_EXCESS
that calibration actually reports. This traces the whole chain per frame:

    anchored deficit -> p600 -> baseline.p600_median -> p600_excess

``p600_excess`` is ``max(0.0, p600 - p600_median)`` (gates.ErpBaseline) and the
median is calibrated on GRAMMATICAL sentences over ALL positions. The prediction
under test: p600 is a narrow high band for every word -- grammatical and
violation alike -- so the grammatical median lands INSIDE that band, and the
subtraction clamps both arms to zero. If so the defect is dynamic range, not
sign: the metric separates (0.8915 vs 0.9352) but the baseline subtraction is
larger than the separation and destroys it.

That is a falsifiable prediction, so it is printed rather than assumed: the
band width and the median's position within it decide it either way.
"""

from __future__ import annotations

import os
import statistics

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

from neural_assemblies.assembly_calculus.emergent.evaluation.erp.adapters import (
    N400_WEIGHT,
    P600_WEIGHT,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.calibration import (
    calibrate_erp_thresholds,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.gates import (
    assess_erp_readiness,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (
    get_parser_cache,
)


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = get_parser_cache().fork("SENTENCES", seed=42)
    readiness = assess_erp_readiness(parser)
    print(f"\nreadiness: p600={readiness.p600_ready} n400={readiness.n400_ready} "
          f"(lex={readiness.prediction_lexicon_size}, "
          f"sents={readiness.sentences_seen}, "
          f"roles={readiness.role_pathways_linked})")

    report = calibrate_erp_thresholds(parser)
    base = report.baseline
    print(f"\nbaseline: p600_median={base.p600_median:.4f} "
          f"n={base.sample_size} src={base.source}")

    # -- per-frame chain -----------------------------------------------------
    # p600 = N400_WEIGHT * (1 - stability) + P600_WEIGHT * (1 - energy).
    # Both terms are printed rather than inferred from the total, because the
    # question is WHICH one is saturated -- inverting the sum to recover a term
    # assumes the other is exactly at its ceiling, which is the hypothesis.
    print(f"\n  {'label':<20} {'word':<8} {'cat':<6} "
          f"{'stab':>7} {'instab':>7} {'deficit':>8} {'energy':>8} "
          f"{'p600':>8} {'excess':>8}")
    raw = {}
    energies = {}
    for s in report.samples:
        raw.setdefault(s.label, []).append(s.p600)
        instab = 1.0 - s.phrase_stability
        deficit = (s.p600 - N400_WEIGHT * instab) / P600_WEIGHT
        energy = 1.0 - deficit
        energies.setdefault(s.label, []).append(energy)
        print(f"  {s.label:<20} {s.word:<8} {s.category:<6} "
              f"{s.phrase_stability:>7.4f} {instab:>7.4f} {deficit:>8.4f} "
              f"{energy:>8.4f} {s.p600:>8.4f} {s.p600_excess:>8.4f}")

    # -- does the RAW p600 separate, before the baseline subtraction? --------
    print("\n  raw p600 (pre-subtraction), by label")
    for label in ("grammatical", "category_violation", "novel_noun"):
        vals = raw.get(label, [])
        if not vals:
            continue
        print(f"    {label:<20} n={len(vals)} "
              f"median={statistics.median(vals):.4f} "
              f"min={min(vals):.4f} max={max(vals):.4f}")

    # -- the energies themselves, which is what the metric is built on -------
    print("\n  pre-k-WTA energy into the role area, by label")
    for label in ("grammatical", "category_violation", "novel_noun"):
        vals = energies.get(label, [])
        if not vals:
            continue
        print(f"    {label:<20} n={len(vals)} "
              f"median={statistics.median(vals):.4f} "
              f"min={min(vals):.4f} max={max(vals):.4f}")
    ge = energies.get("grammatical", [])
    ce = energies.get("category_violation", [])
    if ge and ce:
        gm, cm = statistics.median(ge), statistics.median(ce)
        print(f"\n    energy drop catv vs gram: {gm:.4f} -> {cm:.4f} "
              f"= {(gm - cm) / gm * 100:+.1f}% relative")
        print(f"    same contrast as 1-energy: {1 - gm:.4f} -> {1 - cm:.4f} "
              f"= {((1 - cm) - (1 - gm)) / (1 - gm) * 100:+.1f}% relative")

    gram = raw.get("grammatical", [])
    catv = raw.get("category_violation", [])
    if gram and catv:
        band_lo = min(gram + catv)
        band_hi = max(gram + catv)
        sep = statistics.median(catv) - statistics.median(gram)
        print(f"\n  band       [{band_lo:.4f}, {band_hi:.4f}]  width={band_hi - band_lo:.4f}")
        print(f"  median     {base.p600_median:.4f} "
              f"({'INSIDE' if band_lo <= base.p600_median <= band_hi else 'outside'} the band)")
        print(f"  separation catv - gram = {sep:+.4f}")
        print(f"  verdict    subtraction {'DESTROYS' if base.p600_median >= band_hi else 'preserves'} "
              f"the separation")

    print(f"\n  cohens_d(p600_excess) = {report.separation.get('p600_cohens_d', 0):+.4f}")


if __name__ == "__main__":
    main()

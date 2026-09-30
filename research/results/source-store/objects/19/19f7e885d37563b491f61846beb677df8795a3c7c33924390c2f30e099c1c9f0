"""Emergent parser ERP layer (gates → adapters → runner → frames → calibration).

WHAT THIS LAYER IS FOR.  The two ERP kernels in
``assembly_calculus.metrics`` compute raw quantities: N400 as prediction error
(1 - overlap) and P600 as settling cost (summed Jaccard change).  Those
numbers are not directly interpretable, for a reason that is intrinsic rather
than incidental -- their scale depends on the parser instance.  A parser with
a small vocabulary, a shallow curriculum stage, or a different ``k`` produces
systematically different magnitudes for the SAME sentence, because assembly
sizes and chance overlap differ.

So the layer's job is to turn a raw quantity into a judgement, in four steps:

    frames        stimulus sets with a known critical position -- the word at
                  which a violation, if any, occurs
    adapters      measure N400/P600 at that position on a live parser
    calibration   establish this parser's own grammatical baseline, so what
                  counts as elevated is defined per instance
    gates         decide whether the excess over baseline is a violation, and
                  refuse to answer at all when the required pathways are not
                  trained

That last part is the important discipline.  An untrained prediction bridge
gives a large N400 for every word, grammatical or not, and an unconsolidated
role pathway gives a large P600 for every word -- so a model that has not
learned anything looks maximally sensitive to violations.  The readiness gates
exist to make that case return "not ready" rather than a false positive.

Package layout::

    gates.py       — readiness, baselines, violation typing
    adapters.py    — live/fresh N400/P600 measure functions
    runner.py      — incremental probe runner, ErpProbeResult
    frames.py      — calibration frames, critical positions
    calibration.py — empirical threshold tuning
"""

from .adapters import (
    N400_WEIGHT,
    P600_WEIGHT,
    anchored_p600_live,
    measure_fresh_stimulus_integration,
    measure_integration_instability,
    measure_lexical_surprise,
    measure_live_integration,
    measure_n400_surprise,
    phrase_stability,
    structural_role_area,
)
from .calibration import (
    ErpCalibrationReport,
    calibrate_erp_thresholds,
    ensure_parser_erp_calibration,
    tune_thresholds_from_samples,
)
from .frames import (
    DEFAULT_CALIBRATION_FRAMES,
    AREA_MATCHED_CALIBRATION_FRAMES,
    TRAINED_AREA_MATCHED_CALIBRATION_FRAMES,
    audit_frame_vocabulary,
    frame_word_status,
    unusable_frame_words,
    CalibrationFrame,
    PositionErpSample,
    collect_frame_samples,
    collect_position_samples,
    critical_position_for_frame,
)
from .gates import (
    ErpBaseline,
    ErpReadiness,
    ErpThresholds,
    ErpViolation,
    MIN_DISTRIBUTIONAL_SENTENCES,
    MIN_PREDICTION_LEXICON,
    N400_EXCESS_MARGIN,
    NOVEL_N400_EXCESS,
    P600_EXCESS_MARGIN,
    PHRASE_STABILITY_RATIO,
    REFERENCE_N400_CATEGORY_VIOLATION,
    REFERENCE_N400_GRAMMATICAL,
    REFERENCE_P600_CATEGORY_VIOLATION,
    REFERENCE_P600_GRAMMATICAL,
    assess_erp_readiness,
    areas_with_active_assembly,
    calibrate_erp_baseline,
    classify_erp_violation,
    default_erp_thresholds,
    parser_erp_baseline,
    parser_erp_thresholds,
)
from .runner import ErpProbeResult, probe_word_at_position, run_incremental_erp_probes

__all__ = [
    "DEFAULT_CALIBRATION_FRAMES",
    "AREA_MATCHED_CALIBRATION_FRAMES",
    "TRAINED_AREA_MATCHED_CALIBRATION_FRAMES",
    "audit_frame_vocabulary",
    "frame_word_status",
    "unusable_frame_words",
    "CalibrationFrame",
    "ErpBaseline",
    "ErpCalibrationReport",
    "ErpProbeResult",
    "ErpReadiness",
    "ErpThresholds",
    "ErpViolation",
    "N400_EXCESS_MARGIN",
    "N400_WEIGHT",
    "NOVEL_N400_EXCESS",
    "P600_EXCESS_MARGIN",
    "P600_WEIGHT",
    "PHRASE_STABILITY_RATIO",
    "PositionErpSample",
    "REFERENCE_N400_CATEGORY_VIOLATION",
    "REFERENCE_N400_GRAMMATICAL",
    "REFERENCE_P600_CATEGORY_VIOLATION",
    "REFERENCE_P600_GRAMMATICAL",
    "anchored_p600_live",
    "assess_erp_readiness",
    "areas_with_active_assembly",
    "calibrate_erp_baseline",
    "calibrate_erp_thresholds",
    "classify_erp_violation",
    "collect_frame_samples",
    "collect_position_samples",
    "critical_position_for_frame",
    "default_erp_thresholds",
    "ensure_parser_erp_calibration",
    "measure_fresh_stimulus_integration",
    "measure_integration_instability",
    "measure_lexical_surprise",
    "measure_live_integration",
    "measure_n400_surprise",
    "parser_erp_baseline",
    "parser_erp_thresholds",
    "phrase_stability",
    "probe_word_at_position",
    "run_incremental_erp_probes",
    "structural_role_area",
    "tune_thresholds_from_samples",
]

"""Mine wobbly episodes from raw sentence exposure."""

from __future__ import annotations

from typing import List, Optional, Set, TYPE_CHECKING

from ...evaluation.erp import (
    assess_erp_readiness,
    calibrate_erp_baseline,
    parser_erp_baseline,
    parser_erp_thresholds,
    run_incremental_erp_probes,
)
from .hypotheses import generate_pos_hypotheses
from .memory import WobblyEpisode, WobblyMemory, activate_error, STRUCTURAL_SIGNATURES

if TYPE_CHECKING:
    from ...parser import EmergentParser


def mine_wobbly_episodes(
    parser: "EmergentParser",
    sentences: List[List[str]],
    *,
    memory: Optional[WobblyMemory] = None,
    min_sentence_len: int = 2,
    target_words: Optional[Set[str]] = None,
    probe_depth: str = "mining",
    collected_probes: Optional[list] = None,
) -> WobblyMemory:
    """Scan raw sentences; store positions where live parse integration wobbles."""
    from ..pos_inference import is_word_in_lexicon
    from ...evaluation.erp.probe_util import critical_probe_measure_fn

    mem = memory if memory is not None else WobblyMemory()
    if not hasattr(parser, "_wobbly_memory"):
        parser._wobbly_memory = mem
    else:
        mem = parser._wobbly_memory

    readiness = assess_erp_readiness(parser)
    if not readiness.any_ready:
        return mem

    if readiness.p600_ready:
        from ...evaluation.erp import ensure_parser_erp_calibration
        from ...evaluation.erp.gates import ErpThresholds

        th = getattr(parser, "_erp_thresholds", None)
        if not isinstance(th, ErpThresholds) or th.source != "empirical":
            ensure_parser_erp_calibration(parser)

    baseline = parser_erp_baseline(parser)
    if baseline.source != "calibrated" and baseline.sample_size == 0:
        from ...evaluation.erp.probe_util import critical_probe_measure_fn

        baseline = calibrate_erp_baseline(
            parser,
            sentences,
            max_sentences=4,
            measure_fn=critical_probe_measure_fn(probe_depth),
            critical_position_only=True,
        )
        parser._erp_baseline = baseline

    thresholds = parser_erp_thresholds(parser)

    for sent in sentences:
        known = [w for w in sent if w in parser.stim_map]
        if len(known) < min_sentence_len:
            continue
        _, probes = run_incremental_erp_probes(
            parser,
            known,
            apply_calibration=True,
            activate_error=True,
            error_callback=activate_error,
            baseline=baseline,
            readiness=readiness,
            thresholds=thresholds,
            probe_depth=probe_depth,
            finalize_parse=False,
        )
        if collected_probes is not None:
            if target_words:
                collected_probes.extend(p for p in probes if p.word in target_words)
            else:
                collected_probes.extend(probes)
        for probe in probes:
            if not probe.wobbly:
                continue
            if target_words is not None and probe.word not in target_words:
                continue
            if is_word_in_lexicon(parser, probe.word):
                continue
            if probe.failure_signature not in STRUCTURAL_SIGNATURES:
                continue
            hyps = generate_pos_hypotheses(
                parser,
                probe.word,
                probe.category,
                failure_signature=probe.failure_signature,
            )
            mem.add(
                WobblyEpisode(
                    sentence=tuple(known),
                    probe=probe,
                    hypotheses=hyps,
                ),
            )
    return mem

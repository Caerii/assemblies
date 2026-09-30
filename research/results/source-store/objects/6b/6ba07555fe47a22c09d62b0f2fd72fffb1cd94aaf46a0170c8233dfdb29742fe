"""Incremental ERP probe runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Set, Tuple, TYPE_CHECKING

from .protocol import ErpProtocol
from .adapters import (
    N400_WEIGHT,
    P600_WEIGHT,
    measure_lexical_surprise,
    measure_live_integration,
)
from .gates import (
    ErpBaseline,
    ErpReadiness,
    ErpThresholds,
    ErpViolation,
    assess_erp_readiness,
    classify_erp_violation,
    parser_erp_baseline,
    parser_erp_thresholds,
)
from ...core.areas import CATEGORY_TO_CORE

if TYPE_CHECKING:
    from ...parser import EmergentParser


@dataclass
class ErpProbeResult:
    """ERP readout at one word position during incremental parse."""
    word: str
    position: int
    prefix: Tuple[str, ...]
    category: str
    n400: float
    p600: float
    combined: float
    phrase_stability: float
    role_area: str
    wobbly: bool
    error_active: bool = False
    failure_signature: str = ""
    violation: Optional[ErpViolation] = None


def _failure_signature(violation: ErpViolation) -> str:
    return violation.failure_signature if violation.wobbly else ""


def run_incremental_erp_probes(
    parser: "EmergentParser",
    words: List[str],
    *,
    apply_calibration: bool = True,
    activate_error: bool = False,
    baseline: Optional[ErpBaseline] = None,
    readiness: Optional[ErpReadiness] = None,
    thresholds: Optional[ErpThresholds] = None,
    error_callback=None,
    probe_depth: str = "calibration",
    probe_positions: Optional[Set[int]] = None,
    stop_at_position: Optional[int] = None,
    finalize_parse: bool = True,
    protocol: Optional[ErpProtocol] = None,
) -> Tuple[dict, List[ErpProbeResult]]:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-erp-context-reset

    Full incremental parse with per-word N400/P600 probes.
    The protocol records construction versus activity-only context reset. Use
    separate brain.probe scopes for isolated reads of an initialized parser;
    activity-only reset itself does not make this entire operation read-only.

    When *probe_positions* is set, only those indices produce probe records
    (parse still runs through *stop_at_position* or sentence end).
    *stop_at_position* ends the loop after probing that index (inclusive).

    *protocol* is resolved ONCE here and passed to every probe, so one parse
    cannot measure its first word under one protocol and its last under
    another -- which an environment read per probe permitted (#115).
    """
    protocol = ErpProtocol.from_environment() if protocol is None else protocol
    _check_engine_identity(parser, protocol)
    if not words:
        return {"categories": {}, "roles": {}, "phrases": {}, "wobbly_probes": [], "erp_protocol": asdict(protocol)}, []

    readiness = readiness or assess_erp_readiness(parser)
    baseline = baseline if baseline is not None else parser_erp_baseline(parser)
    thresholds = thresholds or parser_erp_thresholds(parser)

    if protocol.context_reset == "activity":
        parser._reset_context_winners(preserve_mapping=True)
    else:
        parser._reset_context_state()
    circuit = parser._get_incremental_circuit(reset=True)
    result: dict = {
        "erp_protocol": asdict(protocol),
        "categories": {},
        "roles": {},
        "phrases": {},
        "steps": [],
    }
    probes: List[ErpProbeResult] = []
    verb_seen = False
    noun_count = 0
    subject_core: Optional[str] = None
    main_verb: Optional[str] = None
    prefix: List[str] = []
    last_index = len(words) - 1
    if stop_at_position is not None:
        last_index = min(last_index, stop_at_position)

    with parser.brain.frozen():
        for i, word in enumerate(words):
            if i > last_index:
                break
            record_probe = probe_positions is None or i in probe_positions
            n400 = 0.0
            if record_probe and prefix:
                # BEHAVIOUR-PRESERVING: each undefined branch carries the float
                # it used to return in `detail["legacy"]`, so the arithmetic is
                # byte-identical while the fallback is now VISIBLE. The 1.0
                # branch is the one that matters -- a word missing from the
                # prediction lexicon read as MAXIMUM surprise, which is what an
                # anomaly is supposed to look like (#28).
                n400_m = measure_lexical_surprise(
                    parser, tuple(prefix), word, readiness=readiness,
                    protocol=protocol,
                )
                n400 = n400_m.or_else(
                    float((n400_m.detail or {}).get("legacy", 0.0)))
                if not n400_m.defined:
                    result.setdefault("n400_undefined", []).append(
                        {"word": word, "position": i, "why": n400_m.why,
                         "legacy": n400})

            # CAPTURED BEFORE CONSUMPTION, and that is the whole point.
            # `expected_role_area` asks what the parse predicted at this
            # position; taking `verb_seen` after the word is consumed would make
            # the verb itself predict an object slot, which is the same class of
            # error the expected-slot dispatch exists to fix.
            verb_seen_before = verb_seen

            cat, verb_seen, noun_count = parser._advance_incremental_word(
                word, circuit, verb_seen, noun_count,
            )
            result["categories"][word] = cat
            prefix.append(word)

            if cat == "VERB" and main_verb is None:
                main_verb = word

            if cat in ("NOUN", "PRON") and len(prefix) <= 2 and not verb_seen:
                subject_core = CATEGORY_TO_CORE.get(cat)
            elif verb_seen and cat in ("NOUN", "PRON") and subject_core is None:
                subject_core = CATEGORY_TO_CORE.get(cat)

            if not record_probe:
                continue

            p600, role_area, stability = measure_live_integration(
                parser,
                word,
                cat,
                verb_seen=verb_seen,
                subject_core=subject_core,
                readiness=readiness,
                probe_depth=probe_depth,
                verb_seen_before=verb_seen_before,
                object_open=parser._verb_takes_an_object(main_verb),
                protocol=protocol,
            )
            combined = N400_WEIGHT * n400 + P600_WEIGHT * p600

            violation: Optional[ErpViolation] = None
            if apply_calibration:
                violation = classify_erp_violation(
                    n400,
                    p600,
                    readiness=readiness,
                    baseline=baseline,
                    phrase_stability=stability,
                    thresholds=thresholds,
                )
                wobbly = violation.wobbly
                sig = _failure_signature(violation)
            else:
                wobbly = False
                sig = ""

            error_active = False
            if wobbly and activate_error and error_callback is not None:
                error_active = error_callback(parser, combined)

            probes.append(
                ErpProbeResult(
                    word=word,
                    position=i,
                    prefix=tuple(prefix[:-1]),
                    category=cat,
                    n400=round(n400, 4),
                    p600=round(p600, 4),
                    combined=round(combined, 4),
                    phrase_stability=round(stability, 4),
                    role_area=role_area,
                    wobbly=wobbly,
                    error_active=error_active,
                    failure_signature=sig,
                    violation=violation,
                ),
            )
            result["steps"].append({
                "word": word,
                "category": cat,
                "position": i,
                "n400": probes[-1].n400,
                "p600": probes[-1].p600,
                "phrase_stability": probes[-1].phrase_stability,
                "wobbly": wobbly,
            })
            if stop_at_position is not None and i >= stop_at_position:
                break

    if finalize_parse:
        # Early-stop probes intentionally consume only a prefix, but final
        # role/phrase parsing is a whole-sentence operation. Fill categories
        # for unconsumed tokens through the canonical classifier before the
        # total-map contracts in those operations are invoked.
        for word in words:
            if word not in result["categories"]:
                result["categories"][word] = parser.classify_word_cached(word)[0]
        result["roles"] = parser._assign_roles_neural(words, result["categories"])
        result["phrases"] = parser._identify_phrases(words, result["categories"])
    else:
        result["roles"] = {}
        result["phrases"] = {}
    result["wobbly_probes"] = probes
    result["erp_readiness"] = readiness
    result["erp_baseline"] = baseline
    return result, probes


def _check_engine_identity(parser: "EmergentParser", protocol: ErpProtocol) -> None:
    """Reject a measurement when its declared engine is not the live engine."""
    if protocol.engine_name is None:
        return
    actual = str(getattr(parser, "engine_name", "unknown"))
    if actual != protocol.engine_name:
        raise ValueError(
            "ERP protocol requires engine "
            f"{protocol.engine_name!r}, parser uses {actual!r}; "
            "the observation is void until the engine is explicit and matched",
        )


def probe_word_at_position(
    parser: "EmergentParser",
    words: List[str],
    position: int,
    *,
    baseline: Optional[ErpBaseline] = None,
    readiness: Optional[ErpReadiness] = None,
    **probe_kw,
) -> ErpProbeResult:
    """ERP probe at a single position (re-parses through that index)."""
    _, probes = run_incremental_erp_probes(
        parser,
        words,
        baseline=baseline,
        readiness=readiness,
        stop_at_position=position,
        probe_positions={position},
        **probe_kw,
    )
    for p in probes:
        if p.position == position:
            return p
    raise IndexError(f"no probe at position {position}")

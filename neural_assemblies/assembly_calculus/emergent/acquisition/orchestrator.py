"""Developmental acquisition orchestrator — ground-up curriculum with reflection."""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings
from typing import Dict, List, Optional, Set, TYPE_CHECKING

from .adaptive import AdaptiveHint, RemediationResult
from .stage_gates import StageGateResult, evaluate_stage_gate, inter_stage_sleep

if TYPE_CHECKING:
    from ..curriculum import StageResult
    from ..parser import EmergentParser

# Full child-development order (includes pre-lexical babble).
DEVELOPMENTAL_STAGE_ORDER: tuple[str, ...] = (
    "BABBLE",
    "FIRST_WORDS",
    "VOCABULARY_SPURT",
    "TWO_WORD",
    "SENTENCES",
    "COMPLEX_GRAMMAR",
    "DIALOGUE",
    "CONVERSATION",
)


@dataclass
class StageReflection:
    """Qualitative + quantitative notes after each developmental stage."""
    stage: str
    vocab_size: int
    classification_accuracy: float
    phases_run: List[str]
    observations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    adaptive_hints: List[AdaptiveHint] = field(default_factory=list)
    remediation: Optional[RemediationResult] = None
    wobbly_bootstrap: Optional[Dict[str, object]] = None
    gate: Optional[StageGateResult] = None


@dataclass
class AcquisitionReport:
    """Full developmental run with per-stage reflection."""
    max_stage: str
    stages_run: List[str]
    reflections: List[StageReflection]
    skipped_early: bool
    fuzzy_variant_count: int = 0
    babble_forms: int = 0
    remediations: List[RemediationResult] = field(default_factory=list)
    wobbly_bootstraps: List[Dict[str, object]] = field(default_factory=list)
    gate_results: List[StageGateResult] = field(default_factory=list)
    blocked_at_stage: Optional[str] = None
    final_generalization: Optional[Dict[str, object]] = None


def reflect_after_stage(
    parser: "EmergentParser",
    result: "StageResult",
    *,
    holdout_words: Optional[Set[str]] = None,
) -> StageReflection:
    """Deep reflection: what was learned, what to fix next."""
    from .pos_inference import decompose_holdout_classification
    from .adaptive import classification_accuracy_usable

    stage = result.stage_name
    obs: List[str] = []
    recs: List[str] = []
    hints: List[AdaptiveHint] = []
    metrics: Dict[str, float] = {
        "classification_accuracy": result.classification_accuracy,
        "sentences_trained": float(result.sentences_trained),
    }

    holdout_map = {
        "bird": "NOUN",
        "finds": "VERB",
        "small": "ADJ",
    }
    if holdout_words:
        holdout_map = {
            w: holdout_map.get(w, "NOUN")
            for w in holdout_words
        }

    if stage == "BABBLE":
        forms = getattr(parser, "babble_forms", [])
        obs.append(f"registered {len(forms)} pre-lexical babble forms")
        obs.append("no lexicon commitment — phon exposure only")
        if len(forms) < 12:
            recs.append("increase babble form diversity")
            hints.append(AdaptiveHint("babble_more", detail="low form diversity"))
        metrics["babble_forms"] = float(len(forms))

    elif stage == "FIRST_WORDS":
        obs.append("first grounded lexicon assemblies forming")
        if (
            classification_accuracy_usable(result.classification_accuracy)
            and result.classification_accuracy < 0.5
        ):
            recs.append("extend FIRST_WORDS exposure or check grounding stimuli")
            hints.append(AdaptiveHint("remedial_pos"))
        fuzzy = getattr(parser, "surface_to_canonical", {})
        if fuzzy:
            obs.append(f"{len(fuzzy)} fuzzy surface forms linked to canonical lemmas")
        else:
            recs.append("register fuzzy variants for early nouns (dog/dgg, mama/mma)")
            hints.append(AdaptiveHint("register_fuzzy"))

    elif stage in ("VOCABULARY_SPURT", "TWO_WORD"):
        obs.append("distributional POS bootstrap should be accumulating")
        if "distributional" not in result.phases_run:
            recs.append("ensure distributional phase runs at this stage")
            hints.append(AdaptiveHint("ensure_distributional"))

    elif stage == "SENTENCES":
        obs.append("structural generalization (roles, word order) hinges on this stage")
        if (
            classification_accuracy_usable(result.classification_accuracy)
            and result.classification_accuracy < 0.7
        ):
            recs.append("holdout decomposition: check adjective distributional exposure")
            hints.append(AdaptiveHint("remedial_pos"))
        try:
            decomp = decompose_holdout_classification(parser, holdout_map)
            metrics["holdout_bootstrap"] = decomp["accuracy_bootstrapped"]
            if decomp["accuracy_bootstrapped"] < 1.0:
                for word, mode in decomp.get("failure_modes", {}).items():
                    recs.append(f"holdout {word}: {mode}")
                    expected = holdout_map.get(word, "NOUN")
                    hints.append(
                        AdaptiveHint(
                            "holdout_remedial",
                            words=((word, expected),),
                            detail=str(mode or ""),
                        ),
                    )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            warnings.warn(
                "stage reflection could not compute holdout decomposition; "
                f"holdout_bootstrap is unavailable ({error!r})",
                RuntimeWarning,
                stacklevel=2,
            )

    elif stage in ("DIALOGUE", "CONVERSATION"):
        obs.append("bridge + dialogue pathways for interactive chat")
        if "word_order" not in result.phases_run:
            recs.append(
                "word_order phase skipped — risk of SVO regression; "
                "run consolidation or replay",
            )
            hints.append(AdaptiveHint("word_order_replay"))
        pred_lex = getattr(parser, "prediction_lexicon", {})
        metrics["prediction_lexicon_size"] = float(len(pred_lex))
        if len(pred_lex) < 20:
            recs.append("train more prediction bridges before expecting fluent chat")

    if (
        classification_accuracy_usable(result.classification_accuracy)
        and result.classification_accuracy >= 0.85
    ):
        obs.append("classification strong for stage vocabulary")
    elif (
        stage not in ("BABBLE",)
        and classification_accuracy_usable(result.classification_accuracy)
    ):
        recs.append("run adaptive curriculum: add sentences for misclassified POS")
        if not any(h.action == "remedial_pos" for h in hints):
            hints.append(AdaptiveHint("remedial_pos"))

    return StageReflection(
        stage=stage,
        vocab_size=result.vocab_size,
        classification_accuracy=result.classification_accuracy,
        phases_run=list(result.phases_run),
        observations=obs,
        recommendations=recs,
        metrics=metrics,
        adaptive_hints=hints,
    )


def format_acquisition_report(report: AcquisitionReport) -> str:
    lines = [
        "Developmental acquisition report",
        f"  target stage: {report.max_stage}",
        f"  stages run: {' -> '.join(report.stages_run)}",
        f"  skipped early: {report.skipped_early}",
        f"  babble forms: {report.babble_forms}",
        f"  fuzzy surfaces: {report.fuzzy_variant_count}",
        "",
    ]
    for ref in report.reflections:
        lines.append(f"--- {ref.stage} ---")
        acc = ref.classification_accuracy
        acc_label = "skipped (sweep)" if acc < 0 else f"{acc:.1%}"
        lines.append(
            f"  vocab={ref.vocab_size} acc={acc_label} "
            f"phases={ref.phases_run}",
        )
        if ref.gate is not None:
            status = "PASS" if ref.gate.passed else "FAIL"
            lines.append(f"  gate: {status} checks={ref.gate.checks}")
            if ref.gate.failures:
                lines.append(f"  gate failures: {ref.gate.failures}")
        for o in ref.observations:
            lines.append(f"  observe: {o}")
        for r in ref.recommendations:
            lines.append(f"  next: {r}")
        if ref.remediation:
            rem = ref.remediation
            lines.append(
                f"  remedial: {rem.sentences_trained} sents "
                f"phases={rem.phases_run} targets={rem.targets_addressed}",
            )
        if ref.wobbly_bootstrap:
            wb = ref.wobbly_bootstrap
            lines.append(
                f"  wobbly: {wb.get('episodes', 0)} episodes "
                f"words={wb.get('wobbly_words', [])}",
            )
        lines.append("")
    if report.blocked_at_stage:
        lines.append(f"BLOCKED at stage: {report.blocked_at_stage}")
        lines.append("")
    return "\n".join(lines)


def acquisition_report_to_dict(report: AcquisitionReport) -> Dict[str, object]:
    """Serialize an acquisition report for JSON export."""
    return {
        "max_stage": report.max_stage,
        "stages_run": report.stages_run,
        "skipped_early": report.skipped_early,
        "blocked_at_stage": report.blocked_at_stage,
        "babble_forms": report.babble_forms,
        "fuzzy_variant_count": report.fuzzy_variant_count,
        "stages": [
            {
                "stage": ref.stage,
                "vocab_size": ref.vocab_size,
                "classification_accuracy": ref.classification_accuracy,
                "phases_run": ref.phases_run,
                "metrics": ref.metrics,
                "gate": (
                    {
                        "passed": ref.gate.passed,
                        "checks": ref.gate.checks,
                        "metrics": ref.gate.metrics,
                        "failures": ref.gate.failures,
                    }
                    if ref.gate is not None
                    else None
                ),
            }
            for ref in report.reflections
        ],
        "final_generalization": report.final_generalization,
    }


def run_developmental_acquisition(
    parser: "EmergentParser",
    *,
    max_stage: str = "SENTENCES",
    holdout_words: Optional[Set[str]] = None,
    babble: bool = True,
    fuzzy_early_words: bool = True,
    adaptive: bool = True,
    wobbly_bootstrap: bool = True,
    gate_enforcement: bool = True,
    gate_retries: int = 1,
    inter_stage_sleep_enabled: bool = True,
    seed: int = 42,
) -> AcquisitionReport:
    """Ground-up developmental training with reflection after each stage."""
    from .adaptive import apply_adaptive_plan, build_adaptive_plan
    from .babble import (
        isolate_babble_forms,
        register_early_fuzzy_variants,
        train_babble_stage,
    )
    from ..curriculum import CurriculumTrainer
    from ..train_progress import current_progress
    from ..training.perf import should_skip_early_curriculum

    if max_stage not in DEVELOPMENTAL_STAGE_ORDER:
        raise ValueError(
            f"max_stage {max_stage!r} not in {DEVELOPMENTAL_STAGE_ORDER}"
        )

    trainer = CurriculumTrainer(parser, holdout_words=holdout_words)
    reflections: List[StageReflection] = []
    remediations: List[RemediationResult] = []
    wobbly_bootstraps: List[Dict[str, object]] = []
    gate_results: List[StageGateResult] = []
    blocked_at_stage: Optional[str] = None
    from ..evaluation.sweep import wobbly_bootstrap_stages
    _WOBBLY_STAGES = wobbly_bootstrap_stages(fast=parser.fast_training)
    stages_run: List[str] = []
    prog = current_progress()
    holdout_set = set(holdout_words or ())

    skip_early = should_skip_early_curriculum(len(parser.stim_map), max_stage)
    if skip_early and max_stage in ("DIALOGUE", "CONVERSATION"):
        stage_order = [
            s for s in DEVELOPMENTAL_STAGE_ORDER
            if s in ("DIALOGUE", "CONVERSATION")
        ]
        prog.info("perf path: skipping babble + early grammar stages")
    else:
        stage_order = list(DEVELOPMENTAL_STAGE_ORDER)

    for stage_name in stage_order:
        with prog.section(stage_name):
            result = None
            reflection = None

            for gate_attempt in range(gate_retries + 1):
                if stage_name == "BABBLE":
                    if not babble:
                        break
                    if gate_attempt == 0:
                        train_babble_stage(parser, seed=seed)
                    result = trainer.train_stage("BABBLE")
                    isolate_babble_forms(parser)
                elif stage_name == "FIRST_WORDS" and fuzzy_early_words:
                    if gate_attempt == 0:
                        stage_words = trainer._get_stage_words("FIRST_WORDS")
                        lemmas = [w.lemma for w in stage_words[:16]]
                        if lemmas:
                            register_early_fuzzy_variants(parser, lemmas)
                    result = trainer.train_stage(stage_name)
                else:
                    result = trainer.train_stage(stage_name)

                reflection = reflect_after_stage(
                    parser, result, holdout_words=holdout_set or None,
                )

                gate = evaluate_stage_gate(
                    parser,
                    result,
                    holdout_words=holdout_set or None,
                )
                reflection.gate = gate
                gate_results.append(gate)
                reflection.metrics.update(gate.metrics)

                if adaptive and reflection.adaptive_hints:
                    hints = list(reflection.adaptive_hints)
                    if gate.passed:
                        hints = [
                            h for h in hints
                            if h.action != "remedial_pos"
                        ]
                    if hints:
                        reflection.adaptive_hints = hints
                        stage_words = trainer._get_stage_words(stage_name)
                        plan = build_adaptive_plan(
                            reflection,
                            parser,
                            stage_words,
                            holdout_words=holdout_set or None,
                        )
                        remediation = apply_adaptive_plan(
                            parser,
                            trainer,
                            reflection,
                            plan,
                            stage_words=stage_words,
                            holdout_words=holdout_set or None,
                            seed=seed + len(stages_run) + gate_attempt,
                        )
                        if remediation is not None:
                            reflection.remediation = remediation
                            remediations.append(remediation)
                            reflection.metrics["remedial_sentences"] = float(
                                remediation.sentences_trained,
                            )
                            reflection.observations.append(
                                f"adaptive remediation: {remediation.sentences_trained} "
                                f"sentences, phases {remediation.phases_run}",
                            )
                            gate = evaluate_stage_gate(
                                parser,
                                result,
                                holdout_words=holdout_set or None,
                            )
                            reflection.gate = gate
                            gate_results.append(gate)
                            reflection.metrics.update(gate.metrics)

                if wobbly_bootstrap and stage_name in _WOBBLY_STAGES:
                    from ..evaluation import assess_erp_readiness
                    from .wobbly import mine_and_bootstrap_from_exposure

                    readiness = assess_erp_readiness(parser)
                    if readiness.p600_ready:
                        wb = mine_and_bootstrap_from_exposure(
                            parser,
                            trainer=trainer,
                            target_words=holdout_set or None,
                        )
                    else:
                        wb = {
                            "episodes": 0,
                            "assigned": {},
                            "remedial_sentences": 0,
                            "skipped": "p600_not_ready",
                        }
                    if wb.get("episodes", 0) > 0:
                        reflection.wobbly_bootstrap = wb
                        wobbly_bootstraps.append(wb)
                        reflection.metrics["wobbly_episodes"] = float(wb["episodes"])
                        reflection.observations.append(
                            f"wobbly-parse bootstrap: {wb['episodes']} episodes, "
                            f"{len(wb.get('assigned', {}))} category commits",
                        )

                if gate.passed or not gate_enforcement:
                    break

                if gate_attempt < gate_retries:
                    prog.info(
                        f"exit gate failed ({gate.failures}); "
                        f"retry {gate_attempt + 1}/{gate_retries}",
                    )
                else:
                    reflection.observations.append(
                        f"exit gate blocked advance: {gate.failures}",
                    )
                    blocked_at_stage = stage_name

            if stage_name == "BABBLE" and not babble:
                continue

            if reflection is not None:
                reflections.append(reflection)
                stages_run.append(stage_name)

            if blocked_at_stage is not None:
                break

        if inter_stage_sleep_enabled and blocked_at_stage is None:
            inter_stage_sleep(parser, stage_name)

        if stage_name == max_stage:
            break

    fuzzy_count = len(getattr(parser, "surface_to_canonical", {}))
    babble_n = len(getattr(parser, "babble_forms", []))

    final_generalization = None
    if max_stage in ("SENTENCES", "COMPLEX_GRAMMAR", "DIALOGUE", "CONVERSATION"):
        from ..evaluation.generalization import (
            DEFAULT_LEXICON_HOLDOUTS,
            evaluate_generalization_metrics,
        )

        holdout_dict = {
            w: DEFAULT_LEXICON_HOLDOUTS.get(w, "NOUN")
            for w in (holdout_set or set(DEFAULT_LEXICON_HOLDOUTS))
        }
        final_generalization = evaluate_generalization_metrics(
            parser,
            holdout_words=holdout_dict,
            max_bridge_probes=10,
            seed=seed,
        )

    return AcquisitionReport(
        max_stage=max_stage,
        stages_run=stages_run,
        reflections=reflections,
        skipped_early=skip_early,
        fuzzy_variant_count=fuzzy_count,
        babble_forms=babble_n,
        remediations=remediations,
        wobbly_bootstraps=wobbly_bootstraps,
        gate_results=gate_results,
        blocked_at_stage=blocked_at_stage,
        final_generalization=final_generalization,
    )

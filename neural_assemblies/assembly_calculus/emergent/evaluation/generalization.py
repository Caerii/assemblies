"""Robust generalization probes and compiled-vs-exact generalization gates.

TWO DIFFERENT QUESTIONS, both called "generalization" here.

1. Does the model handle words it never learned?  ``DEFAULT_LEXICON_HOLDOUTS``
   names words withheld from lexicon training whose GROUNDING FEATURES overlap
   trained peers -- "bird" shares visual features with "dog", "finds" shares
   motor features with other verbs.  That overlap is the whole hypothesis: if
   categories really live in shared grounding rather than in memorised word
   identities, a word never trained should still land in the right core area
   because its features drive it there.  Holdouts are chosen to be
   feature-adjacent for exactly this reason; a holdout sharing nothing with
   any trained word would test nothing.

2. Does the fast training path give the same answers as the exact one?  The
   compiled-vs-exact gates compare a parser trained under compiled projection
   fidelity against one trained exactly.  Compiled fidelity skips candidate
   sampling and connectome expansion (see ``core.projection_fidelity``), so it
   is not step-identical by construction -- the gate asks whether it is
   BEHAVIOURALLY equivalent at readout, which is the only equivalence claimed.
   A failure here means the speed-up changed the science, not that a test is
   flaky.

Keeping both in one module is deliberate: question 2 is only meaningful when
measured on question 1's probes, since agreement on memorised training items
would be uninformative.
"""

from __future__ import annotations

from numbers import Real
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.corpus_index import CorpusIndex
    from ..parser import EmergentParser
    from ..training.schedule import TrainingSchedule

# Words skipped during lexicon training; grounding overlaps trained peers.
DEFAULT_LEXICON_HOLDOUTS: Dict[str, str] = {
    "bird": "NOUN",
    "finds": "VERB",
    "small": "ADJ",
}

DEFAULT_ROLE_PROBES: List[dict] = [
    {
        "words": ["the", "dog", "chases", "the", "cat"],
        "expected_roles": {
            "dog": "AGENT", "chases": "ACTION", "cat": "PATIENT",
        },
    },
    {
        "words": ["she", "sees", "the", "bird"],
        "expected_roles": {
            "she": "AGENT", "sees": "ACTION", "bird": "PATIENT",
        },
    },
    {
        "words": ["the", "cat", "sleeps"],
        "expected_roles": {"cat": "AGENT", "sleeps": "ACTION"},
    },
]

# Sentences that combine held-out lemmas with trained structure.
NOVEL_COMPOSITION_PROBES: List[dict] = [
    {
        "words": ["the", "bird", "chases", "the", "boy"],
        "expected_roles": {
            "bird": "AGENT", "chases": "ACTION", "boy": "PATIENT",
        },
    },
    {
        "words": ["she", "finds", "the", "ball"],
        "expected_roles": {
            "she": "AGENT", "finds": "ACTION", "ball": "PATIENT",
        },
    },
    {
        "words": ["the", "small", "dog", "runs"],
        "expected_roles": {
            "dog": "AGENT", "runs": "ACTION",
        },
    },
]


def default_holdout_set() -> Set[str]:
    return set(DEFAULT_LEXICON_HOLDOUTS)


def resolve_holdout_set(words: Optional[Set[str]]) -> Set[str]:
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-parser-cache-identity

    None selects defaults; an explicitly empty collection stays empty.
    """
    return default_holdout_set() if words is None else set(words)


def is_word_in_lexicon(parser: "EmergentParser", word: str) -> bool:
    for lex in parser.core_lexicons.values():
        if word in lex:
            return True
    return False


def collect_bridge_probes(
    corpus_index: "CorpusIndex",
    parser: "EmergentParser",
    holdout_words: Set[str],
    *,
    oov_only: bool = True,
    max_probes: Optional[int] = 25,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Prefix probes where the next word is held out (OOV) or in-vocab (seen)."""
    prefixes: List[List[str]] = []
    expected: List[List[str]] = []
    seen: set = set()

    for trans in corpus_index.transitions:
        next_w = trans.next_word
        in_holdout = next_w in holdout_words
        if oov_only and not in_holdout:
            continue
        if not oov_only and in_holdout:
            continue
        if not all(is_word_in_lexicon(parser, w) for w in trans.prefix_words):
            continue
        if oov_only and is_word_in_lexicon(parser, next_w):
            continue

        key = (trans.prefix_words, next_w)
        if key in seen:
            continue
        seen.add(key)
        prefixes.append(list(trans.prefix_words))
        expected.append([next_w])

    if max_probes is not None and len(prefixes) > max_probes:
        prefixes = prefixes[:max_probes]
        expected = expected[:max_probes]

    return prefixes, expected


def evaluate_generalization_metrics(
    parser: "EmergentParser",
    *,
    holdout_words: Optional[Dict[str, str]] = None,
    corpus_index: Optional["CorpusIndex"] = None,
    qa_pairs: Optional[list] = None,
    max_bridge_probes: int = 20,
    seed: int = 42,
) -> Dict[str, object]:
    """Multi-metric generalization report for a trained parser."""
    from ..core.corpus_index import compile_corpus
    from ..curriculum.dialogue import get_dialogue_pairs
    from .suite import EvaluationSuite
    from ..curriculum.data import create_training_sentences
    from ..evaluation.parity import (
        collect_transition_probes,
        score_next_token_probes,
    )

    holdouts = holdout_words or dict(DEFAULT_LEXICON_HOLDOUTS)
    holdout_set = set(holdouts)
    suite = EvaluationSuite(parser)

    if corpus_index is None:
        corpus_index = compile_corpus(parser, create_training_sentences())

    if qa_pairs is None:
        qa_pairs = get_dialogue_pairs(seed=seed)

    dialogue_cases = [
        {
            "question": " ".join(p.question.words),
            "acceptable": p.answer.words[:3],
            "pattern_type": p.pattern_type,
        }
        for p in qa_pairs
    ]

    oov_prefixes, oov_expected = collect_bridge_probes(
        corpus_index,
        parser,
        holdout_set,
        oov_only=True,
        max_probes=max_bridge_probes,
    )
    seen_prefixes, seen_expected = collect_transition_probes(
        corpus_index, max_probes=max_bridge_probes,
    )

    lexicon = suite.evaluate_generalization(holdouts)
    from ..acquisition.pos_inference import decompose_holdout_classification

    holdout_decomposition = decompose_holdout_classification(parser, holdouts)
    roles = suite.evaluate_roles(DEFAULT_ROLE_PROBES)
    novel = suite.evaluate_roles(NOVEL_COMPOSITION_PROBES)
    word_order = suite.evaluate_word_order(target="SVO")
    dialogue = suite.evaluate_dialogue(dialogue_cases)
    bridge_oov = score_next_token_probes(parser, oov_prefixes, oov_expected)
    bridge_seen = score_next_token_probes(parser, seen_prefixes, seen_expected)

    composite_parts = [
        lexicon["accuracy"],
        roles["accuracy"],
        novel["accuracy"],
        bridge_seen["top5"],
        dialogue["accuracy"],
    ]
    if bridge_oov["total"] > 0:
        composite_parts.append(bridge_oov["top5"])

    return {
        "lexicon_holdout": lexicon,
        "holdout_decomposition": holdout_decomposition,
        "roles": roles,
        "novel_composition": novel,
        "word_order": word_order,
        "dialogue": dialogue,
        "bridge_oov": bridge_oov,
        "bridge_seen": bridge_seen,
        "composite": sum(composite_parts) / max(len(composite_parts), 1),
        "holdout_words": sorted(holdout_set),
        "oov_bridge_probe_count": bridge_oov["total"],
    }


def compare_generalization_parity(
    exact_parser: "EmergentParser",
    compiled_parser: "EmergentParser",
    *,
    holdout_words: Optional[Dict[str, str]] = None,
    corpus_index: Optional["CorpusIndex"] = None,
    qa_pairs: Optional[list] = None,
    max_bridge_probes: int = 20,
    seed: int = 42,
) -> Dict[str, object]:
    """Compare generalization metrics after exact vs compiled DIALOGUE training."""
    exact = evaluate_generalization_metrics(
        exact_parser, holdout_words=holdout_words, corpus_index=corpus_index,
        qa_pairs=qa_pairs, max_bridge_probes=max_bridge_probes, seed=seed,
    )
    compiled = evaluate_generalization_metrics(
        compiled_parser, holdout_words=holdout_words, corpus_index=corpus_index,
        qa_pairs=qa_pairs, max_bridge_probes=max_bridge_probes, seed=seed,
    )

    def _nested_metric(report: Mapping[str, object], key: str,
                       field: str) -> float:
        value = report.get(key)
        if not isinstance(value, Mapping):
            raise TypeError(f"generalization result {key!r} must be a mapping")
        metric = value.get(field)
        if isinstance(metric, bool) or not isinstance(metric, Real):
            raise TypeError(f"generalization metric {key}.{field} must be real")
        return float(metric)

    def _scalar_metric(report: Mapping[str, object], key: str) -> float:
        metric = report.get(key)
        if isinstance(metric, bool) or not isinstance(metric, Real):
            raise TypeError(f"generalization metric {key!r} must be real")
        return float(metric)

    def _delta(key: str, field: str = "accuracy") -> float:
        return (_nested_metric(compiled, key, field)
                - _nested_metric(exact, key, field))

    return {
        "exact": exact,
        "compiled": compiled,
        "lexicon_holdout_delta": _delta("lexicon_holdout"),
        "roles_delta": _delta("roles"),
        "novel_composition_delta": _delta("novel_composition"),
        "bridge_oov_top5_delta": _delta("bridge_oov", "top5"),
        "bridge_seen_top5_delta": _delta("bridge_seen", "top5"),
        "dialogue_delta": _delta("dialogue"),
        "composite_delta": (_scalar_metric(compiled, "composite")
                            - _scalar_metric(exact, "composite")),
    }


def train_dialogue_with_holdouts(
    parser: "EmergentParser",
    schedule: "TrainingSchedule",
    holdout_words: Optional[Set[str]] = None,
) -> None:
    """Run a DIALOGUE schedule while skipping holdout words in lexicon training."""
    from ..evaluation.parity import run_dialogue_stage

    schedule.holdout_words = resolve_holdout_set(holdout_words)
    run_dialogue_stage(parser, schedule)


def run_generalization_gate(
    *,
    n: int = 3000,
    k: int = 30,
    seed: int = 42,
    holdout_words: Optional[Dict[str, str]] = None,
    max_bridge_probes: int = 20,
    qa_subset: Optional[int] = 15,
) -> Dict[str, object]:
    """Train exact vs compiled DIALOGUE with lexicon holdouts; compare generalization."""
    from ..vocabulary_builder import build_vocabulary_preset
    from ..curriculum.dialogue import get_dialogue_pairs
    from ..parser import EmergentParser
    from ..evaluation.parity import (
        build_dialogue_stage_schedule,
        exact_training_mode,
    )

    holdouts = holdout_words or dict(DEFAULT_LEXICON_HOLDOUTS)
    holdout_set = set(holdouts)
    vocab = build_vocabulary_preset("medium")
    qa_pairs = get_dialogue_pairs(seed=seed)
    if qa_subset is not None:
        qa_pairs = qa_pairs[:qa_subset]

    def _train(*, exact: bool) -> EmergentParser:
        parser = EmergentParser(
            n=n, k=k, seed=seed, vocabulary=vocab, fast_training=True,
        )
        schedule = build_dialogue_stage_schedule(parser, seed=seed)
        if exact:
            parser._compiled_training_enabled = False
            parser.brain.projection_fidelity = "exact"
            with exact_training_mode():
                train_dialogue_with_holdouts(parser, schedule, holdout_set)
        else:
            train_dialogue_with_holdouts(parser, schedule, holdout_set)
        return _finish_parser(parser)

    exact = _train(exact=True)
    compiled = _train(exact=False)
    return compare_generalization_parity(
        exact,
        compiled,
        holdout_words=holdouts,
        qa_pairs=qa_pairs,
        max_bridge_probes=max_bridge_probes,
        seed=seed,
    )


# ======================================================================
# Curriculum depth sweep
# ======================================================================

DEFAULT_DEPTH_CHECKPOINTS: Tuple[str, ...] = (
    "TWO_WORD",
    "SENTENCES",
    "DIALOGUE_FAST",
    "DIALOGUE_CUMULATIVE",
    "FULL_TRAIN",
)

_DEPTH_CONVERSATION_STAGE = {
    "DIALOGUE_FAST": ("DIALOGUE", True),
    "DIALOGUE_CUMULATIVE": ("DIALOGUE", False),
}


def _finish_parser(parser: "EmergentParser") -> "EmergentParser":
    """Structural build every depth needs, whichever branch produced it.

    CURRENTLY A NO-OP, and the reason is a measured regression rather than a
    change of mind.

    It called `_pregrow_phrase_pathways()`, which builds the phrase areas'
    self-fibers. That fixed a real defect on paper: VP's self-fiber had ZERO
    columns on the curriculum path, so the ERP P600 violation arm read exactly
    0.000000 and `1 - energy` was a constant 1.0 (#104). Pre-growth took it to
    120 columns on every training path.

    IT DID NOT WORK, AND IT MADE THINGS WORSE.

      * It does not reliably revive the arm. On a fresh n=3000 parser VP read
        0.000060; through the cached fixture it still read exactly 0.000000.
        VP's winners land at compact indices 71..100 -- 71 being EXACTLY the
        materialised count when the pre-growth ran -- i.e. the firing assembly
        sits outside whatever the bootstrap block covers.
      * It INVERTED a shipped contrast. Seed 42 SENTENCES went to
        `p600_auc = 0.444`, below the 0.5 null: violations scoring BELOW
        grammatical. Pre-growth recruits ~71 neurons that never win and
        displaces the real assembly, so it does not merely fail to help.

    Kept as a seam rather than deleted because #108 needs it: the underlying
    defect (a self-fiber that does not cover neurons recruited after it was
    built) is real and unfixed, and `_pregrow_phrase_pathways` is the harness
    for measuring it. Re-enable only alongside a fix that makes the block grow.
    """
    return parser


def train_parser_to_depth(
    depth: str,
    *,
    n: int = 3000,
    k: int = 30,
    beta: float = 0.05,
    p: float = 0.05,
    rounds: int = 10,
    phon_weight: float = 6.0,
    seed: int = 42,
    holdout_words: Optional[Set[str]] = None,
    vocabulary: Optional[dict] = None,
    fast_training: bool = True,
    engine: str = "auto",
) -> "EmergentParser":
    """Train a fresh parser to a named curriculum checkpoint.

    `beta`, `p` and `rounds` are exposed for the same reason `n` and `k` are:
    they are training parameters a study may vary. Defaults track
    `EmergentParser.__init__`. Anything added here MUST also enter the cache
    keys in `sweep.ParserCache` -- a training knob that is not cache-key
    material makes an A/B silently compare an arm against a cached copy of the
    other one.
    """
    from ..vocabulary_builder import build_vocabulary_preset
    from ..curriculum import CurriculumTrainer, _STAGE_CONFIG
    from ..parser import EmergentParser
    from ..curriculum.data import create_training_sentences
    from ..evaluation.parity import build_dialogue_stage_schedule

    holdout = resolve_holdout_set(holdout_words)
    vocab = vocabulary if vocabulary is not None else build_vocabulary_preset("medium")
    parser = EmergentParser(
        n=n, k=k, beta=beta, p=p, rounds=rounds, phon_weight=phon_weight,
        seed=seed, vocabulary=vocab, fast_training=fast_training, engine=engine,
    )

    if depth == "FULL_TRAIN":
        parser.train(
            sentences=create_training_sentences(),
            holdout_words=holdout,
            train_prediction=True,
        )
        return _finish_parser(parser)

    if depth == "DIALOGUE_ONLY":
        schedule = build_dialogue_stage_schedule(parser, seed=seed)
        train_dialogue_with_holdouts(parser, schedule, holdout)
        return _finish_parser(parser)

    if depth in _DEPTH_CONVERSATION_STAGE:
        stage_name, skip_early = _DEPTH_CONVERSATION_STAGE[depth]
        trainer = CurriculumTrainer(parser, holdout_words=holdout)
        trainer.train_conversation_path(
            max_stage=stage_name,
            skip_early_if_loaded=skip_early,
        )
        return _finish_parser(parser)

    if depth not in _STAGE_CONFIG:
        raise ValueError(
            f"unknown depth {depth!r}; expected one of "
            f"{DEFAULT_DEPTH_CHECKPOINTS + ('DIALOGUE_ONLY',)}"
        )

    trainer = CurriculumTrainer(parser, holdout_words=holdout)
    trainer.train_curriculum(max_stage=depth)
    return _finish_parser(parser)


def _metric_scalar(metrics: Dict[str, object], key: str, field: str = "accuracy") -> float:
    block = metrics[key]  # type: ignore[index]
    if field == "correct":
        return 1.0 if block.get("correct") else 0.0  # type: ignore[union-attr]
    return float(block[field])  # type: ignore[index]


def summarize_curriculum_sweep(sweep: Dict[str, object]) -> Dict[str, object]:
    """Derive cross-depth comparisons from a sweep report."""
    depths: Dict[str, dict] = sweep["depths"]  # type: ignore[assignment]
    rows: Dict[str, Dict[str, float]] = {}
    for name, entry in depths.items():
        m = entry["metrics"]
        rows[name] = {
            "lexicon_holdout": _metric_scalar(m, "lexicon_holdout"),
            "holdout_bootstrap": float(
                m.get("holdout_decomposition", {}).get("accuracy_bootstrapped", 0.0)  # type: ignore[union-attr]
            ),
            "roles": _metric_scalar(m, "roles"),
            "novel_composition": _metric_scalar(m, "novel_composition"),
            "bridge_seen_top5": _metric_scalar(m, "bridge_seen", "top5"),
            "bridge_oov_top5": _metric_scalar(m, "bridge_oov", "top5"),
            "dialogue": _metric_scalar(m, "dialogue"),
            "word_order_svo": _metric_scalar(m, "word_order", "correct"),
            "composite": float(m["composite"]),
            "train_seconds": float(entry.get("train_seconds", 0.0)),
        }

    def _best(metric: str) -> str:
        return max(rows, key=lambda d: rows[d][metric])

    ordered = list(depths.keys())
    monotonic_novel = all(
        rows[ordered[i]]["novel_composition"] <= rows[ordered[i + 1]]["novel_composition"]
        for i in range(len(ordered) - 1)
    ) if len(ordered) > 1 else True

    return {
        "rows": rows,
        "best_lexicon_holdout": _best("lexicon_holdout"),
        "best_novel_composition": _best("novel_composition"),
        "best_roles": _best("roles"),
        "best_bridge_seen_top5": _best("bridge_seen_top5"),
        "best_composite": _best("composite"),
        "novel_composition_monotonic_with_depth": monotonic_novel,
    }


def format_curriculum_sweep_table(sweep: Dict[str, object]) -> str:
    """Human-readable table of sweep metrics by curriculum depth."""
    summary = sweep.get("summary")
    if summary is None:
        summary = summarize_curriculum_sweep(sweep)
    if not isinstance(summary, Mapping):
        raise TypeError("curriculum sweep summary must be a mapping")
    rows_value = summary.get("rows")
    if not isinstance(rows_value, Mapping):
        raise TypeError("curriculum sweep summary rows must be a mapping")
    rows: Dict[str, Dict[str, float]] = {
        str(depth): values for depth, values in rows_value.items()
        if isinstance(depth, str) and isinstance(values, dict)
    }

    headers = (
        "depth",
        "lex_hold",
        "bootstrap",
        "roles",
        "novel",
        "bridge5",
        "oov5",
        "dialogue",
        "svo",
        "composite",
        "sec",
    )
    lines = [
        "Curriculum depth generalization sweep",
        f"seed={sweep.get('seed')} holdout={sweep.get('holdout_words')}",
        "",
        "  ".join(f"{h:>10}" for h in headers),
    ]
    for depth, r in rows.items():
        lines.append(
            "  ".join(
                [
                    f"{depth:>10}",
                    f"{r['lexicon_holdout']:>10.1%}",
                    f"{r['holdout_bootstrap']:>10.1%}",
                    f"{r['roles']:>10.1%}",
                    f"{r['novel_composition']:>10.1%}",
                    f"{r['bridge_seen_top5']:>10.1%}",
                    f"{r['bridge_oov_top5']:>10.1%}",
                    f"{r['dialogue']:>10.1%}",
                    f"{r['word_order_svo']:>10.1%}",
                    f"{r['composite']:>10.1%}",
                    f"{r['train_seconds']:>10.1f}",
                ]
            )
        )

    lines.extend([
        "",
        f"best novel composition: {summary.get('best_novel_composition')}",
        f"best roles: {summary.get('best_roles')}",
        f"best composite: {summary.get('best_composite')}",
        f"novel monotonic with depth order: {summary.get('novel_composition_monotonic_with_depth')}",
    ])
    return "\n".join(lines)


def run_curriculum_depth_sweep(
    *,
    depths: Optional[Sequence[str]] = None,
    n: int = 3000,
    k: int = 30,
    seed: int = 42,
    holdout_words: Optional[Dict[str, str]] = None,
    max_bridge_probes: int = 20,
    qa_subset: Optional[int] = 15,
    fast_training: bool = True,
) -> Dict[str, object]:
    """Train fresh parsers at each curriculum depth; measure generalization."""
    import time

    from ..curriculum.dialogue import get_dialogue_pairs

    holdouts = holdout_words or dict(DEFAULT_LEXICON_HOLDOUTS)
    holdout_set = set(holdouts)
    depth_list = list(depths or DEFAULT_DEPTH_CHECKPOINTS)

    qa_pairs = get_dialogue_pairs(seed=seed)
    if qa_subset is not None:
        qa_pairs = qa_pairs[:qa_subset]

    results: Dict[str, object] = {}
    for depth in depth_list:
        t0 = time.perf_counter()
        parser = train_parser_to_depth(
            depth,
            n=n,
            k=k,
            seed=seed,
            holdout_words=holdout_set,
            fast_training=fast_training,
        )
        train_seconds = time.perf_counter() - t0
        metrics = evaluate_generalization_metrics(
            parser,
            holdout_words=holdouts,
            qa_pairs=qa_pairs,
            max_bridge_probes=max_bridge_probes,
            seed=seed,
        )
        results[depth] = {
            "metrics": metrics,
            "train_seconds": train_seconds,
            "phases_hint": _depth_phases_hint(depth),
        }

    sweep = {
        "seed": seed,
        "n": n,
        "k": k,
        "fast_training": fast_training,
        "holdout_words": sorted(holdout_set),
        "depths": results,
    }
    sweep["summary"] = summarize_curriculum_sweep(sweep)
    return sweep


def _depth_phases_hint(depth: str) -> str:
    from ..curriculum import _STAGE_CONFIG

    if depth == "FULL_TRAIN":
        return "train() corpus + prediction"
    if depth == "DIALOGUE_ONLY":
        return "DIALOGUE stage schedule only"
    if depth == "DIALOGUE_FAST":
        return "skip early -> DIALOGUE"
    if depth == "DIALOGUE_CUMULATIVE":
        return "FIRST_WORDS -> DIALOGUE"
    if depth in _STAGE_CONFIG:
        return " -> ".join(_STAGE_CONFIG[depth]["phases"])
    return depth

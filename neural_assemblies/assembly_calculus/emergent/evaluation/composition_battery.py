"""Harder composition probes — discriminate training paths beyond easy ceiling.

A CEILING IS NOT A RESULT.  The default generalization probes saturate: every
training path scores at or near 100%, so they cannot tell a good path from a
merely adequate one.  A metric pinned at its maximum has zero variance and
therefore zero discriminative power, however impressive the number looks in
isolation.

This module exists to break that ceiling on purpose.  The probes here demand
composition the parser cannot have memorised -- held-out words appearing in
constructions they were never trained in, prefixes whose continuations are
constrained by structure rather than by co-occurrence.  Scores are expected to
be well below 100%, and that is the design: the useful quantity is the GAP
between paths, not the absolute value.

Read results from here accordingly.  A drop relative to the easy probes is not
a regression; it is the battery doing its job.  What would be a genuine
finding is two paths that differ on the easy probes but not on these, or vice
versa.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..parser import EmergentParser

# Easy probes (often saturate at 100% for all paths).
from .generalization import (
    DEFAULT_LEXICON_HOLDOUTS,
    NOVEL_COMPOSITION_PROBES,
    collect_bridge_probes,
    evaluate_generalization_metrics,
)
from .parity import (
    collect_transition_probes,
    score_holdout_constrained_probes,
    score_next_token_probes,
)

# Holdout-heavy, multi-token frames — stresses bootstrap + role assignment.
STRAIN_COMPOSITION_PROBES: List[dict] = [
    {
        "words": ["the", "small", "bird", "finds", "the", "dog"],
        "expected_roles": {
            "bird": "AGENT", "finds": "ACTION", "dog": "PATIENT",
        },
    },
    {
        "words": ["the", "dog", "finds", "the", "small", "bird"],
        "expected_roles": {
            "dog": "AGENT", "finds": "ACTION", "bird": "PATIENT",
        },
    },
    {
        "words": ["she", "finds", "the", "small", "bird"],
        "expected_roles": {
            "she": "AGENT", "finds": "ACTION", "bird": "PATIENT",
        },
    },
    {
        "words": ["the", "bird", "finds", "the", "small", "dog"],
        "expected_roles": {
            "bird": "AGENT", "finds": "ACTION", "dog": "PATIENT",
        },
    },
    {
        "words": ["the", "small", "dog", "chases", "the", "bird"],
        "expected_roles": {
            "dog": "AGENT", "chases": "ACTION", "bird": "PATIENT",
        },
    },
    {
        "words": ["the", "boy", "finds", "the", "small", "bird"],
        "expected_roles": {
            "boy": "AGENT", "finds": "ACTION", "bird": "PATIENT",
        },
    },
]

# Agent/patient swap on trained lemmas — tests structural binding, not lexicon.
SYSTEMATICITY_PROBES: List[dict] = [
    {
        "words": ["the", "cat", "chases", "the", "dog"],
        "expected_roles": {
            "cat": "AGENT", "chases": "ACTION", "dog": "PATIENT",
        },
    },
    {
        "words": ["the", "dog", "chases", "the", "cat"],
        "expected_roles": {
            "dog": "AGENT", "chases": "ACTION", "cat": "PATIENT",
        },
    },
    {
        "words": ["the", "bird", "finds", "the", "cat"],
        "expected_roles": {
            "bird": "AGENT", "finds": "ACTION", "cat": "PATIENT",
        },
    },
]


# Direct prefix → holdout next-token probes (matches OOV bridge eval).
DIRECT_BRIDGE_OOV_PROBES: List[dict] = [
    {"prefix": ["the", "small", "bird"], "expected": ["finds"]},
    {"prefix": ["the", "bird", "finds"], "expected": ["small", "the"]},
    {"prefix": ["the", "cat", "finds"], "expected": ["small", "the", "bird"]},
    {"prefix": ["she", "finds"], "expected": ["the", "small", "bird"]},
    {"prefix": ["the", "boy", "finds"], "expected": ["the", "small", "bird"]},
    {"prefix": ["the", "small", "dog"], "expected": ["chases", "runs"]},
    {"prefix": ["the", "dog", "finds"], "expected": ["the", "small", "bird"]},
]


def _score_direct_bridge_probes(parser) -> Dict[str, float]:
    from .parity import score_next_token_probes

    prefixes = [p["prefix"] for p in DIRECT_BRIDGE_OOV_PROBES]
    expected = [p["expected"] for p in DIRECT_BRIDGE_OOV_PROBES]
    return score_next_token_probes(parser, prefixes, expected)


def evaluate_composition_battery(
    parser: "EmergentParser",
    *,
    holdout_words: Optional[Dict[str, str]] = None,
    max_bridge_probes: int = 20,
    seed: int = 42,
) -> Dict[str, object]:
    """Science-oriented metrics: easy vs strain composition + bridge prediction."""
    from ..core.corpus_index import compile_corpus
    from ..curriculum.data import create_training_sentences
    from .suite import EvaluationSuite

    holdouts = holdout_words or dict(DEFAULT_LEXICON_HOLDOUTS)
    holdout_set = set(holdouts)
    suite = EvaluationSuite(parser)

    from ..curriculum.holdout_bridges import compile_holdout_bridge_probe_index

    corpus_index = compile_corpus(parser, create_training_sentences())
    bridge_index = compile_holdout_bridge_probe_index(parser, holdout_set)
    base = evaluate_generalization_metrics(
        parser,
        holdout_words=holdouts,
        corpus_index=corpus_index,
        max_bridge_probes=max_bridge_probes,
        seed=seed,
    )

    easy = suite.evaluate_roles(NOVEL_COMPOSITION_PROBES)
    strain = suite.evaluate_roles(STRAIN_COMPOSITION_PROBES)
    systematicity = suite.evaluate_roles(SYSTEMATICITY_PROBES)

    oov_prefixes, oov_expected = collect_bridge_probes(
        bridge_index, parser, holdout_set, oov_only=True, max_probes=max_bridge_probes,
    )
    seen_prefixes, seen_expected = collect_transition_probes(
        corpus_index, max_probes=max_bridge_probes,
    )
    bridge_oov = score_holdout_constrained_probes(
        parser, oov_prefixes, oov_expected, holdout_set, seed=seed,
    )
    bridge_seen = score_next_token_probes(parser, seen_prefixes, seen_expected)
    direct_bridge = _score_direct_bridge_probes(parser)

    easy_acc = float(easy["accuracy"])
    strain_acc = float(strain["accuracy"])
    sys_acc = float(systematicity["accuracy"])

    return {
        "novel_easy": easy_acc,
        "novel_strain": strain_acc,
        "systematicity": sys_acc,
        "strain_gap": easy_acc - strain_acc,
        "holdout_bootstrap": float(
            base.get("holdout_decomposition", {}).get("accuracy_bootstrapped", 0.0)  # type: ignore[union-attr]
        ),
        "lexicon_holdout": float(base["lexicon_holdout"]["accuracy"]),  # type: ignore[index]
        "roles": float(base["roles"]["accuracy"]),  # type: ignore[index]
        "bridge_oov_top5": float(bridge_oov["top5"]),
        "bridge_seen_top5": float(bridge_seen["top5"]),
        "bridge_direct_top5": float(direct_bridge["top5"]),
        "dialogue": float(base["dialogue"]["accuracy"]),  # type: ignore[index]
        "composite": float(base["composite"]),
        "science_score": (
            strain_acc * 0.30
            + sys_acc * 0.20
            + float(bridge_oov["top5"]) * 0.20
            + float(direct_bridge["top5"]) * 0.15
            + float(base.get("holdout_decomposition", {}).get("accuracy_bootstrapped", 0.0)) * 0.15  # type: ignore[union-attr]
        ),
        "probe_counts": {
            "easy": len(NOVEL_COMPOSITION_PROBES),
            "strain": len(STRAIN_COMPOSITION_PROBES),
            "systematicity": len(SYSTEMATICITY_PROBES),
            "bridge_oov": bridge_oov["total"],
            "bridge_seen": bridge_seen["total"],
            "bridge_direct": direct_bridge["total"],
        },
    }

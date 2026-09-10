"""Training performance helpers: engine selection and round budgets."""

from __future__ import annotations

import os
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..curriculum.data import GroundedSentence


def fast_training_enabled() -> bool:
    return os.environ.get("EMERGENT_FAST_TRAINING", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def sweep_mode_enabled() -> bool:
    return os.environ.get("EMERGENT_SWEEP_MODE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def force_gpu_enabled() -> bool:
    return os.environ.get("ASSEMBLIES_FORCE_GPU", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def resolve_engine(requested: str = "auto", n_hint: int = 0) -> str:
    """Pick the best available compute engine.

    Priority when ``requested == "auto"``:
    1. ``ASSEMBLIES_ENGINE`` env override (exact engine name)
    2. ``ASSEMBLIES_FORCE_GPU=1`` → ``torch_sparse`` if registered
    3. ``detect_best_engine(n_hint)`` — torch at n >= 1M neurons/area
    4. First registered among torch_sparse, cuda_implicit, cupy_sparse, numpy_sparse

    Install GPU stack: ``uv sync --extra gpu`` (CUDA torch on Linux/Windows).
  """
    if requested and requested != "auto":
        return requested

    explicit = os.environ.get("ASSEMBLIES_ENGINE", "").strip()
    if explicit:
        return explicit

    from neural_assemblies.core.backend import detect_best_engine
    from neural_assemblies.core.engine import ensure_engine, list_engines

    # Resolve against the ONE engine we expect before falling back to loading
    # every engine. detect_best_engine only returns torch_sparse at n >= 1M, so
    # ordinary CPU-sized runs settle on numpy_sparse here and never import torch
    # -- which profiling showed cost 9.1s of a 22.6s training run. Same engine is
    # chosen either way; only the import side effects differ.
    if force_gpu_enabled() and ensure_engine("torch_sparse"):
        return "torch_sparse"

    preferred = detect_best_engine(n_hint)
    if ensure_engine(preferred):
        return preferred

    available = list_engines()
    if preferred in available:
        return preferred

    for name in ("torch_sparse", "cuda_implicit", "cupy_sparse", "numpy_sparse"):
        if name in available:
            return name
    return "numpy_sparse"


_GPU_ENGINES = frozenset({"torch_sparse", "cuda_implicit", "cupy_sparse"})
_TORCH_SPARSE_N_THRESHOLD = 1_000_000


def warn_engine_scale_mismatch(
    engine: str,
    n_hint: int,
    *,
    where: str = "",
) -> None:
    """Warn when a GPU sparse engine is used below the torch_sparse crossover."""
    import sys

    if n_hint <= 0 or n_hint >= _TORCH_SPARSE_N_THRESHOLD:
        return
    if engine not in _GPU_ENGINES:
        return

    prefix = f"{where}: " if where else ""
    explicit = os.environ.get("ASSEMBLIES_ENGINE", "").strip()
    hint = ""
    if explicit:
        hint = (
            f" (ASSEMBLIES_ENGINE={explicit!r} — unset for "
            f"n<{_TORCH_SPARSE_N_THRESHOLD:,})"
        )
    elif force_gpu_enabled():
        hint = " (ASSEMBLIES_FORCE_GPU=1 — unset for small-n sweeps)"

    print(
        f"WARNING: {prefix}engine={engine} at n={n_hint:,} is much slower than "
        f"numpy_sparse at this scale; prefer n>={_TORCH_SPARSE_N_THRESHOLD:,} "
        f"for GPU CSR{hint}",
        file=sys.stderr,
        flush=True,
    )


def budget_rounds(
    training_rounds: int,
    *,
    inference_rounds: Optional[int] = None,
    bridge_rounds: Optional[int] = None,
    fast: bool = False,
) -> tuple[int, int, int]:
    """Return (training, inference, bridge) projection round counts."""
    train_r = training_rounds
    if fast:
        train_r = max(3, min(training_rounds, training_rounds * 2 // 3))

    infer_r = inference_rounds if inference_rounds is not None else max(3, train_r // 2)
    if bridge_rounds is not None:
        bridge_r = bridge_rounds
    elif fast:
        bridge_r = max(2, train_r // 2)
    else:
        bridge_r = max(3, train_r // 2)
    return train_r, infer_r, bridge_r


def sequence_rounds_per_step(
    training_rounds: int,
    inference_rounds: int,
    *,
    fast: bool = False,
) -> int:
    """Projection rounds per step in ``sequence_memorize`` (word-order phase)."""
    if fast:
        return inference_rounds
    return training_rounds


def dedupe_grounded_sentences(
    sentences: List["GroundedSentence"],
    stim_map: dict,
) -> List["GroundedSentence"]:
    """Drop duplicate grounded word sequences (keeps first occurrence)."""
    seen: set = set()
    out: List["GroundedSentence"] = []
    for sent in sentences:
        words = tuple(w for w in sent.words if w in stim_map)
        if len(words) < 2 or words in seen:
            continue
        seen.add(words)
        out.append(sent)
    return out


# Per-curriculum-stage round overrides (training projections).
#
# WHY THESE ARE SMALL (6-8), and why bumping them would HURT. Core and role
# areas each store MANY assemblies (one per word / per filler). Formation
# reinforcement trades off directly against how distinct those stored
# assemblies stay: measured under norm_init (n=10000, k=50, beta=0.1),
# distinctness of six co-stored assemblies is ~0.95 up to ~20 rounds and then
# COLLAPSES (0.16 at 40 rounds) as recurrence pulls every filler into the
# area's single dominant attractor and merges them. The prior-contradicting
# OVS role-binding probe confirms it end-to-end: 1.00 at 5-10 rounds, degrading
# to 0.50/0.33 as rounds/beta rise. So these low counts are CORRECT for the
# multi-assembly regime, not an under-tuning to be "fixed".
#
# A single-assembly area (one attractor, e.g. a coin or an FSM state) is the
# OPPOSITE regime: it wants DEEP reinforcement (>=30 rounds or beta>=0.3) to
# build a self-sustaining basin, and under-reinforcing it there is the actual
# failure. There is no global "right" number; it is per-area. See the
# reinforcement-tradeoff finding and _ROLE_BINDING_ROUNDS in parser_mixins/core.
STAGE_TRAINING_ROUNDS = {
    "BABBLE": 2,
    "FIRST_WORDS": 4,
    "VOCABULARY_SPURT": 5,
    "TWO_WORD": 6,
    "SENTENCES": 7,
    "COMPLEX_GRAMMAR": 8,
    "INSTRUCTIONS": 7,
    "DIALOGUE": 7,
    "CONVERSATION": 8,
}

# Sequence-memorization repetitions for word-order phase.
STAGE_WORD_ORDER_REPS = {
    "FIRST_WORDS": 1,
    "VOCABULARY_SPURT": 1,
    "TWO_WORD": 1,
    "SENTENCES": 2,
    "COMPLEX_GRAMMAR": 2,
    "DIALOGUE": 2,
    "CONVERSATION": 2,
}

# Skip full developmental stages when preset vocab is already this large.
PRESET_VOCAB_SKIP_THRESHOLD = 80


def developmental_curriculum_enabled() -> bool:
    """When True, always run babble + early grammar (no preset skip)."""
    return os.environ.get("EMERGENT_DEV_CURRICULUM", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def should_skip_early_curriculum(
    vocab_size: int,
    max_stage: str,
) -> bool:
    """Whether to jump to DIALOGUE/CONVERSATION and skip early stages."""
    if developmental_curriculum_enabled():
        return False
    return (
        vocab_size >= PRESET_VOCAB_SKIP_THRESHOLD
        and max_stage in ("DIALOGUE", "CONVERSATION")
    )

# Distributional / unsupervised passes per curriculum stage.
STAGE_DISTRIBUTIONAL_REPS = {
    "BABBLE": 1,
    "FIRST_WORDS": 1,
    "VOCABULARY_SPURT": 2,
    "TWO_WORD": 2,
    "SENTENCES": 3,
    "COMPLEX_GRAMMAR": 3,
    "INSTRUCTIONS": 2,
    "DIALOGUE": 2,
    "CONVERSATION": 3,
}


def stage_distributional_reps(stage_name: str, *, fast: bool = False) -> int:
    """Repetition count for distributional / unsupervised curriculum phases."""
    if sweep_mode_enabled():
        return 1
    base = STAGE_DISTRIBUTIONAL_REPS.get(stage_name, 3)
    if fast:
        return max(1, base - 1)
    return base


# Curriculum phases skipped when ``fast_training`` is enabled.
FAST_SKIP_STAGE_PHASES = {
    "SENTENCES": frozenset({"mood"}),
    "COMPLEX_GRAMMAR": frozenset({"mood"}),
    "DIALOGUE": frozenset({"word_order"}),
    "CONVERSATION": frozenset({"word_order", "tense", "polarity", "conjunctions"}),
}

# Extra phase skips when ``EMERGENT_SWEEP_MODE=1`` (throughput sweeps).
SWEEP_SKIP_STAGE_PHASES = {
    "SENTENCES": frozenset({"mood", "polarity"}),
    "TWO_WORD": frozenset(),
    "COMPLEX_GRAMMAR": frozenset({"mood", "polarity", "conjunctions"}),
}


def stage_training_rounds(stage_name: str, *, fast: bool = False) -> Optional[int]:
    """Training projection rounds for a curriculum stage (respects fast budget)."""
    base = STAGE_TRAINING_ROUNDS.get(stage_name)
    if base is None:
        return None
    if fast:
        train_r, _, _ = budget_rounds(base, fast=True)
        return train_r
    return base


def effective_stage_phases(
    stage_name: str,
    phases: list,
    *,
    fast: bool = False,
) -> list:
    """Return curriculum phases to run, applying fast-mode skips."""
    if not fast:
        return phases
    skip = FAST_SKIP_STAGE_PHASES.get(stage_name, frozenset())
    if sweep_mode_enabled():
        skip = skip | SWEEP_SKIP_STAGE_PHASES.get(stage_name, frozenset())
    return [p for p in phases if p not in skip]


# Consolidation replay passes during fast curriculum (after roles/phrases).
STAGE_CONSOLIDATION_PASSES = {
    "DIALOGUE": 1,
    "CONVERSATION": 1,
}


def stage_consolidation_passes(stage_name: str, *, fast: bool = False) -> int:
    """Consolidation replay passes for a curriculum stage."""
    if not fast:
        return 0
    return STAGE_CONSOLIDATION_PASSES.get(stage_name, 0)


# Extra repetitions of holdout-bridge sentences in prediction corpus.
STAGE_HOLDOUT_BRIDGE_REPS = {
    "SENTENCES": 12,
    "COMPLEX_GRAMMAR": 8,
    "DIALOGUE": 6,
}


def stage_holdout_bridge_reps(stage_name: str, *, fast: bool = False) -> int:
    """How many times to repeat holdout-bridge sentences before compile."""
    base = STAGE_HOLDOUT_BRIDGE_REPS.get(stage_name, 4)
    if fast:
        return max(4, base)
    return base * 2


def adaptive_rounds(
    base: int,
    word_freq: int,
    *,
    min_rounds: int = 2,
) -> int:
    """Scale projection rounds by corpus frequency (high-freq → fewer rounds).

    The budget is per PRESENTATION, and a frequent word gets many
    presentations.  Its assembly is therefore already deep by the time any one
    presentation runs, so the extra rounds spent re-settling an assembly that
    is already at its fixed point buy nothing -- total exposure, not
    per-presentation rounds, is what carved it out.  A rare word gets the full
    budget because a single presentation may be most of what it ever gets.

    ``min_rounds = 2`` is a floor, not a tuned value: below two rounds there is
    no recurrent step at all, so the assembly is whatever the feedforward
    input alone selected.

    The thresholds (20, 8) and the step size of one round are tuned, not
    derived.  Treat the ladder as a unit if you change it.
    """
    if word_freq >= 20:
        return max(min_rounds, base - 2)
    if word_freq >= 8:
        return max(min_rounds, base - 1)
    return base

"""
Base classes and utilities for scientific experiments.

Provides:
- ExperimentBase: Base class for all experiments
- ExperimentResult: Structured result container
- Utilities for reproducibility, logging, and result storage
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

from research.json_documents import load_document, write_new_document
from neural_assemblies.assembly_calculus.assembly import (
    chance_overlap as _canonical_chance_overlap,
    neuron_overlap as _canonical_neuron_overlap,
)
from neural_assemblies.core.index_spaces import NeuronIds
from neural_assemblies.assembly_calculus.metrics import jaccard_similarity


def _validate_execution_success(value):
    if type(value) is not bool:
        raise ValueError('success must be a boolean execution status, not a string or numeric verdict')


def reported_null_test(values, null):
    """Keep undefined test statistics explicit and serializable, never significant."""
    result = ttest_vs_null(values, null)
    if result.get('degenerate'):
        return {**result, 't': None, 'p': None, 'd': None}
    return result


def summarize_paired(values1, values2, *, seed_ids):
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#paired-study-reporting"""
    from neural_assemblies.diagnostics import ensemble_from_values, paired_delta
    left = ensemble_from_values(values1, keys=seed_ids)
    right = ensemble_from_values(values2, keys=seed_ids)
    values = list(paired_delta(left, right).values)
    return {"values": values, "summary": summarize(values),
            "test": reported_null_test(values, 0.)}


def effect_text(test):
    return f"undefined ({test['degenerate']})" if test['d'] is None else f"{test['d']:.1f}"


@dataclass
class ExperimentResult:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-legacy-execution-status

    Execution result; success does not mean that scientific adoption bars passed.
    """
    
    experiment_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    
    def __setattr__(self, name, value):
        if name == 'success':
            _validate_execution_success(value)
        super().__setattr__(name, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        _validate_execution_success(self.success)
        return asdict(self)
    
    def save(self, path: Path) -> None:
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-evidence-json

        Preserve existing evidence; unsupported or nonfinite values must be resolved
        by the experiment instead of silently converted to strings.
        """
        write_new_document(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> 'ExperimentResult':
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-evidence-json"""
        data = load_document(path)
        if not isinstance(data, dict) or 'success' not in data:
            raise ValueError('result document must explicitly contain its success status')
        return cls(**data)


class ExperimentBase(ABC):
    """Base class for all scientific experiments."""
    
    def __init__(
        self,
        name: str,
        seed: int = 42,
        results_dir: Optional[Path] = None,
        verbose: bool = True
    ):
        self.name = name
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        
        # Set up results directory
        if results_dir is None:
            results_dir = Path(__file__).parent.parent / "results"
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self._start_time: Optional[float] = None
    
    def log(self, message: str) -> None:
        """Log message if verbose mode is on."""
        if self.verbose:
            print(f"[{self.name}] {message}")
    
    @abstractmethod
    def run(self, **kwargs) -> ExperimentResult:
        """Run the experiment and return results."""
        pass
    
    def _start_timer(self) -> None:
        """Start timing the experiment."""
        self._start_time = time.perf_counter()
    
    def _stop_timer(self) -> float:
        """Stop timing and return duration in seconds."""
        if self._start_time is None:
            return 0.0
        duration = time.perf_counter() - self._start_time
        self._start_time = None
        return duration
    
    def save_result(
        self,
        result: ExperimentResult,
        suffix: str = "",
        *,
        tag: str | None = None,
    ) -> Path:
        """Save an immutable experiment result.

        ``tag`` is an optional run identity for compatibility with the legacy
        experiment classes.  When supplied it is included in the filename and
        restricted to one path-safe component; the exclusive JSON writer still
        refuses replacement.  New studies should use :func:`research.runner.run_experiment`,
        where tags are required and provenance is captured automatically.
        """
        if tag is not None:
            if not isinstance(tag, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag
            ):
                raise ValueError(
                    "tag must be a nonempty path-safe name (letters, digits, dot, dash, underscore)"
                )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag_suffix = f"_{tag}" if tag is not None else ""
        filename = f"{self.name}_{timestamp}{tag_suffix}{suffix}.json"
        path = self.results_dir / filename
        result.save(path)
        self.log(f"Results saved to {path}")
        return path


def measure_overlap(winners_a: np.ndarray, winners_b: np.ndarray) -> float:
    """Compatibility name for the canonical assembly overlap measurement."""
    return _canonical_neuron_overlap(
        NeuronIds(np.asarray(winners_a)), NeuronIds(np.asarray(winners_b))
    )


def measure_jaccard(winners_a: np.ndarray, winners_b: np.ndarray) -> float:
    """Compatibility name for the canonical set Jaccard kernel."""
    return jaccard_similarity(winners_a, winners_b)


def convergence_metric(history: List[np.ndarray]) -> Dict[str, Any]:
    """
    Analyze convergence of assembly over projection rounds.
    
    Args:
        history: List of winner arrays over time
    
    Returns:
        Dictionary with convergence metrics
    """
    if len(history) < 2:
        return {"converged": False, "steps": 0, "final_stability": 0.0}
    
    # Calculate overlap between consecutive rounds
    overlaps = []
    for i in range(1, len(history)):
        overlap = measure_overlap(history[i-1], history[i])
        overlaps.append(overlap)
    
    # Find convergence point (overlap > 0.95 for 3 consecutive rounds)
    converged = False
    convergence_step = len(history)
    stability_threshold = 0.95
    
    for i in range(len(overlaps) - 2):
        if all(o >= stability_threshold for o in overlaps[i:i+3]):
            converged = True
            convergence_step = i + 1
            break
    
    return {
        "converged": converged,
        "convergence_step": convergence_step,
        "final_stability": overlaps[-1] if overlaps else 0.0,
        "overlap_history": overlaps,
        "mean_overlap": np.mean(overlaps) if overlaps else 0.0,
        "std_overlap": np.std(overlaps) if overlaps else 0.0,
    }


# -- Statistical helpers (shared across all experiments) -----------------------


def chance_overlap(k: int, n: int) -> float:
    """Expected overlap between two random k-subsets of [n].

    If A and B are independent uniform random k-subsets, then
    E[|A ∩ B|] / k = k / n  (hypergeometric mean / k).
    """
    return _canonical_chance_overlap(k, n)


def summarize(values: List[float]) -> Dict[str, float]:
    """Compatibility shape for the canonical per-seed Student-t ensemble.

    Requires at least three finite values, like diagnostics.ensemble_from_values.
    Historical artifacts used a normal interval here; they are not rewritten.
    """
    from neural_assemblies.diagnostics import ensemble_from_values

    result = ensemble_from_values(values)
    arr = np.asarray(result.values)
    n = len(result.values)
    mean = result.mean
    std = float(np.std(arr, ddof=1))
    sem = std / np.sqrt(n)
    ci95 = result.ci
    return {
        "mean": mean, "std": std, "sem": sem,
        "ci95_lo": mean - ci95, "ci95_hi": mean + ci95,
        "min": float(np.min(arr)), "max": float(np.max(arr)), "n": n,
    }


def ttest_vs_null(values: List[float], null_mean: float) -> Dict[str, Any]:
    """One-sample t-test against null mean. Returns t, p, Cohen's d."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return {"t": float("nan"), "p": float("nan"), "d": float("nan"),
                "significant": False, "degenerate": "too_few_samples"}
    if np.std(arr, ddof=1) == 0:
        # Zero variance is saturation, not evidence. Reporting p=0 here made
        # every ceiling-bound metric look maximally significant.
        at_null = bool(np.isclose(float(np.mean(arr)), null_mean))
        return {"t": float("nan"), "p": float("nan"), "d": float("nan"),
                "significant": False,
                "degenerate": "at_null" if at_null else "zero_variance"}
    t_stat, p_val = stats.ttest_1samp(arr, null_mean)
    t_value = float(np.asarray(t_stat).item())
    p_value = float(np.asarray(p_val).item())
    d = (np.mean(arr) - null_mean) / np.std(arr, ddof=1)
    return {"t": t_value, "p": p_value, "d": float(d),
            "significant": bool(p_value < 0.05)}


def paired_ttest(values1: List[float], values2: List[float]) -> Dict[str, Any]:
    """Paired t-test between two matched conditions. Returns t, p, Cohen's d."""
    arr1 = np.asarray(values1, dtype=float)
    arr2 = np.asarray(values2, dtype=float)
    diff = arr1 - arr2
    if len(diff) < 2 or np.std(diff, ddof=1) == 0:
        return {"t": 0.0, "p": 1.0, "d": 0.0, "significant": False}
    t_stat, p_val = stats.ttest_rel(arr1, arr2)
    t_value = float(np.asarray(t_stat).item())
    p_value = float(np.asarray(p_val).item())
    d = float(np.mean(diff) / np.std(diff, ddof=1))
    return {"t": t_value, "p": p_value, "d": d,
            "significant": bool(p_value < 0.05)}


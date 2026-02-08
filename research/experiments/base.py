"""
Base classes and utilities for scientific experiments.

Provides:
- ExperimentBase: Base class for all experiments
- ExperimentResult: Structured result container
- Utilities for reproducibility, logging, and result storage
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats


@dataclass
class ExperimentResult:
    """Container for experiment results with metadata."""
    
    experiment_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def save(self, path: Path) -> None:
        """Save result to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentResult':
        """Load result from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
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
    
    def save_result(self, result: ExperimentResult, suffix: str = "") -> Path:
        """Save experiment result to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_{timestamp}{suffix}.json"
        path = self.results_dir / filename
        result.save(path)
        self.log(f"Results saved to {path}")
        return path


def measure_overlap(winners_a: np.ndarray, winners_b: np.ndarray) -> float:
    """
    Measure overlap between two assemblies as fraction of shared neurons.
    
    Args:
        winners_a: Array of neuron indices in assembly A
        winners_b: Array of neuron indices in assembly B
    
    Returns:
        Overlap ratio (intersection / min(len_a, len_b))
    """
    if len(winners_a) == 0 or len(winners_b) == 0:
        return 0.0
    
    set_a = set(winners_a.tolist() if isinstance(winners_a, np.ndarray) else winners_a)
    set_b = set(winners_b.tolist() if isinstance(winners_b, np.ndarray) else winners_b)
    
    intersection = len(set_a & set_b)
    min_size = min(len(set_a), len(set_b))
    
    return intersection / min_size if min_size > 0 else 0.0


def measure_jaccard(winners_a: np.ndarray, winners_b: np.ndarray) -> float:
    """
    Measure Jaccard similarity between two assemblies.
    
    Args:
        winners_a: Array of neuron indices in assembly A
        winners_b: Array of neuron indices in assembly B
    
    Returns:
        Jaccard index (intersection / union)
    """
    if len(winners_a) == 0 and len(winners_b) == 0:
        return 1.0
    if len(winners_a) == 0 or len(winners_b) == 0:
        return 0.0
    
    set_a = set(winners_a.tolist() if isinstance(winners_a, np.ndarray) else winners_a)
    set_b = set(winners_b.tolist() if isinstance(winners_b, np.ndarray) else winners_b)
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / union if union > 0 else 0.0


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
    return k / n


def summarize(values: List[float]) -> Dict[str, float]:
    """Compute mean, SEM, 95% CI, and range across seeds."""
    arr = np.array(values)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    ci95 = 1.96 * sem
    return {
        "mean": mean, "std": std, "sem": sem,
        "ci95_lo": mean - ci95, "ci95_hi": mean + ci95,
        "min": float(np.min(arr)), "max": float(np.max(arr)), "n": n,
    }


def ttest_vs_null(values: List[float], null_mean: float) -> Dict[str, Any]:
    """One-sample t-test against null mean. Returns t, p, Cohen's d."""
    arr = np.array(values)
    if len(arr) < 2 or np.std(arr, ddof=1) == 0:
        return {"t": float("inf"), "p": 0.0, "d": float("inf"), "significant": True}
    t_stat, p_val = stats.ttest_1samp(arr, null_mean)
    d = (np.mean(arr) - null_mean) / np.std(arr, ddof=1)
    return {"t": float(t_stat), "p": float(p_val), "d": float(d),
            "significant": p_val < 0.05}


def paired_ttest(values1: List[float], values2: List[float]) -> Dict[str, Any]:
    """Paired t-test between two matched conditions. Returns t, p, Cohen's d."""
    arr1 = np.array(values1)
    arr2 = np.array(values2)
    diff = arr1 - arr2
    if len(diff) < 2 or np.std(diff, ddof=1) == 0:
        return {"t": 0.0, "p": 1.0, "d": 0.0, "significant": False}
    t_stat, p_val = stats.ttest_rel(arr1, arr2)
    d = float(np.mean(diff) / np.std(diff, ddof=1))
    return {"t": float(t_stat), "p": float(p_val), "d": d,
            "significant": p_val < 0.05}


"""One learning-on stopping phase; callers own initialization and evaluation."""
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
from scipy import stats
from neural_assemblies.assembly_calculus.assembly import Assembly
from neural_assemblies.core.registration import validate_round_count
from neural_assemblies.diagnostics import ensemble_from_values
from research.experiments.base import measure_overlap


def validate_convergence_rule(max_rounds, window, threshold):
    validate_round_count(max_rounds)
    validate_round_count(window)
    if (isinstance(threshold, bool) or not isinstance(threshold, Real) or
            not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("convergence threshold must be finite and in [0, 1]")


@dataclass(frozen=True)
class ConvergenceObservation:
    assembly: Assembly
    training_rounds: int
    converged: bool

    def record(self):
        return {"training_rounds": self.training_rounds, "converged": self.converged,
                "convergence_time": self.training_rounds if self.converged else None}


def run_convergence_phase(brain, *, stimulus, area, max_rounds, window=3, threshold=.98):
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#shared-convergence-phase"""
    validate_convergence_rule(max_rounds, window, threshold)
    previous = None
    streak = 0
    for count in range(1, max_rounds + 1):
        brain.project({stimulus: [area]}, {area: [area]})
        current = Assembly.from_area(brain, area)
        if previous is not None:
            agrees = measure_overlap(previous.neuron_ids, current.neuron_ids) > threshold
            streak = streak + 1 if agrees else 0
        if streak >= window:
            return ConvergenceObservation(current, count, True)
        previous = current
    return ConvergenceObservation(current, max_rounds, False)


def convergence_scaling_fit(sizes, event_times):
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#shared-convergence-phase

    Descriptive linear fit in log10(size), never an asymptotic complexity class.
    """
    if len(sizes) < 2 or len(sizes) != len(event_times) or len(set(sizes)) != len(sizes):
        raise ValueError("fit requires at least two distinct matched sizes")
    for size, times in zip(sizes, event_times):
        if size <= 0 or not math.isfinite(size) or len(times) < 3:
            raise ValueError("fit requires positive sizes and at least three observations per size")
        for time in times:
            if time is not None:
                validate_round_count(time)
    if any(time is None for times in event_times for time in times):
        return dict(slope=None, intercept=None, r_squared=None, p_value=None,
                    degenerate="censored_observations", equation=None)
    means = np.array([ensemble_from_values(times).mean for times in event_times])
    if np.ptp(means) == 0:
        return dict(slope=0., intercept=float(means[0]), r_squared=None, p_value=None,
                    degenerate="constant_response", equation=f"T = {float(means[0]):.2f}")
    slope, intercept, r_value, p_value, _ = stats.linregress(np.log10(sizes), means)
    return dict(slope=float(slope), intercept=float(intercept), r_squared=float(r_value**2),
                p_value=float(p_value), equation=f"T = {slope:.2f} * log10(n) + {intercept:.2f}")

"""Resolve experiment inputs before measurement or source capture."""
from numbers import Integral, Real
import math


def resolve_seed_ids(n_seeds=None, seed_ids=None, *, base_seed=42, default_count=10):
    """Resolve ordered independent brain identities without re-offsetting explicit seeds."""
    if n_seeds is None:
        n_seeds = len(seed_ids) if seed_ids is not None else default_count
    if isinstance(n_seeds, bool) or not isinstance(n_seeds, Integral) or n_seeds < 3:
        raise ValueError('historical study summaries require at least three integer seed identities')
    seeds = list(seed_ids) if seed_ids is not None else [base_seed + i for i in range(n_seeds)]
    if (len(seeds) != n_seeds or any(isinstance(s, bool) or not isinstance(s, Integral) or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('seed identities must be unique nonnegative integers matching n_seeds')
    return [int(s) for s in seeds]



def resolve_real_grid(values, *, name, minimum=0., maximum=None):
    """Resolve a finite, ordered, unique numeric grid before constructing trials."""
    resolved = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} requires real numeric values")
        value = float(value)
        if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
            raise ValueError(f"{name} contains a value outside its finite domain")
        resolved.append(value)
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must be nonempty and unique")
    return resolved

"""Exact dyadic audit of a canonical top-k readout, outside the kernel hot path."""
from dataclasses import dataclass
from numbers import Integral
import math

import numpy as np


@dataclass(frozen=True)
class WinnerComparison:
    """Integer units represent values divided by 2**scale_exponent."""
    reference_winners: tuple[int, ...]
    candidate_winners: tuple[int, ...]
    scale_exponent: int
    max_error_units: int
    margin_units: int | None

    @property
    def winners_agree(self) -> bool:
        return self.reference_winners == self.candidate_winners

    @property
    def margin_certified(self) -> bool:
        # Empty/all-selected sets have no opposing candidate.
        return self.margin_units is None or self.margin_units > 2 * self.max_error_units


def compare_winner_selection(reference, candidate, k: int) -> WinnerComparison:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-winner-margin

    Compare equal-length finite binary-float/integer score vectors. Every input
    is represented exactly as an integer with a common power-of-two denominator;
    neither subtraction overflow nor rounded tolerances can certify a false gap.
    Winners use descending score and ascending index for ties. This checks one
    observed score pair, not future rounds, backend tie rules, or scientific merit.
    """
    arrays = [np.asarray(value, dtype=object) for value in (reference, candidate)]
    if any(a.ndim != 1 for a in arrays):
        raise ValueError("scores must be finite one-dimensional binary floats or integers")
    if arrays[0].shape != arrays[1].shape:
        raise ValueError("score vectors must have the same shape")
    n = len(arrays[0])
    if isinstance(k, bool) or not isinstance(k, Integral) or not 0 <= k <= n:
        raise ValueError("k must be an integer between zero and the score count")
    def ratio(value):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("boolean scores are not supported")
        if isinstance(value, Integral):
            return int(value), 1
        if (isinstance(value, (float, np.floating))
                and (not isinstance(value, np.floating) or value.dtype.itemsize <= 8)
                and math.isfinite(value)):
            return float(value).as_integer_ratio()
        raise ValueError("scores must be finite binary floats or integers")
    ratios = [[ratio(v) for v in a] for a in arrays]
    exponent = max((d.bit_length() - 1 for row in ratios for _, d in row), default=0)
    scaled = [[num << (exponent - (den.bit_length() - 1)) for num, den in row]
              for row in ratios]
    orders = [sorted(range(n), key=lambda i: (-row[i], i)) for row in scaled]
    margin = (scaled[0][orders[0][k - 1]] - scaled[0][orders[0][k]]
              if 0 < k < n else None)
    return WinnerComparison(
        tuple(sorted(orders[0][:k])), tuple(sorted(orders[1][:k])), exponent,
        max((abs(a - b) for a, b in zip(*scaled)), default=0), margin)

"""Where a golden number came from, readable at the assertion that uses it.

`assert w <= 4000` tells you nothing about whether 4000 is a claim from a
published paper or a number this repo recorded by running its own code last
month. Those two failures mean opposite things:

  * a TIER A golden failing is evidence about the substrate;
  * a TIER C golden failing may only mean the substrate changed, and the
    constant is now pinning an artifact of the code that produced it.

The 2026-07-31 audit found exactly one of 23 goldens came from a paper. So the
default reading of a failing threshold in this repo should be "which of these
two is it?", and that question is unanswerable at the assertion site unless the
answer is carried there.

Wrapping the constant does that without changing its value or its type:

    assert w <= golden(4000, "C", "our own sampler, pre-2026-07 -- reads 3505")

    AssertionError: assert 4273 <= 4000 [tier C: our own sampler,
                                         pre-2026-07 -- reads 3505]

`Golden` subclasses `int`/`float`, so it compares, formats and serializes
exactly like the bare number everywhere else. Only `repr` differs, which is
what pytest prints.

Tiers match `research/literature/parity/audit_golden_provenance.py`:

    A  quoted from the paper (verbatim, with the figure or table)
    B  produced by the authors' released reference code
    C  recorded by running THIS repo -- a regression pin, not evidence
"""

from __future__ import annotations

from typing import Union

__all__ = ["GoldenInt", "GoldenFloat", "golden", "TIERS"]

#: tier -> what a failure means.
TIERS = {
    "A": "from the paper -- a failure is evidence about the substrate",
    "B": "from the authors' reference code -- a failure is a parity divergence",
    "C": "recorded from this repo -- a failure may only mean the code changed",
}


class _GoldenMixin:
    """Carries provenance through arithmetic-free use; see `golden`."""

    tier: str
    source: str

    def _tag(self) -> str:
        return f"[tier {self.tier}: {self.source}]"

    def __repr__(self) -> str:                      # what pytest prints
        base = int.__repr__(self) if isinstance(self, int) else float.__repr__(self)
        return f"{base} {self._tag()}"

    @property
    def meaning(self) -> str:
        """What a failure against this constant actually implies."""
        return TIERS.get(self.tier, "unknown tier")


class GoldenInt(_GoldenMixin, int):
    def __new__(cls, value, tier: str, source: str):
        obj = int.__new__(cls, value)
        obj.tier, obj.source = tier, source
        return obj


class GoldenFloat(_GoldenMixin, float):
    def __new__(cls, value, tier: str, source: str):
        obj = float.__new__(cls, value)
        obj.tier, obj.source = tier, source
        return obj


def golden(value: Union[int, float], tier: str,
           source: str) -> Union[GoldenInt, GoldenFloat]:
    """Tag a threshold with its provenance. Returns a number, not a wrapper.

    Args:
        value: the constant, unchanged.
        tier: ``"A"``, ``"B"`` or ``"C"`` -- see `TIERS`.
        source: where it came from, specific enough to re-derive. For tier C
            say WHAT recorded it and WHEN, and give the value it recorded, so a
            later reader can tell a drifted substrate from a broken one.

    The value is returned unmodified and compares identically; only `repr`
    carries the tag, so an assertion failure explains its own authority.
    """
    tier = str(tier).upper()
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {sorted(TIERS)}, got {tier!r}")
    if not source or not str(source).strip():
        raise ValueError(
            "a golden needs a source; an untraceable constant is the thing "
            "this helper exists to prevent")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"golden() takes an int or float, got {type(value)}")
    return (GoldenInt(value, tier, source) if isinstance(value, int)
            else GoldenFloat(value, tier, source))

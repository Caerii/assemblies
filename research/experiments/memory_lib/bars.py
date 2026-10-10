"""Bars as DATA, judged by confidence bounds.

A registered bar used to exist three times -- a sentence in the registration, an ``evaluate``
function, a scorecard row -- and the three could drift. Here a bar is one object:

    Bar(id, name, statement, checks)     passes when every check passes at every cell
    Check(read, op, threshold)           read a cell's observations, compare

and what it READS decides how it is JUDGED:

    brains(path)            per-brain values      the mean's 95% CONFIDENCE BOUND against the
                                                  threshold (the lower bound for >=, the upper
                                                  for <=), via diagnostics.ensemble_from_values
    delta(path_a, path_b)   per-brain a - b       the same, on the paired differences (the
                                                  test an A/B asks; diagnostics.paired_delta)
    below(path, level)      a count of brains     the count itself: no interval is claimed
    scalar(path)            one pooled number     the number itself
    every(path_a, path_b)   per-brain a - b       every brain, not the mean

WHY THE BOUND. Amendments 48-55 judged "mean >= threshold" on twenty brains: a point estimate
that clears a bar by less than its own interval is a draw, not a result (the methodology
ratchet's lesson, neural_assemblies/tests/test_methodology_ratchet.py). ``judge="mean"`` keeps
the old reading, so a recorded verdict can be reproduced and set beside the bounded one.

Paths name a value inside a cell's observations with "/" (``"reuse/median"``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

_OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b, "<": lambda a, b: a < b}


def _at(obs, path):
    for part in path.split("/"):
        obs = obs[part]
    return obs


@dataclass(frozen=True)
class Reading:
    kind: str                   # "brains", "count", "scalar" or "every"
    values: Any                 # a tuple of per-brain values, or one number
    label: str


def brains(path):
    return lambda cell: Reading("brains", tuple(float(v) for v in _at(cell, path)), path)


def delta(path_a, path_b):
    def read(cell):
        a, b = _at(cell, path_a), _at(cell, path_b)
        if len(a) != len(b):
            raise ValueError(f"{path_a} and {path_b} hold different numbers of brains")
        return Reading("brains", tuple(float(x) - float(y) for x, y in zip(a, b)), f"{path_a} - {path_b}")
    return read


def below(path, level):
    return lambda cell: Reading("count", sum(float(v) < level for v in _at(cell, path)),
                                f"brains with {path} < {level:g}")


def scalar(path):
    return lambda cell: Reading("scalar", float(_at(cell, path)), path)


def every(path_a, path_b=None):
    """per-brain a (or a - b), judged on EVERY brain"""
    def read(cell):
        a = _at(cell, path_a)
        b = _at(cell, path_b) if path_b else [0.0] * len(a)
        return Reading("every", tuple(float(x) - float(y) for x, y in zip(a, b)),
                       path_a if path_b is None else f"{path_a} - {path_b}")
    return read


@dataclass(frozen=True)
class Check:
    read: Callable[[dict], Reading]
    op: str                     # ">=", "<=", ">" or "<"
    threshold: float

    def judge(self, cell, how="bound"):
        """{"pass", "value", and for brains "mean", "ci", "low", "high"} at one cell"""
        r = self.read(cell)
        cmp = _OPS[self.op]
        if r.kind == "brains":
            from neural_assemblies.diagnostics import ensemble_from_values
            e = ensemble_from_values(r.values, r.label)
            bound = e.low if self.op in (">=", ">") else e.high
            value = e.mean if how == "mean" else bound
            return {"pass": bool(cmp(value, self.threshold)), "label": r.label, "mean": e.mean,
                    "ci": e.ci, "low": e.low, "high": e.high, "judged": how}
        if r.kind == "every":
            worst = min(r.values) if self.op in (">=", ">") else max(r.values)
            return {"pass": bool(cmp(worst, self.threshold)), "label": r.label, "worst": worst,
                    "failing": sum(not cmp(v, self.threshold) for v in r.values)}
        return {"pass": bool(cmp(r.values, self.threshold)), "label": r.label, "value": r.values}

    def describe(self):
        return f"{self.op} {self.threshold:g}"


@dataclass(frozen=True)
class Bar:
    id: str                     # "R2"
    name: str                   # "REACH"
    statement: str              # the registration's sentence
    checks: tuple = field(default_factory=tuple)

    def judge(self, cells, how="bound"):
        """``cells`` {cell key: observations}: passes when every check passes at every cell"""
        out = {key: [c.judge(obs, how) for c in self.checks] for key, obs in cells.items()}
        return {"pass": all(r["pass"] for rs in out.values() for r in rs), "cells": out}

    def text(self):
        """the bar as the registration states it"""
        return f"    {self.id}  {self.name}. {self.statement}"


def evaluate(bars, cells, how="bound"):
    """{bar id: {"pass", "cells"}} for every bar at every cell; ``how`` "bound" (the default and
    the rule for new registrations) or "mean" (the reading Amendments 48-55 used)"""
    return {b.id: b.judge(cells, how) for b in bars}

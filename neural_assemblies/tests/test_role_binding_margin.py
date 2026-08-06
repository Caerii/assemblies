"""`_role_binding_margin` must report ONE quantity, or say it has none.

THE DEFECT (task #110). The function returned two different things under one
name:

    if not others:
        return own                                        # RAW OVERLAP
    base = sum(others) / len(others)
    return max(0.0, (own - base) / max(1e-6, 1.0 - base))  # RESIDUAL

Its own docstring explains why the raw overlap is useless -- "~0.9 for EVERY
candidate role", "nearly independent of what was learned" -- which is the whole
reason the residual exists. So the single-filler case returned precisely the
quantity the function was written to avoid, and it returned it LARGE.

That matters because `_assign_roles_neural` normalizes these margins against
each other into a distribution over candidate roles. A raw ~0.9 competing with
properly-baselined residuals takes probability mass it did not earn, which makes
LEXICON SIZE move the role decision.

Measured before changing anything (`research/experiments/
role_margin_branch_census.py`): across seeds 11/12/42, the single-filler branch
fires **0 times** and the residual branch 83.3%, so the fix was free -- the
post-fix margins are identical to five decimal places. These tests exist so it
stays that way: the branch is dead, not absent, and dead branches come back.
"""
import contextlib

import numpy as np
import pytest

from neural_assemblies.assembly_calculus.emergent.parser_mixins import roles
from neural_assemblies.assembly_calculus.emergent.parser_mixins.roles import (
    RoleBindingMixin)
from neural_assemblies.core.measurement import UndefinedMeasurement


class _Brain:
    """Enough brain to reach the branch logic, and nothing more."""

    @contextlib.contextmanager
    def frozen(self):
        yield

    def project(self, stimuli, areas):
        return None


class _Parser(RoleBindingMixin):
    def __init__(self, role_lexicons):
        self.role_lexicons = role_lexicons
        self.brain = _Brain()


@pytest.fixture
def read_returns(monkeypatch):
    """Pin what the probe reads back, so only the BRANCH is under test."""
    def _set(assembly):
        monkeypatch.setattr(roles, "_snap", lambda brain, area: assembly)
    return _set


def _Asm(ids):
    """Stored assemblies are plain neuron-ID arrays -- what `overlap` takes."""
    return np.array(list(ids), dtype=np.int64)


def _margin(parser, word, assembly, read_returns):
    read_returns(assembly)
    return parser._role_binding_margin(word, "NOUN_CORE", "ROLE_AGENT")


# --------------------------------------------------------------------------
# the branch that used to return the wrong quantity
# --------------------------------------------------------------------------

def test_a_lone_filler_has_no_baseline_so_the_residual_is_undefined(read_returns):
    """The residual is `own - base`. With no other filler there is no base."""
    stored = _Asm(range(10))
    parser = _Parser({"ROLE_AGENT": {"dog": stored}})

    margin = _margin(parser, "dog", stored, read_returns)

    assert not margin.defined
    assert "no baseline" in margin.why
    assert (margin.detail or {}).get("own") == pytest.approx(1.0), (
        "the raw overlap is still reported in `detail` -- it is not lost, it "
        "is just no longer passed off as a residual")


def test_the_lone_filler_case_does_not_return_the_raw_overlap_as_a_margin(
        read_returns):
    """The specific regression: a perfect self-match must not read as ~1.0.

    This is what made lexicon size move the role decision.
    """
    stored = _Asm(range(10))
    parser = _Parser({"ROLE_AGENT": {"dog": stored}})

    margin = _margin(parser, "dog", stored, read_returns)

    with pytest.raises(UndefinedMeasurement):
        float(margin)
    assert margin.or_else(0.0) == 0.0, (
        "an undefined margin must contribute NOTHING to the role distribution, "
        "not the near-1.0 raw overlap it used to contribute")


# --------------------------------------------------------------------------
# the other direction: the intended quantity is unchanged
# --------------------------------------------------------------------------

def test_the_residual_is_still_computed_when_there_is_a_baseline(read_returns):
    """With a competitor present the arithmetic is exactly what it always was."""
    stored = _Asm(range(10))
    other = _Asm(range(5, 15))          # overlaps `stored` by 5 of 10
    parser = _Parser({"ROLE_AGENT": {"dog": stored, "cat": other}})

    margin = _margin(parser, "dog", stored, read_returns)

    assert margin.defined
    # own = 1.0, base = 0.5  ->  (1.0 - 0.5) / (1.0 - 0.5) = 1.0
    assert float(margin) == pytest.approx(1.0)


def test_a_projection_no_better_than_the_baseline_scores_zero(read_returns):
    """A DEFINED zero is a real reading, and stays distinct from an undefined one."""
    stored = _Asm(range(10))
    other = _Asm(range(10))             # identical, so own == base
    parser = _Parser({"ROLE_AGENT": {"dog": stored, "cat": other}})

    margin = _margin(parser, "dog", stored, read_returns)

    assert margin.defined and float(margin) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# no stored binding at all
# --------------------------------------------------------------------------

def test_an_untrained_word_has_no_evidence_rather_than_evidence_of_zero(
        read_returns):
    parser = _Parser({"ROLE_AGENT": {"cat": _Asm(range(10))}})

    margin = _margin(parser, "dog", _Asm(range(10)), read_returns)

    assert not margin.defined
    assert "no stored binding" in margin.why
    assert margin.or_else(0.0) == 0.0, (
        "the caller's default is unchanged -- this branch was ALREADY 0.0, "
        "which is why the migration was behaviour-preserving")

"""An out-of-regime organ must WARN, not merely print a table.

WHY THIS EXISTS. `regime_audit` has computed the `k*p >= 3 ln n` verdict for
months and four experiments called it -- and it only ever PRINTED. At the
bottom of a run log a printed row is indistinguishable from silence, and it
cost this project twice in one session:

  * the S5 word-problem organ printed `kp 28.0 vs floor 29.7` on every run;
    the violation went unacted-on until a whole study was built to explain the
    soft defects it caused, and clearing the floor removed them entirely;
  * a recurrence study then ran at `kp = 2.5` against a floor of `22.8` --
    9.1x below -- and its pattern-completion null was read as a fact about the
    substrate before anyone checked the precondition.

Both are exactly the failure `regime_audit`'s own docstring warns about: "a
null there is not evidence about the mechanism." A RuntimeWarning reaches
stderr, survives into logs, and can be promoted to an error with `-W`. A
printed row can do none of those.

These tests pin the mechanism, not any particular number.
"""
from __future__ import annotations

import warnings

import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.diagnostics import regime_audit, require_regime


def _under_floor():
    """k*p = 2.5 against 3*ln(2000) = 22.8 -- the recurrence study's setting."""
    b = Brain(p=0.05, seed=0, engine="numpy_sparse")
    b.add_area("A", 2000, 50, beta=0.1)
    b.add_area("B", 2000, 50, beta=0.1)
    return b


def _over_floor():
    """Same shape with the fiber density raised until the floor is cleared."""
    b = Brain(p=0.05, seed=0, engine="numpy_sparse")
    b.add_area("A", 2000, 50, beta=0.1)
    b.add_area("B", 2000, 50, beta=0.1)
    b.add_connectivity("B", "A", 0.6)          # kp = 30 > 22.8
    return b


def test_out_of_regime_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        regime_audit(_under_floor(), {"A": ["B"]})
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, RuntimeWarning)]
    assert msgs, "an out-of-regime organ produced no RuntimeWarning"
    # The message must name the consequence, not just the numbers: the whole
    # failure mode is reading an out-of-regime null as a fact about the
    # mechanism.
    assert "OUT OF REGIME" in msgs[0]
    assert "not evidence about" in msgs[0]


def test_in_regime_is_silent():
    """A warning that fires always is a warning nobody reads."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        regime_audit(_over_floor(), {"A": ["B"]})
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_warn_false_silences_deliberate_out_of_regime_work():
    """Plenty of legitimate work sits below floor on purpose; it must be able
    to say so once rather than be nagged per call."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        regime_audit(_under_floor(), {"A": ["B"]}, warn=False)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_require_regime_raises_and_names_the_area():
    with pytest.raises(RuntimeError) as e:
        require_regime(_under_floor(), {"A": ["B"]})
    assert "A" in str(e.value) and "floor" in str(e.value)


def test_require_regime_returns_rows_when_clear():
    """Bystander rows are still RETURNED (the caller may want to print them);
    they are simply not gated on."""
    rows = require_regime(_over_floor(), {"A": ["B"]})
    driven = [r for r in rows if r.driven]
    assert driven and all(r.in_regime for r in driven)


def test_bystander_areas_are_not_flagged():
    """An area the drive map does not mention is not being driven, so it has
    nothing to be out of regime about. Flagging bystanders is what turns a
    warning into noise."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        regime_audit(_over_floor(), {"A": ["B"]})     # B is a bystander
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_dead_fiber_reads_as_dead_not_as_a_huge_ratio():
    """A zero-afferent area is a dead fiber, a DIFFERENT defect from being
    below floor, and printing it as '10^10 x below' hides that."""
    b = Brain(p=0.05, seed=0, engine="numpy_sparse")
    b.add_area("A", 2000, 50, beta=0.1)
    b.add_area("B", 2000, 50, beta=0.1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # p=0 override: every declared fiber delivers nothing. The engine
        # wires every area to every other at ambient p, so this is the clean
        # way to construct a genuinely afferent-less area.
        regime_audit(b, {"A": ["B"]}, p=0.0)
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, RuntimeWarning)]
    assert msgs and "NO AFFERENTS" in msgs[0], msgs

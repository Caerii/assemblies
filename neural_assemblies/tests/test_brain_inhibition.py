"""AC area/fiber inhibition on the Brain: the calculus's only control primitives.

Papadimitriou et al. (2020) and Mitropolsky et al. (2021) give the AC exactly
two control operations -- inhibit/disinhibit an AREA, inhibit/disinhibit a
FIBER -- and nothing else. Plasticity is not a mode; it is step 3 of the state
transition. So gating is the whole of control, and until now `Brain.project`
honoured none of it: the state machine existed in `core.inhibition` but only
`nemo_parse` consulted it, and `Brain.inhibit_areas` (used in ~20 modules) does
something else entirely -- it wipes winners for exactly one step.

THE TESTS THAT MATTER MOST ARE THE NEGATIVE ONES. On this substrate a
projection that should not have happened still returns k winners, so a gate
that silently fails to close reads as a normal result. Every test here that
closes something also asserts the OPEN case, because "nothing changed" is what
both a working gate and a dead one look like from the target's side.
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_assemblies.assembly_calculus.parse_errors import (
    empty_project_on_brain,
)
from neural_assemblies.core.brain import Brain


def _brain():
    b = Brain(p=0.05, save_winners=True, seed=1, engine="numpy_sparse")
    b.add_stimulus("s", 20)
    for a in ("LEX", "SUBJ", "OBJ"):
        b.add_area(a, 500, 20, 0.1)
    b.project({"s": ["LEX"]}, {})
    return b


class TestDefaultIsFullyOpen:
    """A Brain nobody has gated must behave EXACTLY as before this existed."""

    def test_no_state_is_allocated_until_something_is_inhibited(self):
        b = _brain()
        assert b._inhibition is None, (
            "the gating state was created eagerly; every Brain now pays for a "
            "feature almost none of them use, and `any_closed()` runs per "
            "projection")

    def test_touching_the_property_leaves_everything_open(self):
        b = _brain()
        assert not b.inhibition.any_closed()

    def test_a_projection_still_happens_with_the_state_materialised(self):
        b = _brain()
        _ = b.inhibition                      # allocate, change nothing
        b.project({}, {"LEX": ["SUBJ"]})
        assert len(b.areas["SUBJ"].winners) > 0

    def test_areas_added_later_are_registered_and_OPEN(self):
        """A new area must not be born gated shut.

        `InhibitionState`'s own constructor closes everything, which is right
        for a parser rule set and wrong here. If `add_area` produced a closed
        area, projections into it would silently stop.
        """
        b = _brain()
        b.inhibit_area("OBJ")                 # force the state into existence
        b.add_area("LATE", 500, 20, 0.1)
        assert b.inhibition.area_open("LATE")
        assert b.inhibition.fiber_open("LEX", "LATE")


class TestAreaInhibitionGatesProjection:

    def test_an_inhibited_area_is_NOT_written(self):
        b = _brain()
        b.project({}, {"LEX": ["SUBJ"]})
        before = b.areas["SUBJ"].winners.copy()

        b.inhibit_area("SUBJ")
        b.project({}, {"LEX": ["SUBJ"]})
        assert np.array_equal(b.areas["SUBJ"].winners, before), (
            "an inhibited area was still updated -- the gate did not reach "
            "Brain._project_impl")

    def test_and_the_SAME_projection_does_write_when_open(self):
        """The positive control. Without it the test above passes on a Brain
        that has stopped projecting altogether."""
        b = _brain()
        b.inhibit_area("SUBJ")
        b.project({}, {"LEX": ["SUBJ"]})
        assert len(b.areas["SUBJ"].winners) == 0

        b.disinhibit_area("SUBJ")
        b.project({}, {"LEX": ["SUBJ"]})
        assert len(b.areas["SUBJ"].winners) > 0

    def test_an_inhibited_SOURCE_does_not_fire(self):
        b = _brain()
        b.inhibit_area("LEX")
        b.project({}, {"LEX": ["SUBJ"]})
        assert len(b.areas["SUBJ"].winners) == 0

    def test_a_stimulus_cannot_drive_an_inhibited_area(self):
        b = _brain()
        b.inhibit_area("OBJ")
        b.project({"s": ["OBJ"]}, {})
        assert len(b.areas["OBJ"].winners) == 0

    def test_indices_are_independent_channels(self):
        """The reference relies on this and a boolean flag cannot express it.

        `generic_trans_verb` disinhibits ADVERB at index 1 while
        `generic_adverb` inhibits it at index 1 -- a separate channel from the
        index-0 default. An area reopens only when EVERY holder has released.
        """
        b = _brain()
        b.inhibit_area("SUBJ", index=0)
        b.inhibit_area("SUBJ", index=1)
        b.disinhibit_area("SUBJ", index=0)
        assert not b.inhibition.area_open("SUBJ"), (
            "releasing one holder reopened the area while another still held "
            "it closed")
        b.disinhibit_area("SUBJ", index=1)
        assert b.inhibition.area_open("SUBJ")


class TestFiberInhibitionGatesProjection:

    def test_a_closed_fiber_carries_nothing(self):
        b = _brain()
        b.inhibit_fiber("LEX", "SUBJ")
        b.project({}, {"LEX": ["SUBJ"]})
        assert len(b.areas["SUBJ"].winners) == 0

    def test_closing_one_fiber_leaves_the_others_alone(self):
        """Scope. A gate that closes everything would pass the test above."""
        b = _brain()
        b.inhibit_fiber("LEX", "SUBJ")
        b.project({}, {"LEX": ["SUBJ", "OBJ"]})
        assert len(b.areas["SUBJ"].winners) == 0
        assert len(b.areas["OBJ"].winners) > 0, (
            "closing LEX<->SUBJ also stopped LEX->OBJ")

    def test_fiber_rules_are_SYMMETRIC_as_in_the_reference(self):
        b = _brain()
        b.inhibit_fiber("LEX", "SUBJ")
        assert not b.inhibition.fiber_open("SUBJ", "LEX")


class TestEmptyProjectOnTheBrain:
    """The paper's violation signal, read off the gate that actually runs."""

    def test_an_open_route_is_not_empty(self):
        b = _brain()
        out = empty_project_on_brain(b, "LEX")
        assert not out.empty
        assert out.detected_by == "drive"

    def test_the_papers_example_fires(self):
        """'the dogs lived' then 'cats': every noun area is inhibited."""
        b = _brain()
        for a in ("SUBJ", "OBJ"):
            b.inhibit_area(a)
        out = empty_project_on_brain(b, "LEX")
        assert out.empty, f"LEX still reaches {out.lex_targets}"

    def test_and_the_projection_really_does_nothing_there(self):
        """Ties the detector to the substrate.

        `empty_project` reading True while `project()` happily writes winners
        would be a detector describing a state machine nobody consults -- which
        is exactly what mutual inhibition turned out to be (#24).
        """
        b = _brain()
        for a in ("SUBJ", "OBJ"):
            b.inhibit_area(a)
        assert empty_project_on_brain(b, "LEX").empty
        b.project({}, {"LEX": ["SUBJ", "OBJ"]})
        assert len(b.areas["SUBJ"].winners) == 0
        assert len(b.areas["OBJ"].winners) == 0


class TestGatingSurvivesAFork:

    def test_a_clone_carries_the_gating_state(self):
        """#103 was a fork handing out state its parent had moved on from.

        A fork that silently reopened every fiber would parse the next word
        differently from its parent while looking identical.
        """
        b = _brain()
        b.inhibit_area("SUBJ")
        b.inhibit_fiber("LEX", "OBJ")
        clone = b.clone() if hasattr(b, "clone") else None
        if clone is None:                     # pragma: no cover - API guard
            pytest.skip("Brain has no clone()")
        assert not clone.inhibition.area_open("SUBJ")
        assert not clone.inhibition.fiber_open("LEX", "OBJ")

    def test_and_the_clone_is_INDEPENDENT(self):
        b = _brain()
        b.inhibit_area("SUBJ")
        clone = b.clone() if hasattr(b, "clone") else None
        if clone is None:                     # pragma: no cover - API guard
            pytest.skip("Brain has no clone()")
        clone.disinhibit_area("SUBJ")
        assert not b.inhibition.area_open("SUBJ"), (
            "releasing the clone's gate released the parent's -- the state is "
            "shared, not copied")


class TestClearActivityIsNotInhibition:
    """`inhibit_areas` names a different mechanism, and that mattered."""

    def test_the_alias_still_works(self):
        b = _brain()
        b.inhibit_areas(["LEX"])
        assert len(b.areas["LEX"].winners) == 0

    def test_clearing_does_NOT_gate_the_next_projection(self):
        """The distinction, made concrete.

        `clear_activity` wipes winners for one step; the area is fully live
        immediately. `inhibit_area` persists. Reading the first as the second
        is why the paper's mechanism looked implemented when it was not.
        """
        b = _brain()
        b.clear_activity(["SUBJ"])
        b.project({}, {"LEX": ["SUBJ"]})
        assert len(b.areas["SUBJ"].winners) > 0, (
            "clear_activity persisted -- it must not; that is inhibit_area's job")
        assert b._inhibition is None, "clear_activity must not touch the gate"

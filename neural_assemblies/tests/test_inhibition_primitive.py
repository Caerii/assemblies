import pytest

"""Pin the NEMO inhibition semantics, especially the parts that look optional.

Ported from `.reference/dmitropolsky-assemblies/parser.py`. Each test here
corresponds to a property the reference rule sets actually depend on, so a
"simplification" that breaks one will break parsing in a way that is hard to
trace back.
"""

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.inhibition import (
    DISINHIBIT, INHIBIT, InhibitionState, apply_rule, area_rule, fiber_rule,
    prepare_targets,
)

AREAS = ["LEX", "SUBJ", "VERB", "OBJ"]


class TestDefaultState:
    def test_everything_starts_closed(self):
        """A fiber nobody opened must not carry signal."""
        s = InhibitionState(AREAS)
        assert not s.area_open("SUBJ")
        assert not s.fiber_open("LEX", "SUBJ")

    def test_initial_areas_are_open(self):
        s = InhibitionState(AREAS, initial_areas=["LEX", "SUBJ"])
        assert s.area_open("LEX") and s.area_open("SUBJ")
        assert not s.area_open("OBJ")


class TestIndexChannels:
    """The index is load-bearing; a boolean flag cannot express these."""

    def test_two_holders_both_must_release(self):
        s = InhibitionState(AREAS, initial_areas=AREAS)
        s.inhibit_fiber("LEX", "SUBJ", 0)
        s.inhibit_fiber("LEX", "SUBJ", 1)
        s.disinhibit_fiber("LEX", "SUBJ", 0)
        assert not s.fiber_open("LEX", "SUBJ"), (
            "fiber reopened while holder 1 still had it inhibited -- index "
            "channels collapsed into a single flag")
        s.disinhibit_fiber("LEX", "SUBJ", 1)
        assert s.fiber_open("LEX", "SUBJ")

    def test_area_channels_are_independent(self):
        """Exactly the reference's ADVERB pattern.

        `generic_trans_verb` does DISINHIBIT ADVERB @1 while `generic_adverb`
        does INHIBIT ADVERB @1 -- a channel distinct from the @0 default.
        """
        s = InhibitionState(AREAS, initial_areas=[])
        s.disinhibit_area("SUBJ", 1)          # releases a channel never taken
        assert not s.area_open("SUBJ"), "the index-0 default was dropped"
        s.disinhibit_area("SUBJ", 0)
        assert s.area_open("SUBJ")

    def test_disinhibit_is_idempotent(self):
        s = InhibitionState(AREAS, initial_areas=AREAS)
        s.disinhibit_fiber("LEX", "SUBJ", 0)
        s.disinhibit_fiber("LEX", "SUBJ", 0)
        assert s.fiber_open("LEX", "SUBJ")


class TestFiberSymmetry:
    def test_inhibit_closes_both_directions(self):
        s = InhibitionState(AREAS, initial_areas=AREAS)
        s.disinhibit_fiber("LEX", "SUBJ", 0)
        s.inhibit_fiber("LEX", "SUBJ", 0)
        assert not s.fiber_open("LEX", "SUBJ")
        assert not s.fiber_open("SUBJ", "LEX"), (
            "fiber rules must be symmetric; the reference rule sets assume it")


class TestProjectMapDerivation:
    """Targets are DERIVED from state, never named by a caller."""

    def _brain(self):
        b = Brain(p=0.1, seed=5, norm_init=False)
        b.add_stimulus("s", 30)
        for a in AREAS:
            b.add_area(a, 300, 30, 0.1)
        b.project({"s": ["LEX"]}, {})     # LEX alone has winners
        return b

    def test_closed_fiber_yields_no_projection(self):
        b = self._brain()
        s = InhibitionState(AREAS, initial_areas=AREAS)
        assert "SUBJ" not in s.project_map(b).get("LEX", [])

    def test_open_fiber_with_active_source_projects(self):
        b = self._brain()
        s = InhibitionState(AREAS, initial_areas=AREAS)
        s.disinhibit_fiber("LEX", "SUBJ", 0)
        assert "SUBJ" in s.project_map(b)["LEX"]

    def test_self_recurrence_is_implied_by_any_open_input_fiber(self):
        """You CANNOT gate an area's self-fiber while it is receiving.

        This looks like a bug and is not: `parser.py:413` puts
        `proj_map[area2].add(area2)` inside the `fiber_states[area1][area2]`
        guard, so the self-projection rides on the state of the INPUT fiber,
        never on `fiber_states[area2][area2]`. Faithfully ported.

        The consequence is a real limit on what gating can express. In NEMO,
        "receive input" and "sustain yourself" are the same event, so the only
        way to switch off an area's self-recurrence is to switch the area (or
        every fiber into it) off entirely -- see case C below.

        This matters because self-recurrence during TRAINING is this repo's
        measured collapse channel for shared areas. Gating cannot close it
        selectively; only a per-fiber beta can (`Brain.update_plasticity`,
        task #88). Anyone "fixing" this to consult the self-fiber would diverge
        from the reference and silently change every parse.
        """
        b = self._brain()
        b.project({"s": ["LEX"]}, {"LEX": ["SUBJ"]})     # give SUBJ winners
        s = InhibitionState(AREAS, initial_areas=AREAS)
        s.disinhibit_fiber("LEX", "SUBJ", 0)
        s.inhibit_fiber("SUBJ", "SUBJ", 0)               # explicitly close it

        assert not s.fiber_open("SUBJ", "SUBJ")
        assert "SUBJ" in s.project_map(b)["SUBJ"], (
            "self-recurrence was gated by the self-fiber -- the reference "
            "derives it from the INPUT fiber (parser.py:413)")

        s.inhibit_fiber("LEX", "SUBJ", 0)                # close the input too
        assert "SUBJ" not in s.project_map(b).get("SUBJ", []), (
            "closing every fiber into an area must stop its self-projection")

    def test_inhibited_area_blocks_its_open_fiber(self):
        """Area state gates independently of fiber state -- this is how the
        reference routes a noun: the noun opens fibers to BOTH SUBJ and OBJ,
        and whichever AREA is currently open decides where it lands."""
        b = self._brain()
        s = InhibitionState(AREAS, initial_areas=AREAS)
        s.disinhibit_fiber("LEX", "SUBJ", 0)
        s.disinhibit_fiber("LEX", "OBJ", 0)
        s.inhibit_area("OBJ", 0)                     # verb not seen yet
        targets = s.project_map(b)["LEX"]
        assert "SUBJ" in targets and "OBJ" not in targets

    def test_silent_source_projects_nothing(self):
        b = self._brain()
        s = InhibitionState(AREAS, initial_areas=AREAS)
        s.disinhibit_fiber("SUBJ", "VERB", 0)
        assert "SUBJ" not in s.project_map(b), (
            "an area with no winners must not appear as a source")

    def test_noun_then_verb_routes_to_different_slots(self):
        """The end-to-end property the whole primitive exists for.

        Same fibers open in both steps; only the AREA state differs, and the
        word lands in a different role. No cross-area comparison, no
        winner-take-all, no Python set.
        """
        b = self._brain()
        s = InhibitionState(AREAS, initial_areas=AREAS)
        for tgt in ("SUBJ", "OBJ"):
            s.disinhibit_fiber("LEX", tgt, 0)

        # LEX->LEX is expected in the map: the reference's own subclass says
        # so (`# because LEX->LEX`), which is why its war-of-fibers bound is 2.
        s.inhibit_area("OBJ", 0)                     # pre-verb
        assert s.project_map(b)["LEX"] == ["LEX", "SUBJ"]

        s.disinhibit_area("OBJ", 0)                  # verb's POST_RULES
        s.inhibit_area("SUBJ", 0)
        assert s.project_map(b)["LEX"] == ["LEX", "OBJ"]


class TestApplyRule:
    def test_rules_drive_the_state(self):
        s = InhibitionState(AREAS, initial_areas=AREAS)
        apply_rule(s, fiber_rule(DISINHIBIT, "LEX", "VERB", 0))
        assert s.fiber_open("LEX", "VERB")
        apply_rule(s, area_rule(INHIBIT, "SUBJ", 0))
        assert not s.area_open("SUBJ")


class TestPrepareTargets:
    """Omitting this silently corrupts binding, so it is pinned."""

    def _brain(self):
        b = Brain(p=0.1, seed=6, norm_init=False)
        b.add_stimulus("s", 30)
        for a in AREAS:
            b.add_area(a, 300, 30, 0.1)
        b.project({"s": ["LEX"]}, {})
        b.project({"s": ["SUBJ"]}, {})    # something already bound in SUBJ
        return b

    def test_unreached_area_is_fixed_reached_area_is_cleared(self):
        b = self._brain()
        s = InhibitionState(AREAS, initial_areas=AREAS)
        s.disinhibit_fiber("LEX", "OBJ", 0)      # LEX reaches OBJ, not SUBJ
        s.disinhibit_fiber("SUBJ", "VERB", 0)    # SUBJ still in the map
        before = len(b.areas["SUBJ"].winners)
        prepare_targets(b, s, lex_area="LEX")
        assert len(b.areas["SUBJ"].winners) == before, (
            "SUBJ is not reached by LEX, so its binding must be preserved")
        assert len(b.areas["OBJ"].winners) == 0, (
            "LEX reaches OBJ, so it must be cleared to receive the new word")


class TestWarOfFibers:
    """The reference raises when LEX reaches more than one role slot."""

    def _brain(self):
        b = Brain(p=0.1, seed=7, norm_init=False)
        b.add_stimulus("s", 30)
        for a in AREAS:
            b.add_area(a, 300, 30, 0.1)
        b.project({"s": ["LEX"]}, {})
        return b

    def test_one_target_plus_self_is_legal(self):
        b = self._brain()
        s = InhibitionState(AREAS, initial_areas=AREAS)
        s.disinhibit_fiber("LEX", "SUBJ", 0)
        s.check_war_of_fibers(s.project_map(b), "LEX")   # must not raise

    def test_two_role_slots_at_once_raises(self):
        """Precisely the ambiguity a winner-take-all would have to arbitrate."""
        b = self._brain()
        s = InhibitionState(AREAS, initial_areas=AREAS)
        s.disinhibit_fiber("LEX", "SUBJ", 0)
        s.disinhibit_fiber("LEX", "OBJ", 0)
        with pytest.raises(ValueError, match="war of fibers"):
            s.check_war_of_fibers(s.project_map(b), "LEX")

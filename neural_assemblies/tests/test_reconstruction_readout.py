"""The reconstruction readout: roles from THIS parse, and the event invariants.

WHAT IS PINNED, and why each is a claim about the SUBSTRATE rather than the
gate (`research/notes/the_assembly_now_decides.md`; measured in
`research/experiments/sentence_conditioned_readout.py` before wiring):

  * Both voices read correctly through `parse_roles_by_reconstruction` --
    including nouns whose TRAINED role bias is the opposite of their role in
    this sentence, which is exactly where the stored-lexicon margin route
    measurably fails (it reads 'the child is entered by the mouse' as
    child=AGENT on training bias).
  * C1 VOICE INVARIANCE: the two voicings of ONE event produce IDENTICAL
    role-area winners -- a canonical event representation, surface order
    absorbed by the learned gating.
  * C2 EVENT SEPARATION: the REVERSED event produces (near-)disjoint winners.
    At the old phon_weight=1 default C2 read 0.9528 -- the substrate could not
    distinguish "child enters mouse" from "mouse enters child" -- and the
    current defaults were flipped partly on this number. Either invariant
    moving is a SUBSTRATE regression, not a parser bug.
  * The occupant-vs-runner-up GAP is positive with margin. The occupant
    reproducing the winners is partly determinism (frozen parse, same
    protocol); the gap to the runner-up is the substrate's contribution, and
    it is what collapsed to ties at the old defaults.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

from neural_assemblies.assembly_calculus.emergent import EmergentParser
from neural_assemblies.assembly_calculus.emergent.curriculum import (
    CurriculumTrainer,
)


@pytest.mark.slow
class TestReconstructionReadout:

    @pytest.fixture(scope="class")
    def trained(self):
        from neural_assemblies.assembly_calculus.emergent.vocabulary_builder import (
            build_vocabulary_preset,
        )

        parser = EmergentParser(n=3000, k=30, seed=42,
                                vocabulary=build_vocabulary_preset("core"),
                                fast_training=True)
        trainer = CurriculumTrainer(parser)
        for stage in ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD",
                      "SENTENCES"):
            trainer.train_stage(stage)
        return parser

    @pytest.mark.parametrize("text,agent,patient", [
        ("the dog chases the cat", "dog", "cat"),
        ("the cat is chased by the dog", "dog", "cat"),
        ("the child enters the mouse", "child", "mouse"),
        # The trained-bias trap: `child` is usually an agent in training, and
        # here it is the patient. The margin route fails this one.
        ("the child is entered by the mouse", "mouse", "child"),
    ])
    def test_roles_read_from_this_parse(self, trained, text, agent, patient):
        roles, diag = trained.parse_roles_by_reconstruction(text.split())
        assert roles.get(agent) == "AGENT", (text, roles, diag["gaps"])
        assert roles.get(patient) == "PATIENT", (text, roles, diag["gaps"])

    def test_gap_is_positive_with_margin(self, trained):
        """Ties, not wrong answers, are how substrate collapse presents here."""
        for text in ("the dog chases the cat", "the cat is chased by the dog"):
            _roles, diag = trained.parse_roles_by_reconstruction(text.split())
            assert diag["gaps"], f"{text!r}: nothing was read at all"
            for (_role, _occ, _top, _runner, gap) in diag["gaps"]:
                assert gap > 0.5, (
                    f"{text!r}: occupant gap {gap:.3f} -- the images are "
                    f"collapsing; see the phon_weight evidence trail")

    def test_c1_voice_invariance(self, trained):
        """One event, two voicings -> IDENTICAL parse state."""
        _r1, d1 = trained.parse_roles_by_reconstruction(
            "the dog chases the cat".split())
        _r2, d2 = trained.parse_roles_by_reconstruction(
            "the cat is chased by the dog".split())
        assert d1["winners"] and d1["winners"] == d2["winners"], (
            "the two voicings of one event no longer produce the same "
            "substrate state -- voice invariance has regressed")

    def test_c2_event_separation(self, trained):
        """The reversed event -> (near-)disjoint parse state.

        0.9528 shared at the old defaults is what 'the substrate cannot
        represent who-did-what' looks like; <= 0.2 is the pinned bound, well
        above the measured 0.0417 and far below the degenerate regime.
        """
        _r1, d1 = trained.parse_roles_by_reconstruction(
            "the child enters the mouse".split())
        _r2, d2 = trained.parse_roles_by_reconstruction(
            "the child is entered by the mouse".split())
        common = set(d1["winners"]) & set(d2["winners"])
        assert common, "no shared role areas to compare"
        shared = sum(
            len(set(d1["winners"][r]) & set(d2["winners"][r]))
            / max(1, len(d1["winners"][r]))
            for r in common) / len(common)
        assert shared <= 0.2, (
            f"reversed events share {shared:.4f} of their parse state -- the "
            f"event representation is degenerating (0.95 was the broken "
            f"regime, 0.04 the measured healthy one)")

"""Passive voice: generated, learned contrastively, and applied at parse.

WHY PASSIVES ARE THE POINT. In an all-active corpus, position predicts role
perfectly, so a role representation buys nothing over a position counter. The
passive states the SAME event in the opposite order, which is the one place a
positional reading is not merely uninformative but INVERTED. It is therefore
both the reason roles must come from the scene and the evidence that they do.

WHAT HAD TO BE TRUE, each verified rather than assumed (#128):

  1. the corpus contains well-formed passives -- aux + past participle + `by`;
  2. their roles come out INVERTED relative to position: subject=patient,
     by-phrase noun=agent;
  3. `_learn_gating_patterns` learns that MARKER reverses roles, CONTRASTIVELY
     (n_contrast > 0), while the determiner -- present in both voices -- scores
     ~0;
  4. `is_passive` actually FIRES during a parse;
  5. the active voice does NOT regress. This is the real failure mode: teach
     the determiner to reverse and every sentence parses backwards.

Two defects were found by running these as measurements before writing them as
tests, and both were one-question-two-spellings:

  * `_learn_gating_patterns` skipped every GROUNDED word, so "by" -- which
    carries spatial grounding -- never reached `_func_subcat_of`, whose own
    docstring promises to recognise exactly that word as a MARKER. Only DET was
    learned.
  * `_determine_role_order` then read subcategories with raw
    `get_func_subcategory` instead of `_func_subcat_of`, so even once MARKER
    was learned at confidence 0.960 it could not be looked up: `is_passive`
    fired 0/2.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

from neural_assemblies.assembly_calculus.emergent import EmergentParser
from neural_assemblies.assembly_calculus.emergent.core.areas import FUNC_MARKER
from neural_assemblies.assembly_calculus.emergent.core.sentence import (
    ground_plans,
)
from neural_assemblies.assembly_calculus.emergent.curriculum import (
    CurriculumTrainer,
)


@pytest.fixture(scope="module")
def plans():
    """Generated SENTENCES plans, grounded. No substrate training needed."""
    parser = EmergentParser(n=500, k=20, seed=42)
    trainer = CurriculumTrainer(parser)
    words = trainer._get_stage_words("SENTENCES")
    for w in words:
        parser.register_word(w.lemma)
    out = trainer.generation.generate(words, 4, stage_name="SENTENCES")
    trainer.generation.register_surface_forms(out, words)
    return ground_plans(parser, out)


def _passives(grounded):
    return [g for g in grounded if "by" in g.words]


def test_the_corpus_contains_passives(plans):
    """A dormant branch and an absent construction look identical from outside."""
    passives = _passives(plans)
    assert passives, "no passive was generated at all"
    # Not pinned to an exact rate -- PASSIVE_EVERY is a stated choice, not a
    # measured constant -- but a corpus that is mostly passive would teach the
    # determiner to reverse, so both ends are bounded.
    share = len(passives) / len(plans)
    assert 0.02 <= share <= 0.35, f"passive share {share:.1%} is out of range"


def test_passives_are_wellformed(plans):
    """aux + past participle + `by` + agent. Checked BEFORE anything trains.

    The same guard that would have caught "the store build the dog": a
    malformed passive teaches the marker to reverse roles in sentences that are
    not passives.
    """
    for g in _passives(plans):
        by_i = g.words.index("by")
        assert by_i >= 2, f"{' '.join(g.words)!r}: `by` with no clause before it"
        assert by_i + 2 < len(g.words), (
            f"{' '.join(g.words)!r}: `by` with no agent after it")
        # aux immediately before the participle, participle immediately
        # before `by`. is/was: the corpus now varies tense, and the passive
        # auxiliary carries it.
        assert g.words[by_i - 2] in ("is", "was"), (
            f"{' '.join(g.words)!r}: no auxiliary before the participle")


def test_passive_roles_are_inverted_relative_to_position(plans):
    """subject = PATIENT, by-phrase noun = AGENT.

    This is the whole claim. A positional inducer reads the first noun as the
    agent, so agreeing with it here would mean the scene was not consulted.
    """
    checked = 0
    for g in _passives(plans):
        by_i = g.words.index("by")
        pre = [r for r in g.roles[:by_i] if r not in (None, "action")]
        post = [r for r in g.roles[by_i + 1:] if r is not None]
        if not pre or not post:
            # Two participants a learner cannot tell apart get no role at all
            # -- honest, and not a wrong answer. See SceneEvent.role_of_features.
            continue
        checked += 1
        assert pre[0] == "patient", (
            f"{' '.join(g.words)!r}: subject scored {pre[0]!r}, not patient")
        assert post[0] == "agent", (
            f"{' '.join(g.words)!r}: by-phrase scored {post[0]!r}, not agent")
    assert checked > 0, "no passive had both roles derivable"


@pytest.mark.slow
class TestTheParserLearnsAndAppliesIt:
    """Needs real curriculum training, so it lives in the slow tier."""

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

    def test_marker_is_learned_contrastively(self, trained):
        """`n_contrast > 0` is what separates a cue from a constant.

        A subcategory present in EVERY sentence explains no variation in role
        order; scoring it on the marginal rate instead is how "the" ends up
        reversing roles.
        """
        # WORD level, not subcategory level. The MARKER pool now contains
        # both 'by' (reverses voice) and 'to' (marks a recipient), and pooled
        # they CANCEL -- measured conf 0.438 the moment ditransitives entered
        # the corpus, killing passives. The claim "the marker is learned
        # contrastively" is carried by the per-word gating, which is also the
        # papers' model of control (per-word action programs).
        wg = trained.learned_word_gating
        assert "by" in wg, f"'by' never learned; learned {sorted(wg)}"
        marker = wg["by"]
        assert marker["reverses_roles"] is True
        assert marker["confidence"] > 0.5, marker
        assert marker["n_contrast"] > 0, (
            "'by' was present in every sentence, so it explains no variation")

    def test_the_determiner_does_not_reverse(self, trained):
        """"the" occurs in both voices and must score ~0."""
        det = trained.learned_gating.get("DET")
        if det is None:
            pytest.skip("no determiner subcategory learned")
        assert not det["reverses_roles"], det
        assert det["confidence"] < 0.5, det

    @pytest.mark.parametrize("text,agent,patient", [
        ("the cat is chased by the dog", "dog", "cat"),
        ("the ball is held by the boy", "boy", "ball"),
    ])
    def test_is_passive_fires_and_roles_invert(self, trained, text, agent,
                                               patient):
        words = text.split()
        cats = {w: trained.classify_word_cached(w)[0] for w in words}
        _order, is_passive = trained._determine_role_order(words, cats)
        assert is_passive, f"{text!r}: the passive branch did not fire"
        # The PRODUCTION route. `_assign_roles_neural` is demoted to the ERP
        # calibration path only, and its word-level role statistics move with
        # the corpus -- pinning them here would pin the demoted instrument.
        roles, _diag = trained.parse_roles_by_reconstruction(words)
        assert roles.get(agent) == "AGENT", roles
        assert roles.get(patient) == "PATIENT", roles

    @pytest.mark.parametrize("text,agent,patient", [
        ("the dog chases the cat", "dog", "cat"),
        ("the boy holds the ball", "boy", "ball"),
    ])
    def test_active_voice_does_not_regress(self, trained, text, agent, patient):
        """The failure mode: a passive-bearing corpus reversing EVERYTHING."""
        words = text.split()
        cats = {w: trained.classify_word_cached(w)[0] for w in words}
        _order, is_passive = trained._determine_role_order(words, cats)
        assert not is_passive, f"{text!r}: active sentence read as passive"
        roles, _diag = trained.parse_roles_by_reconstruction(words)
        assert roles.get(agent) == "AGENT", roles
        assert roles.get(patient) == "PATIENT", roles

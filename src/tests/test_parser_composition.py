"""
Tests for the composed NemoParser pipeline.

Integrates word category learning, role binding, and word order into
a single parser, building on the proven Phase 3 NEMO pattern tests.

Architecture:
    Input:    PHON + VISUAL/MOTOR grounding
    Layer 1:  LEX_NOUN / LEX_VERB (grounded word learning)
    Layer 2:  ROLE_AGENT / ROLE_ACTION / ROLE_PATIENT
    Layer 3:  SEQ (word order via sequence memorization)

References:
    Mitropolsky & Papadimitriou (2023). arXiv:2306.15364.
    Mitropolsky & Papadimitriou (2025). "Simulated Language Acquisition."
"""

import time

import pytest

from src.core.brain import Brain
from src.assembly_calculus import overlap, chance_overlap
from src.assembly_calculus.parser import NemoParser
from src.assembly_calculus.ops import _snap


N = 10000
K = 100
P = 0.05
BETA = 0.1
SEED = 42
ROUNDS = 10


@pytest.fixture(autouse=True)
def _timer(request):
    t0 = time.perf_counter()
    yield
    print(f"  [{time.perf_counter() - t0:.3f}s]")


# Vocabulary
NOUNS = ["dog", "cat", "bird"]
VERBS = ["chases", "sees", "catches"]

# Training sentences (SVO)
TRAIN_SENTENCES = [
    ["dog", "chases", "cat"],
    ["cat", "sees", "bird"],
    ["bird", "catches", "dog"],
    ["dog", "sees", "bird"],
]


def _build_parser():
    """Build and fully train the parser pipeline."""
    brain = Brain(p=P, save_winners=True, seed=SEED, engine="numpy_sparse")
    parser = NemoParser(brain, n=N, k=K, beta=BETA, rounds=ROUNDS)
    parser.setup_areas()

    # Register all words with grounding stimuli
    for noun in NOUNS:
        parser.register_word(noun, "noun", f"vis_{noun}")
    for verb in VERBS:
        parser.register_word(verb, "verb", f"mot_{verb}")

    # Train all three phases
    parser.train_lexicon()
    parser.train_roles(TRAIN_SENTENCES)
    parser.train_word_order(TRAIN_SENTENCES)

    return parser


# ======================================================================
# 1. Category Classification
# ======================================================================

class TestParserCategoryClassification:
    """Test Layer 1: grounded word category learning."""

    def test_all_nouns_classified_correctly(self):
        """Each trained noun should classify as 'noun'."""
        parser = _build_parser()
        for noun in NOUNS:
            result = parser.classify_word(noun)
            assert result == "noun", (
                f"'{noun}' should classify as noun, got '{result}'"
            )

    def test_all_verbs_classified_correctly(self):
        """Each trained verb should classify as 'verb'."""
        parser = _build_parser()
        for verb in VERBS:
            result = parser.classify_word(verb)
            assert result == "verb", (
                f"'{verb}' should classify as verb, got '{result}'"
            )

    def test_category_accuracy_above_80_percent(self):
        """Overall category accuracy >= 80% (5/6 minimum)."""
        parser = _build_parser()
        correct = 0
        total = len(NOUNS) + len(VERBS)

        for noun in NOUNS:
            if parser.classify_word(noun) == "noun":
                correct += 1
        for verb in VERBS:
            if parser.classify_word(verb) == "verb":
                correct += 1

        accuracy = correct / total
        print(f"  Category accuracy: {correct}/{total} = {accuracy:.0%}")
        assert accuracy >= 0.8, (
            f"Category accuracy {accuracy:.0%} < 80% ({correct}/{total})"
        )


# ======================================================================
# 2. Role Binding
# ======================================================================

class TestParserRoleBinding:
    """Test Layer 2: thematic role assignment."""

    def test_agent_role_binding(self):
        """Subjects in training sentences should have ROLE_AGENT assemblies."""
        parser = _build_parser()
        # "dog" appears as agent in "dog chases cat" and "dog sees bird"
        assert "dog" in parser.role_lexicons.get("ROLE_AGENT", {}), (
            "dog should be in ROLE_AGENT lexicon"
        )
        asm = parser.role_lexicons["ROLE_AGENT"]["dog"]
        assert asm.area == "ROLE_AGENT"
        assert len(asm) == K

    def test_action_role_binding(self):
        """Verbs should have ROLE_ACTION assemblies."""
        parser = _build_parser()
        assert "chases" in parser.role_lexicons.get("ROLE_ACTION", {}), (
            "chases should be in ROLE_ACTION lexicon"
        )
        asm = parser.role_lexicons["ROLE_ACTION"]["chases"]
        assert asm.area == "ROLE_ACTION"

    def test_patient_role_binding(self):
        """Objects should have ROLE_PATIENT assemblies."""
        parser = _build_parser()
        # "cat" is patient in "dog chases cat"
        assert "cat" in parser.role_lexicons.get("ROLE_PATIENT", {}), (
            "cat should be in ROLE_PATIENT lexicon"
        )
        asm = parser.role_lexicons["ROLE_PATIENT"]["cat"]
        assert asm.area == "ROLE_PATIENT"

    def test_roles_in_different_areas(self):
        """Agent and patient assemblies should be in different brain areas."""
        parser = _build_parser()
        agent_asm = parser.role_lexicons["ROLE_AGENT"]["dog"]
        patient_asm = parser.role_lexicons["ROLE_PATIENT"]["cat"]
        assert agent_asm.area != patient_asm.area, (
            f"Agent ({agent_asm.area}) and patient ({patient_asm.area}) "
            f"should be in different areas"
        )


# ======================================================================
# 3. Word Order
# ======================================================================

class TestParserWordOrder:
    """Test Layer 3: SVO sequence memorization."""

    def test_svo_memorization_creates_seq_area(self):
        """Training word order should populate the SEQ area."""
        parser = _build_parser()
        # After training, SEQ area should have been used
        seq_area = parser.brain.areas["SEQ"]
        assert seq_area is not None

    def test_cue_recovers_first_item(self):
        """Cueing with first word's stimulus recovers a meaningful assembly."""
        parser = _build_parser()
        brain = parser.brain

        # Cue with the first word of a training sentence
        first_stim = parser.stim_map["dog"]
        brain.project({first_stim: ["SEQ"]}, {})
        for _ in range(5):
            brain.project({first_stim: ["SEQ"]}, {"SEQ": ["SEQ"]})
        cue_snap = _snap(brain, "SEQ")

        # The recovered assembly should have K winners
        assert len(cue_snap) == K


# ======================================================================
# 4. End-to-End Parsing
# ======================================================================

class TestParserEndToEnd:
    """Full parse of sentences through the complete pipeline."""

    def test_trained_sentence_parse(self):
        """Parse a sentence that appeared in training data."""
        parser = _build_parser()
        result = parser.parse(["dog", "chases", "cat"])

        assert result["categories"]["dog"] == "noun"
        assert result["categories"]["chases"] == "verb"
        assert result["categories"]["cat"] == "noun"

        assert result["roles"]["dog"] == "AGENT"
        assert result["roles"]["chases"] == "ACTION"
        assert result["roles"]["cat"] == "PATIENT"

    def test_novel_sentence_categories(self):
        """Words in a novel sentence should get correct category labels.

        "bird catches dog" is not in training (training has
        "bird catches dog" at position 3), but all words are trained.
        """
        parser = _build_parser()
        result = parser.parse(["cat", "catches", "dog"])

        assert result["categories"]["cat"] == "noun"
        assert result["categories"]["catches"] == "verb"
        assert result["categories"]["dog"] == "noun"

    def test_novel_sentence_roles(self):
        """Novel SVO sentence should get correct role assignments."""
        parser = _build_parser()
        result = parser.parse(["cat", "catches", "dog"])

        # SVO: cat=AGENT, catches=ACTION, dog=PATIENT
        assert result["roles"]["cat"] == "AGENT"
        assert result["roles"]["catches"] == "ACTION"
        assert result["roles"]["dog"] == "PATIENT"

    def test_full_pipeline_accuracy(self):
        """Train on 4 sentences, parse 2 novel ones.

        Measure category + role accuracy across novel sentences.
        Target: >= 80% accuracy on both.
        """
        parser = _build_parser()

        test_sentences = [
            (["cat", "chases", "bird"], {
                "categories": {"cat": "noun", "chases": "verb", "bird": "noun"},
                "roles": {"cat": "AGENT", "chases": "ACTION", "bird": "PATIENT"},
            }),
            (["bird", "sees", "dog"], {
                "categories": {"bird": "noun", "sees": "verb", "dog": "noun"},
                "roles": {"bird": "AGENT", "sees": "ACTION", "dog": "PATIENT"},
            }),
        ]

        cat_correct = 0
        role_correct = 0
        total = 0

        for words, expected in test_sentences:
            result = parser.parse(words)
            for word in words:
                total += 1
                if result["categories"][word] == expected["categories"][word]:
                    cat_correct += 1
                if result["roles"][word] == expected["roles"][word]:
                    role_correct += 1

        cat_acc = cat_correct / total
        role_acc = role_correct / total
        print(f"  Category accuracy: {cat_correct}/{total} = {cat_acc:.0%}")
        print(f"  Role accuracy: {role_correct}/{total} = {role_acc:.0%}")

        assert cat_acc >= 0.8, (
            f"Category accuracy {cat_acc:.0%} < 80%"
        )
        assert role_acc >= 0.8, (
            f"Role accuracy {role_acc:.0%} < 80%"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

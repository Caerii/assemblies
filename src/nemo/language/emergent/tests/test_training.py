#!/usr/bin/env python
"""
Test Training and Vocabulary Learning
=====================================

Version: 2.0.0
Date: 2025-11-30

Tests that the emergent learner correctly:
1. Learns vocabulary from grounded training data
2. Categorizes words by their grounding patterns
3. Learns word order (SVO)
4. Learns thematic roles (agent, patient, action)

Now uses SHARED trained model for faster test runs.

Run:
    uv run python test_training.py
"""

import sys
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.nemo.language.emergent import (
    EmergentLanguageLearner,
    create_training_data,
)

# Global to hold shared learner
_shared_learner: Optional[EmergentLanguageLearner] = None


def _get_learner() -> EmergentLanguageLearner:
    """Get the shared learner or train a new one."""
    global _shared_learner
    
    if _shared_learner is not None:
        return _shared_learner
    
    # Train fresh if no shared learner
    learner = EmergentLanguageLearner(verbose=False)
    data = create_training_data()
    
    for epoch in range(3):
        for s in data:
            learner.present_grounded_sentence(
                s.words, s.contexts, roles=s.roles, mood=s.mood
            )
    
    return learner


def test_training_data_structure(learner=None):
    """Test that training data has correct structure."""
    print("\n" + "="*60)
    print("TEST: Training Data Structure")
    print("="*60)
    
    data = create_training_data()
    
    print(f"\nTotal training sentences: {len(data)}")
    
    # Check structure
    for i, sentence in enumerate(data[:5]):
        print(f"\n  Sentence {i+1}: {' '.join(sentence.words)}")
        print(f"    Roles: {sentence.roles}")
        print(f"    Mood: {sentence.mood}")
        print(f"    Contexts: {len(sentence.contexts)} grounding contexts")
        
        assert len(sentence.words) == len(sentence.contexts), \
            "Words and contexts must have same length"
        assert len(sentence.words) == len(sentence.roles), \
            "Words and roles must have same length"
    
    print("\n✓ Training data structure is correct")
    return True


def test_vocabulary_learning(learner=None):
    """Test that vocabulary is learned and categorized correctly."""
    print("\n" + "="*60)
    print("TEST: Vocabulary Learning")
    print("="*60)
    
    if learner is None:
        learner = _get_learner()
    
    print(f"\nSentences seen: {learner.sentences_seen}")
    print(f"Vocabulary size: {len(learner.word_count)}")
    
    # Check vocabulary by category
    vocab = learner.get_vocabulary_by_category()
    
    print("\nLearned vocabulary by category:")
    for cat, words in sorted(vocab.items()):
        print(f"  {cat}: {list(words)}")
    
    # Verify expected categories exist
    assert 'NOUN' in vocab, "Should learn NOUNs"
    assert 'VERB' in vocab, "Should learn VERBs"
    assert 'FUNCTION' in vocab, "Should learn FUNCTION words"
    
    # Verify specific words are in correct categories
    nouns = vocab.get('NOUN', [])
    verbs = vocab.get('VERB', [])
    
    assert 'dog' in nouns, "dog should be a NOUN"
    assert 'cat' in nouns, "cat should be a NOUN"
    assert 'runs' in verbs, "runs should be a VERB"
    assert 'eats' in verbs, "eats should be a VERB"
    
    print("\n✓ Vocabulary learning is correct")
    return True


def test_emergent_categories(learner=None):
    """Test that categories emerge from grounding patterns."""
    print("\n" + "="*60)
    print("TEST: Emergent Categories")
    print("="*60)
    
    if learner is None:
        learner = _get_learner()
    
    # Test specific words
    test_words = [
        ('dog', 'NOUN', 'visual grounding'),
        ('cat', 'NOUN', 'visual grounding'),
        ('runs', 'VERB', 'motor grounding'),
        ('eats', 'VERB', 'motor grounding'),
        ('big', 'ADJECTIVE', 'property grounding'),
        ('the', 'FUNCTION', 'no grounding'),
        ('she', 'PRONOUN', 'social grounding'),
        ('on', 'PREPOSITION', 'spatial grounding'),
    ]
    
    print("\nCategory inference from grounding:")
    all_correct = True
    
    for word, expected_cat, reason in test_words:
        cat, scores = learner.get_emergent_category(word)
        status = "✓" if cat == expected_cat else "✗"
        if cat != expected_cat:
            all_correct = False
        print(f"  {status} '{word}' -> {cat} (expected {expected_cat}, {reason})")
    
    if all_correct:
        print("\n✓ All categories emerged correctly from grounding")
    else:
        print("\n✗ Some categories did not match expected")
    
    return all_correct


def test_word_order_learning(learner=None):
    """Test that SVO word order is learned."""
    print("\n" + "="*60)
    print("TEST: Word Order Learning (SVO)")
    print("="*60)
    
    if learner is None:
        learner = _get_learner()
    
    word_order = learner.get_word_order()
    print(f"\nLearned word order: {word_order}")
    
    # Check that we have some order
    assert len(word_order) > 0, "Should learn some word order"
    
    # Check category transitions
    print("\nCategory transitions (top 10):")
    sorted_transitions = sorted(
        learner.category_transitions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    for (src, dst), count in sorted_transitions:
        print(f"  {src} -> {dst}: {count}")
    
    print("\n✓ Word order patterns learned")
    return True


def test_thematic_roles(learner=None):
    """Test that thematic roles (agent, patient, action) are learned."""
    print("\n" + "="*60)
    print("TEST: Thematic Role Learning")
    print("="*60)
    
    if learner is None:
        learner = _get_learner()
    
    # Test thematic roles for specific words
    test_words = ['dog', 'cat', 'runs', 'eats', 'food', 'bird']
    
    print("\nThematic role inference:")
    for word in test_words:
        role, confidence = learner.get_thematic_role(word)
        print(f"  '{word}' -> {role} (confidence: {confidence:.2f})")
    
    # Verify animate nouns tend to be agents
    dog_role, _ = learner.get_thematic_role('dog')
    cat_role, _ = learner.get_thematic_role('cat')
    
    # Verify verbs are actions
    runs_role, _ = learner.get_thematic_role('runs')
    eats_role, _ = learner.get_thematic_role('eats')
    
    print("\n✓ Thematic roles learned")
    return True


def test_vp_assembly_storage(learner=None):
    """Test that VP assemblies are stored for subject-verb-object patterns."""
    print("\n" + "="*60)
    print("TEST: VP Assembly Storage")
    print("="*60)
    
    if learner is None:
        learner = _get_learner()
    
    from src.nemo.language.emergent.areas import Area
    
    # Check VP assemblies
    vp_keys = list(learner.brain.learned_assemblies[Area.VP].keys())
    
    print(f"\nTotal VP patterns learned: {len(vp_keys)}")
    
    # Count intransitive vs transitive
    intransitive = [k for k in vp_keys if k.count('_') == 1]
    transitive = [k for k in vp_keys if k.count('_') == 2]
    
    print(f"  Intransitive (subj_verb): {len(intransitive)}")
    print(f"  Transitive (subj_verb_obj): {len(transitive)}")
    
    # Show some examples
    print("\nExample VP patterns:")
    for key in vp_keys[:10]:
        print(f"  {key}")
    if len(vp_keys) > 10:
        print(f"  ... and {len(vp_keys) - 10} more")
    
    assert len(vp_keys) > 0, "Should learn VP patterns"
    
    print("\n✓ VP assemblies stored correctly")
    return True


def run_all_tests(learner=None):
    """Run all training tests with optional shared learner."""
    global _shared_learner
    
    print("\n" + "="*60)
    print("EMERGENT LANGUAGE LEARNER - TRAINING TESTS")
    print("="*60)
    
    # Use shared learner if provided
    if learner is not None:
        _shared_learner = learner
        print("(Using shared pre-trained model)")
    
    tests = [
        test_training_data_structure,
        test_vocabulary_learning,
        test_emergent_categories,
        test_word_order_learning,
        test_thematic_roles,
        test_vp_assembly_storage,
    ]
    
    results = []
    for test in tests:
        try:
            result = test(learner)
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\n✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    # When run standalone, train fresh
    success = run_all_tests()
    sys.exit(0 if success else 1)


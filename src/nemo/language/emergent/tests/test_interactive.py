"""
Tests for Interactive NEMO
==========================

Tests the interactive learning capabilities.
"""

import time
from src.nemo.language.emergent.interactive import InteractiveLearner


def test_basic_interaction():
    """Test basic interaction flow."""
    print("\n=== Test: Basic Interaction ===")
    
    nemo = InteractiveLearner(use_cuda=True, bootstrap_training=True, verbose=False)
    
    # Test statement learning (use words likely in curriculum)
    response = nemo.interact("the dog runs")
    print("Input: 'the dog runs'")
    print(f"Response: '{response}'")
    # Valid responses: learned something, ok, or asking about unknown word
    assert "learned" in response.lower() or "ok" in response.lower() or "what is" in response.lower()
    
    # Test question answering
    response = nemo.interact("who runs")
    print("Input: 'who runs'")
    print(f"Response: '{response}'")
    assert "dog" in response.lower()
    
    print("✓ Basic interaction works")
    return True


def test_self_queries():
    """Test self-referential queries."""
    print("\n=== Test: Self Queries ===")
    
    nemo = InteractiveLearner(use_cuda=True, bootstrap_training=True, verbose=False)
    
    # Test vocabulary count
    response = nemo.interact("how many words do you know")
    print("Input: 'how many words do you know'")
    print(f"Response: '{response}'")
    assert "words" in response.lower()
    
    # Test knowledge report
    response = nemo.interact("what do you know")
    print("Input: 'what do you know'")
    print(f"Response: '{response}'")
    assert "know" in response.lower() or "noun" in response.lower()
    
    print("✓ Self queries work")
    return True


def test_unknown_word_detection():
    """Test detection and handling of unknown words."""
    print("\n=== Test: Unknown Word Detection ===")
    
    nemo = InteractiveLearner(use_cuda=True, bootstrap_training=True, verbose=False)
    
    # Introduce unknown word
    response = nemo.interact("the giraffe eats leaves")
    print("Input: 'the giraffe eats leaves'")
    print(f"Response: '{response}'")
    
    # Should ask about unknown word or acknowledge learning
    assert "giraffe" in response.lower() or "learned" in response.lower()
    
    print("✓ Unknown word detection works")
    return True


def test_teaching():
    """Test explicit teaching of new words."""
    print("\n=== Test: Teaching New Words ===")
    
    nemo = InteractiveLearner(use_cuda=True, bootstrap_training=True, verbose=False)
    
    # Teach a new word
    nemo.teach("elephant", "NOUN", [
        "the elephant is big",
        "elephants eat plants"
    ])
    
    # Check it was learned
    cat, _ = nemo.learner.get_emergent_category("elephant")
    print(f"Taught 'elephant', category: {cat}")
    assert cat == "NOUN"
    
    # Use it in a sentence
    response = nemo.interact("the elephant runs")
    print("Input: 'the elephant runs'")
    print(f"Response: '{response}'")
    
    print("✓ Teaching works")
    return True


def test_continuous_learning():
    """Test that NEMO learns from each interaction."""
    print("\n=== Test: Continuous Learning ===")
    
    nemo = InteractiveLearner(use_cuda=True, bootstrap_training=True, verbose=False)
    
    initial_vocab = len(nemo.learner.word_count)
    print(f"Initial vocabulary: {initial_vocab}")
    
    # Teach through interaction
    nemo.interact("the zebra runs quickly")
    nemo.interact("zebras are fast")
    nemo.interact("the zebra eats grass")
    
    final_vocab = len(nemo.learner.word_count)
    print(f"Final vocabulary: {final_vocab}")
    
    # Should have learned new words
    assert final_vocab > initial_vocab
    
    # Should now know about zebra
    cat, _ = nemo.learner.get_emergent_category("zebra")
    print(f"Learned 'zebra' as: {cat}")
    
    print("✓ Continuous learning works")
    return True


def test_dialogue_state():
    """Test dialogue state tracking."""
    print("\n=== Test: Dialogue State ===")
    
    nemo = InteractiveLearner(use_cuda=True, bootstrap_training=True, verbose=False)
    
    # Have a conversation
    nemo.interact("the dog runs")
    nemo.interact("the cat sleeps")
    nemo.interact("who runs")
    
    # Check dialogue history
    assert nemo.dialogue.total_turns == 6  # 3 user + 3 system
    assert nemo.dialogue.user_turns == 3
    assert nemo.dialogue.system_turns == 3
    
    print(f"Dialogue turns: {nemo.dialogue.total_turns}")
    print(f"Recent words: {nemo.dialogue.get_recent_words(2)[:5]}")
    
    # Reset and check
    nemo.reset_dialogue()
    assert nemo.dialogue.total_turns == 0
    
    print("✓ Dialogue state tracking works")
    return True


def test_performance():
    """Test interaction performance."""
    print("\n=== Test: Performance ===")
    
    nemo = InteractiveLearner(use_cuda=True, bootstrap_training=True, verbose=False)
    
    # Time multiple interactions
    n_interactions = 20
    start = time.time()
    
    for i in range(n_interactions):
        nemo.interact("the dog runs fast")
        nemo.interact("what does the dog do")
    
    elapsed = time.time() - start
    per_interaction = elapsed / (n_interactions * 2)
    
    print(f"Total time for {n_interactions * 2} interactions: {elapsed:.2f}s")
    print(f"Time per interaction: {per_interaction * 1000:.1f}ms")
    
    # Should be reasonably fast (< 500ms per interaction)
    # Note: Emergent generation is slower than template-based but more NEMO-like
    assert per_interaction < 0.5, f"Too slow: {per_interaction * 1000:.1f}ms per interaction"
    
    print("✓ Performance is acceptable")
    return True


def run_all_tests(learner=None):
    """Run all interactive tests."""
    print("=" * 60)
    print("Interactive NEMO Tests")
    print("=" * 60)
    
    tests = [
        test_basic_interaction,
        test_self_queries,
        test_unknown_word_detection,
        test_teaching,
        test_continuous_learning,
        test_dialogue_state,
        test_performance,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed with error: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)


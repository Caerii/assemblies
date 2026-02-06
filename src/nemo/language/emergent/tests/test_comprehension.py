#!/usr/bin/env python
"""
Test Question Answering / Comprehension
=======================================

Version: 2.0.0
Date: 2025-11-30

Tests that the QuestionAnswerer correctly:
1. Answers "what does X verb?" questions
2. Answers "who verbs X?" questions
3. Answers "does X verb Y?" yes/no questions
4. Reports unknown patterns correctly

Now uses SHARED trained model for faster test runs.

Run:
    uv run python test_comprehension.py
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.nemo.language.emergent import (
    EmergentLanguageLearner,
    create_training_data,
    QuestionAnswerer,
)

# Global shared learner
_shared_learner: Optional[EmergentLanguageLearner] = None


def get_trained_learner(learner=None):
    """Get a trained learner for testing (uses shared if available)."""
    global _shared_learner
    
    if learner is not None:
        return learner
    
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


def test_what_questions(learner=None):
    """Test 'what does X verb?' questions."""
    print("\n" + "="*60)
    print("TEST: 'What' Questions")
    print("="*60)
    
    learner = get_trained_learner(learner)
    qa = QuestionAnswerer(learner)
    
    questions = [
        ['what', 'does', 'the', 'dog', 'eats'],
        ['what', 'does', 'the', 'cat', 'chases'],
        ['what', 'does', 'the', 'girl', 'sees'],
    ]
    
    print("\nAsking 'what' questions:")
    for q in questions:
        answer = qa.answer(q)
        print(f"  Q: {' '.join(q)}")
        print(f"  A: {answer}")
        print()
    
    # Verify answers contain learned objects
    answer = qa.answer(['what', 'does', 'the', 'cat', 'chases'])
    # Cat should chase things like bird, dog, etc.
    has_object = any(obj in answer for obj in ['bird', 'dog', 'boy', 'girl', 'food'])
    
    if has_object:
        print("✓ 'What' questions return learned objects")
    else:
        print("✗ 'What' questions not returning expected objects")
    
    return has_object


def test_who_questions(learner=None):
    """Test 'who verbs X?' questions."""
    print("\n" + "="*60)
    print("TEST: 'Who' Questions")
    print("="*60)
    
    learner = get_trained_learner(learner)
    qa = QuestionAnswerer(learner)
    
    questions = [
        ['who', 'chases', 'the', 'bird'],
        ['who', 'eats', 'the', 'food'],
        ['who', 'sleeps'],
    ]
    
    print("\nAsking 'who' questions:")
    for q in questions:
        answer = qa.answer(q)
        print(f"  Q: {' '.join(q)}")
        print(f"  A: {answer}")
        print()
    
    # Verify answers contain learned subjects
    answer = qa.answer(['who', 'sleeps'])
    # Animate things sleep
    has_subject = any(subj in answer for subj in ['dog', 'cat', 'bird', 'boy', 'girl'])
    
    if has_subject:
        print("✓ 'Who' questions return learned subjects")
    else:
        print("✗ 'Who' questions not returning expected subjects")
    
    return has_subject


def test_does_questions_positive(learner=None):
    """Test 'does X verb Y?' questions with learned patterns."""
    print("\n" + "="*60)
    print("TEST: 'Does' Questions (Positive - Learned Patterns)")
    print("="*60)
    
    learner = get_trained_learner(learner)
    qa = QuestionAnswerer(learner)
    
    # These should be learned (animate subjects with action verbs)
    questions = [
        (['does', 'the', 'dog', 'runs'], True),
        (['does', 'the', 'cat', 'sleeps'], True),
        (['does', 'the', 'bird', 'jumps'], True),
    ]
    
    print("\nAsking about LEARNED patterns:")
    all_correct = True
    
    for q, expected_yes in questions:
        answer = qa.answer(q)
        is_yes = answer.lower().startswith('yes')
        
        status = "✓" if is_yes == expected_yes else "✗"
        if is_yes != expected_yes:
            all_correct = False
        
        print(f"  {status} Q: {' '.join(q)}")
        print(f"      A: {answer}")
        print(f"      Expected: {'Yes' if expected_yes else 'No'}")
        print()
    
    if all_correct:
        print("✓ 'Does' questions correctly confirm learned patterns")
    return all_correct


def test_does_questions_negative(learner=None):
    """Test 'does X verb Y?' questions with unlearned patterns."""
    print("\n" + "="*60)
    print("TEST: 'Does' Questions (Negative - Unlearned Patterns)")
    print("="*60)
    
    learner = get_trained_learner(learner)
    qa = QuestionAnswerer(learner)
    
    # These should NOT be learned (inanimate subjects, or wrong verbs)
    questions = [
        (['does', 'the', 'table', 'runs'], False),  # Inanimate
        (['does', 'the', 'book', 'eats'], False),   # Inanimate
        (['does', 'the', 'dog', 'flies'], False),   # Wrong verb (not trained)
    ]
    
    print("\nAsking about UNLEARNED patterns:")
    all_correct = True
    
    for q, expected_yes in questions:
        answer = qa.answer(q)
        is_yes = answer.lower().startswith('yes')
        
        # For unlearned, we expect "I haven't learned" not "Yes"
        status = "✓" if is_yes == expected_yes else "✗"
        if is_yes != expected_yes:
            all_correct = False
        
        print(f"  {status} Q: {' '.join(q)}")
        print(f"      A: {answer}")
        print(f"      Expected: {'Yes' if expected_yes else 'Not learned'}")
        print()
    
    if all_correct:
        print("✓ 'Does' questions correctly reject unlearned patterns")
    return all_correct


def test_knowledge_summary(learner=None):
    """Test the knowledge summary feature."""
    print("\n" + "="*60)
    print("TEST: Knowledge Summary")
    print("="*60)
    
    learner = get_trained_learner(learner)
    qa = QuestionAnswerer(learner)
    
    summary = qa.get_knowledge_summary()
    
    print("\nKnowledge Summary:")
    print(f"  Total VP patterns: {summary['total_patterns']}")
    print(f"  Intransitive patterns: {summary['intransitive_patterns']}")
    print(f"  Transitive patterns: {summary['transitive_patterns']}")
    print(f"  Unique subjects: {summary['unique_subjects'][:5]}...")
    print(f"  Unique verbs: {summary['unique_verbs']}")
    print(f"  Unique objects: {summary['unique_objects'][:5]}...")
    
    # Verify we have learned something
    has_patterns = summary['total_patterns'] > 0
    has_verbs = len(summary['unique_verbs']) > 0
    
    if has_patterns and has_verbs:
        print("\n✓ Knowledge summary is populated")
    else:
        print("\n✗ Knowledge summary is empty")
    
    return has_patterns and has_verbs


def test_unknown_question_types(learner=None):
    """Test handling of unknown question types."""
    print("\n" + "="*60)
    print("TEST: Unknown Question Types")
    print("="*60)
    
    learner = get_trained_learner(learner)
    qa = QuestionAnswerer(learner)
    
    questions = [
        ['why', 'does', 'the', 'dog', 'run'],  # 'why' not supported
        ['when', 'does', 'the', 'cat', 'sleep'],  # 'when' not supported
    ]
    
    print("\nAsking unsupported question types:")
    for q in questions:
        answer = qa.answer(q)
        print(f"  Q: {' '.join(q)}")
        print(f"  A: {answer}")
        print()
    
    # Should return "I don't understand" for unknown types
    answer = qa.answer(['why', 'does', 'the', 'dog', 'run'])
    handles_unknown = "don't understand" in answer.lower()
    
    if handles_unknown:
        print("✓ Unknown question types handled gracefully")
    else:
        print("✗ Unknown question types not handled properly")
    
    return handles_unknown


def run_all_tests(learner=None):
    """Run all comprehension tests with optional shared learner."""
    global _shared_learner
    
    print("\n" + "="*60)
    print("EMERGENT LANGUAGE LEARNER - COMPREHENSION TESTS")
    print("="*60)
    
    if learner is not None:
        _shared_learner = learner
        print("(Using shared pre-trained model)")
    
    tests = [
        test_what_questions,
        test_who_questions,
        test_does_questions_positive,
        test_does_questions_negative,
        test_knowledge_summary,
        test_unknown_question_types,
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


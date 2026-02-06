#!/usr/bin/env python
"""
Test Sentence Parsing
=====================

Version: 2.0.0
Date: 2025-11-30

Tests that the parser correctly:
1. Identifies subjects, verbs, and objects
2. Handles pronouns as subjects
3. Assigns thematic roles (agent, patient)
4. Handles intransitive and transitive sentences

Now uses SHARED trained model for faster test runs.

Run:
    uv run python test_parser.py
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from nemo.language.emergent import (
    EmergentLanguageLearner,
    create_training_data,
    SentenceParser,
    ParseResult,
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


def test_basic_parsing(learner=None):
    """Test basic sentence parsing."""
    print("\n" + "="*60)
    print("TEST: Basic Sentence Parsing")
    print("="*60)
    
    learner = get_trained_learner(learner)
    parser = SentenceParser(learner)
    
    test_sentences = [
        (['the', 'dog', 'runs'], 'dog', 'runs', None),
        (['the', 'cat', 'sleeps'], 'cat', 'sleeps', None),
        (['the', 'bird', 'jumps'], 'bird', 'jumps', None),
    ]
    
    print("\nIntransitive sentences:")
    all_correct = True
    
    for words, expected_subj, expected_verb, expected_obj in test_sentences:
        result = parser.parse(words)
        
        subj_ok = result.subject == expected_subj
        verb_ok = result.verb == expected_verb
        obj_ok = result.object == expected_obj
        
        status = "✓" if (subj_ok and verb_ok and obj_ok) else "✗"
        if not (subj_ok and verb_ok and obj_ok):
            all_correct = False
        
        print(f"  {status} '{' '.join(words)}'")
        print(f"      Subject: {result.subject} (expected: {expected_subj})")
        print(f"      Verb: {result.verb} (expected: {expected_verb})")
        print(f"      Object: {result.object} (expected: {expected_obj})")
    
    if all_correct:
        print("\n✓ Basic parsing works correctly")
    return all_correct


def test_transitive_parsing(learner=None):
    """Test transitive sentence parsing."""
    print("\n" + "="*60)
    print("TEST: Transitive Sentence Parsing")
    print("="*60)
    
    learner = get_trained_learner(learner)
    parser = SentenceParser(learner)
    
    test_sentences = [
        (['the', 'cat', 'chases', 'the', 'bird'], 'cat', 'chases', 'bird'),
        (['the', 'dog', 'sees', 'the', 'cat'], 'dog', 'sees', 'cat'),
        (['the', 'girl', 'eats', 'the', 'food'], 'girl', 'eats', 'food'),
    ]
    
    print("\nTransitive sentences:")
    all_correct = True
    
    for words, expected_subj, expected_verb, expected_obj in test_sentences:
        result = parser.parse(words)
        
        subj_ok = result.subject == expected_subj
        verb_ok = result.verb == expected_verb
        obj_ok = result.object == expected_obj
        
        status = "✓" if (subj_ok and verb_ok and obj_ok) else "✗"
        if not (subj_ok and verb_ok and obj_ok):
            all_correct = False
        
        print(f"  {status} '{' '.join(words)}'")
        print(f"      Subject: {result.subject} (expected: {expected_subj})")
        print(f"      Verb: {result.verb} (expected: {expected_verb})")
        print(f"      Object: {result.object} (expected: {expected_obj})")
    
    if all_correct:
        print("\n✓ Transitive parsing works correctly")
    return all_correct


def test_pronoun_parsing(learner=None):
    """Test that pronouns are correctly identified as subjects."""
    print("\n" + "="*60)
    print("TEST: Pronoun Parsing")
    print("="*60)
    
    learner = get_trained_learner(learner)
    parser = SentenceParser(learner)
    
    test_sentences = [
        (['she', 'sees', 'the', 'bird'], 'she', 'sees', 'bird'),
        (['he', 'runs'], 'he', 'runs', None),
    ]
    
    print("\nSentences with pronouns:")
    all_correct = True
    
    for words, expected_subj, expected_verb, expected_obj in test_sentences:
        result = parser.parse(words)
        
        subj_ok = result.subject == expected_subj
        verb_ok = result.verb == expected_verb
        obj_ok = result.object == expected_obj
        
        status = "✓" if (subj_ok and verb_ok and obj_ok) else "✗"
        if not (subj_ok and verb_ok and obj_ok):
            all_correct = False
        
        print(f"  {status} '{' '.join(words)}'")
        print(f"      Subject: {result.subject} (expected: {expected_subj})")
        print(f"      Verb: {result.verb} (expected: {expected_verb})")
        print(f"      Object: {result.object} (expected: {expected_obj})")
        print(f"      Categories: {result.word_categories}")
    
    if all_correct:
        print("\n✓ Pronoun parsing works correctly")
    return all_correct


def test_thematic_roles(learner=None):
    """Test that thematic roles are assigned correctly."""
    print("\n" + "="*60)
    print("TEST: Thematic Role Assignment")
    print("="*60)
    
    learner = get_trained_learner(learner)
    parser = SentenceParser(learner)
    
    # Test a transitive sentence
    words = ['the', 'cat', 'chases', 'the', 'bird']
    result = parser.parse(words)
    
    print(f"\nSentence: '{' '.join(words)}'")
    print(f"  Agent: {result.agent}")
    print(f"  Patient: {result.patient}")
    print(f"  Word roles: {result.word_roles}")
    
    # Agent should be the subject
    agent_ok = result.agent == result.subject
    # Patient should be the object
    patient_ok = result.patient == result.object
    
    all_correct = agent_ok and patient_ok
    
    if all_correct:
        print("\n✓ Thematic roles assigned correctly")
    else:
        print("\n✗ Thematic roles not correct")
    
    return all_correct


def test_parse_confidence(learner=None):
    """Test that parse confidence reflects learned patterns."""
    print("\n" + "="*60)
    print("TEST: Parse Confidence")
    print("="*60)
    
    learner = get_trained_learner(learner)
    parser = SentenceParser(learner)
    
    # Learned pattern should have high confidence
    learned = ['the', 'dog', 'runs']
    result_learned = parser.parse(learned)
    
    # Novel pattern might have lower confidence
    novel = ['the', 'table', 'runs']  # Tables don't run in training
    result_novel = parser.parse(novel)
    
    print(f"\nLearned pattern: '{' '.join(learned)}'")
    print(f"  Confidence: {result_learned.confidence:.2f}")
    
    print(f"\nNovel pattern: '{' '.join(novel)}'")
    print(f"  Confidence: {result_novel.confidence:.2f}")
    
    # Learned should have higher or equal confidence
    confidence_ok = result_learned.confidence >= result_novel.confidence
    
    if confidence_ok:
        print("\n✓ Confidence reflects learned patterns")
    else:
        print("\n✗ Confidence not working as expected")
    
    return confidence_ok


def test_parse_result_structure(learner=None):
    """Test ParseResult dataclass structure."""
    print("\n" + "="*60)
    print("TEST: ParseResult Structure")
    print("="*60)
    
    learner = get_trained_learner(learner)
    parser = SentenceParser(learner)
    
    words = ['the', 'big', 'cat', 'eats', 'quickly']
    result = parser.parse(words)
    
    print(f"\nSentence: '{' '.join(words)}'")
    print(f"\nParseResult fields:")
    print(f"  subject: {result.subject}")
    print(f"  verb: {result.verb}")
    print(f"  object: {result.object}")
    print(f"  agent: {result.agent}")
    print(f"  patient: {result.patient}")
    print(f"  adjectives: {result.adjectives}")
    print(f"  adverbs: {result.adverbs}")
    print(f"  determiners: {result.determiners}")
    print(f"  confidence: {result.confidence}")
    print(f"  word_categories: {result.word_categories}")
    
    # Test to_dict
    d = result.to_dict()
    print(f"\nto_dict() works: {isinstance(d, dict)}")
    
    # Test __str__
    print(f"__str__(): {result}")
    
    print("\n✓ ParseResult structure is correct")
    return True


def run_all_tests(learner=None):
    """Run all parser tests with optional shared learner."""
    global _shared_learner
    
    print("\n" + "="*60)
    print("EMERGENT LANGUAGE LEARNER - PARSER TESTS")
    print("="*60)
    
    if learner is not None:
        _shared_learner = learner
        print("(Using shared pre-trained model)")
    
    tests = [
        test_basic_parsing,
        test_transitive_parsing,
        test_pronoun_parsing,
        test_thematic_roles,
        test_parse_confidence,
        test_parse_result_structure,
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


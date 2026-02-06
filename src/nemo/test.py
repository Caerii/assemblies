"""
NEMO Test Suite
===============

Quick tests to verify the modular system works correctly.
"""

import time
from nemo import LanguageLearner, SentenceGenerator, Brain, BrainParams
from nemo.language import CurriculumLearner, StructureType, StructureDetector


def test_core():
    """Test core brain components."""
    print("Testing core...")
    
    brain = Brain(BrainParams(n=1000))
    brain.add_area("TEST")
    
    # Create and project assembly
    assembly = brain.random_assembly()
    output = brain.project("TEST", assembly)
    
    assert len(output) == brain.p.k
    print("  Core: OK")


def test_learner():
    """Test language learner."""
    print("Testing learner...")
    
    learner = LanguageLearner(verbose=False)
    
    # Train
    for _ in range(10):
        learner.hear_sentence(['a', 'b', 'c'])
    
    assert learner.sentences_seen == 10
    assert 'a' in learner.get_vocabulary()
    assert learner.get_word_category('a') == 0
    assert learner.get_word_category('b') == 1
    assert learner.get_word_category('c') == 2
    
    print("  Learner: OK")


def test_generator():
    """Test sentence generator."""
    print("Testing generator...")
    
    learner = LanguageLearner(verbose=False)
    
    for _ in range(20):
        learner.hear_sentence(['dog', 'runs', 'fast'])
        learner.hear_sentence(['cat', 'jumps', 'high'])
    
    generator = SentenceGenerator(learner)
    
    # Generate
    sent = generator.generate_sentence(3)
    assert len(sent) == 3
    
    # Score
    score = generator.score_sentence(['dog', 'runs', 'fast'])
    assert score > 0
    
    print("  Generator: OK")


def test_word_order_learning():
    """Test that different word orders are learned."""
    print("Testing word order learning...")
    
    # SVO
    svo = LanguageLearner(verbose=False)
    for _ in range(30):
        svo.hear_sentence(['subj', 'verb', 'obj'])
    
    # SOV
    sov = LanguageLearner(verbose=False)
    for _ in range(30):
        sov.hear_sentence(['subj', 'obj', 'verb'])
    
    # Check categories differ
    assert svo.get_word_category('verb') == 1  # Middle in SVO
    assert sov.get_word_category('verb') == 2  # End in SOV
    
    print("  Word order: OK")


def test_curriculum():
    """Test curriculum learning."""
    print("Testing curriculum...")
    
    cl = CurriculumLearner(verbose=False)
    
    # Train on simple structures
    cl.train_stage(StructureType.TRIPLE, n_sentences=50)
    
    # Check patterns were learned
    patterns = cl.learner.get_common_patterns(3)
    assert len(patterns) > 0
    
    print("  Curriculum: OK")


def test_structure_detection():
    """Test structure detection."""
    print("Testing structure detection...")
    
    learner = LanguageLearner(verbose=False)
    
    # Train with clear structure
    for _ in range(50):
        learner.hear_sentence(['the', 'dog', 'runs'])
        learner.hear_sentence(['a', 'cat', 'jumps'])
    
    detector = StructureDetector(learner)
    classes = detector.detect_word_classes()
    
    # Should have 3 classes (positions 0, 1, 2)
    assert len(classes) >= 2
    
    print("  Structure detection: OK")


def test_speed():
    """Benchmark speed."""
    print("Testing speed...")
    
    learner = LanguageLearner(verbose=False)
    
    start = time.perf_counter()
    for _ in range(1000):
        learner.hear_sentence(['dog', 'chases', 'cat'])
    elapsed = time.perf_counter() - start
    
    rate = 1000 / elapsed
    print(f"  Speed: {rate:.0f} sentences/sec")


if __name__ == "__main__":
    print("=" * 50)
    print("NEMO 2.0 Test Suite")
    print("=" * 50)
    
    test_core()
    test_learner()
    test_generator()
    test_word_order_learning()
    test_curriculum()
    test_structure_detection()
    test_speed()
    
    print()
    print("All tests passed!")


"""
Test Emergent Generation
========================

Tests the emergent response generation system.
"""

from src.nemo.language.emergent.learner import EmergentLanguageLearner
from src.nemo.language.emergent.curriculum import get_training_curriculum
from src.nemo.language.emergent.generation import EmergentGenerator, VPDecoder, ActivationSpreader
from src.nemo.language.emergent.areas import Area


def train_model():
    """Train a model for testing."""
    print("Training model...")
    learner = EmergentLanguageLearner(verbose=False)
    
    curriculum = get_training_curriculum(include_dialogue=True, stage=4, seed=42)
    
    for epoch in range(3):
        for sentence in curriculum:
            learner.present_grounded_sentence(
                sentence.words,
                sentence.contexts,
                sentence.roles,
                sentence.mood
            )
    
    print(f"  Trained on {learner.sentences_seen} sentences")
    print(f"  Vocabulary: {len(learner.word_count)} words")
    print(f"  VP patterns: {len(learner.brain.learned_assemblies[Area.VP])}")
    
    return learner


def test_vp_decoder(learner):
    """Test VP decoding."""
    print("\n" + "="*60)
    print("TEST: VP Decoder")
    print("="*60)
    
    decoder = VPDecoder(learner)
    
    # Test key-based decoding
    print("\nKey-based decoding:")
    vp_keys = list(learner.brain.learned_assemblies[Area.VP].keys())[:5]
    for key in vp_keys:
        sentence = decoder.decode_vp_key_to_sentence(key)
        print(f"  {key} -> {' '.join(sentence)}")
    
    # Test VP queries
    print("\nVP queries:")
    
    # Find VPs with a specific verb
    verb = 'runs'
    vps = decoder.find_vp_by_verb(verb)
    print(f"  VPs with '{verb}': {vps[:5]}...")
    
    # Find VPs with a specific subject
    subject = 'dog'
    vps = decoder.find_vp_by_subject(subject)
    print(f"  VPs with subject '{subject}': {vps[:5]}...")
    
    print("✓ VP Decoder works")
    return True


def test_activation_spreading(learner):
    """Test activation spreading."""
    print("\n" + "="*60)
    print("TEST: Activation Spreading")
    print("="*60)
    
    spreader = ActivationSpreader(learner.brain)
    
    # Test spreading from a verb
    print("\nSpreading from verb 'runs':")
    activation = spreader.spread_from_verb('runs')
    
    for area, assembly in activation.items():
        if assembly is not None:
            print(f"  {area.name}: {len(assembly)} neurons active")
    
    # Test spreading from a noun
    print("\nSpreading from noun 'dog':")
    activation = spreader.spread_from_noun('dog')
    
    for area, assembly in activation.items():
        if assembly is not None:
            print(f"  {area.name}: {len(assembly)} neurons active")
    
    print("✓ Activation spreading works")
    return True


def test_emergent_generator(learner):
    """Test emergent response generation."""
    print("\n" + "="*60)
    print("TEST: Emergent Generator")
    print("="*60)
    
    generator = EmergentGenerator(learner)
    
    # Test "who" questions
    print("\n'Who' questions:")
    questions = [
        ['who', 'runs'],
        ['who', 'drinks'],
        ['who', 'sleeps'],
    ]
    for q in questions:
        answer = generator.generate_answer(q)
        print(f"  Q: {' '.join(q)}")
        print(f"  A: {answer}")
    
    # Test "what" questions
    print("\n'What' questions:")
    questions = [
        ['what', 'does', 'the', 'dog', 'do'],
        ['what', 'does', 'the', 'cat', 'do'],
    ]
    for q in questions:
        answer = generator.generate_answer(q)
        print(f"  Q: {' '.join(q)}")
        print(f"  A: {answer}")
    
    # Test "does" questions
    print("\n'Does' questions:")
    questions = [
        ['does', 'the', 'dog', 'run'],
        ['does', 'the', 'cat', 'fly'],
    ]
    for q in questions:
        answer = generator.generate_answer(q)
        print(f"  Q: {' '.join(q)}")
        print(f"  A: {answer}")
    
    print("✓ Emergent generator works")
    return True


def test_comparison(learner):
    """Compare emergent vs template-based generation."""
    print("\n" + "="*60)
    print("TEST: Emergent vs Template Comparison")
    print("="*60)
    
    from src.nemo.language.emergent.parser.comprehension import QuestionAnswerer
    
    emergent = EmergentGenerator(learner)
    template = QuestionAnswerer(learner)
    
    questions = [
        ['who', 'runs'],
        ['what', 'does', 'the', 'dog', 'do'],
        ['does', 'the', 'cat', 'sleep'],
    ]
    
    print("\nComparison:")
    for q in questions:
        q_str = ' '.join(q)
        emergent_answer = emergent.generate_answer(q)
        template_answer = template.answer(q)
        
        print(f"\n  Q: {q_str}")
        print(f"  Emergent:  {emergent_answer}")
        print(f"  Template:  {template_answer}")
    
    print("\n✓ Comparison complete")
    return True


def run_all_tests():
    """Run all generation tests."""
    print("="*60)
    print("EMERGENT GENERATION TEST SUITE")
    print("="*60)
    
    # Train model
    learner = train_model()
    
    tests = [
        ("VP Decoder", lambda: test_vp_decoder(learner)),
        ("Activation Spreading", lambda: test_activation_spreading(learner)),
        ("Emergent Generator", lambda: test_emergent_generator(learner)),
        ("Comparison", lambda: test_comparison(learner)),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == '__main__':
    run_all_tests()



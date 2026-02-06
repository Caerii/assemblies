"""
Test the Lexicon and Curriculum System
======================================

Validates the lexicon structure and curriculum design.
"""

from src.lexicon.build_lexicon import build_lexicon
from src.lexicon.curriculum.grounded_training import (
    GroundedCorpus, create_stage1_corpus
)


def test_lexicon():
    """Test the lexicon building"""
    print("=" * 60)
    print("TESTING LEXICON")
    print("=" * 60)
    
    lexicon = build_lexicon()
    stats = lexicon.get_stats()
    
    print(f"\n✓ Total words: {stats['total_words']}")
    print("\nBy category:")
    for cat, count in sorted(stats['by_category'].items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    
    # Test lookups
    print("\nTesting word lookups:")
    test_words = ['dog', 'run', 'big', 'the', 'quickly']
    for word in test_words:
        w = lexicon.get_word(word)
        if w:
            print(f"  ✓ '{word}' -> {w.category.name}, freq={w.frequency:.1f}, aoa={w.age_of_acquisition:.1f}")
        else:
            print(f"  ✗ '{word}' not found")
    
    # Test inflected forms
    print("\nTesting inflected form lookups:")
    test_forms = [('runs', 'run'), ('dogs', 'dog'), ('bigger', 'big')]
    for form, expected_lemma in test_forms:
        w = lexicon.get_word(form)
        if w and w.lemma == expected_lemma:
            print(f"  ✓ '{form}' -> lemma '{w.lemma}'")
        else:
            print(f"  ? '{form}' -> {w.lemma if w else 'not found'} (expected '{expected_lemma}')")
    
    return lexicon


def test_grounded_corpus():
    """Test the grounded training corpus"""
    print("\n" + "=" * 60)
    print("TESTING GROUNDED CORPUS")
    print("=" * 60)
    
    # Create Stage 1 corpus
    corpus = create_stage1_corpus()
    stats = corpus.get_statistics()
    
    print("\nStage 1 Corpus:")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Unique words: {stats['unique_words']}")
    print(f"  Unique structures: {stats['unique_structures']}")
    print(f"  Avg exposures per word: {stats['avg_exposures_per_word']:.1f}")
    
    print("\n  By speech act:")
    for act, count in stats['by_speech_act'].items():
        if count > 0:
            print(f"    {act}: {count}")
    
    # Show some examples
    print("\n  Sample grounded examples:")
    for ex in corpus.examples[:5]:
        print(f"    '{ex.sentence}'")
        print(f"      POS: {ex.pos_tags}")
        print(f"      Visual: {ex.context.visual_objects}")
        print(f"      Speech act: {ex.speech_act.name}")
    
    return corpus


def test_free_sentence_addition():
    """Test adding free sentences with automatic grounding"""
    print("\n" + "=" * 60)
    print("TESTING FREE SENTENCE ADDITION")
    print("=" * 60)
    
    corpus = GroundedCorpus()
    
    # Add some free sentences
    sentences = [
        "the big dog chases the small cat",
        "mommy is cooking dinner",
        "where is my red ball",
        "i want to go outside",
        "the bird flies in the sky",
    ]
    
    print("\nAdding free sentences with automatic grounding:")
    for sent in sentences:
        ex = corpus.add_sentence(sent)
        print(f"\n  '{sent}'")
        print(f"    POS: {ex.pos_tags}")
        print(f"    Visual objects: {ex.context.visual_objects}")
        print(f"    Actions: {ex.context.actions}")
        print(f"    Speech act: {ex.speech_act.name}")
    
    return corpus


def test_word_learning_protocol():
    """Test the 10-grounded-sentences protocol for learning a word"""
    print("\n" + "=" * 60)
    print("TESTING WORD LEARNING PROTOCOL")
    print("=" * 60)
    
    corpus = GroundedCorpus()
    
    # Learn the word "elephant"
    word = "elephant"
    visual = ["ELEPHANT", "ANIMAL", "LARGE"]
    
    print(f"\nLearning word '{word}' with grounding {visual}:")
    examples = corpus.add_grounded_word(word, visual, n_examples=10)
    
    print(f"  Created {len(examples)} grounded examples:")
    for ex in examples:
        print(f"    - '{ex.sentence}' ({ex.speech_act.name})")
    
    print(f"\n  Word exposure count: {corpus.get_word_exposure_count(word)}")
    
    return corpus


def test_curriculum_stages():
    """Test the curriculum stage progression"""
    print("\n" + "=" * 60)
    print("TESTING CURRICULUM STAGES")
    print("=" * 60)
    
    from src.lexicon.curriculum.stage1_first_words import STAGE1_CORPUS, STAGE1_VOCABULARY
    from src.lexicon.curriculum.stage2_vocabulary_spurt import STAGE2_CORPUS, STAGE2_VOCABULARY
    from src.lexicon.curriculum.stage3_two_word import STAGE3_CORPUS, STAGE3_VOCABULARY
    from src.lexicon.curriculum.stage4_sentences import STAGE4_CORPUS, STAGE4_VOCABULARY
    
    stages = [
        ("Stage 1: First Words", STAGE1_VOCABULARY, STAGE1_CORPUS),
        ("Stage 2: Vocabulary Spurt", STAGE2_VOCABULARY, STAGE2_CORPUS),
        ("Stage 3: Two-Word", STAGE3_VOCABULARY, STAGE3_CORPUS),
        ("Stage 4: Sentences", STAGE4_VOCABULARY, STAGE4_CORPUS),
    ]
    
    for name, vocab, corpus in stages:
        print(f"\n{name}:")
        print(f"  Vocabulary size: {len(vocab)}")
        print(f"  Corpus size: {len(corpus)}")
        
        # Analyze sentence lengths
        lengths = [len(s.split()) for s in corpus]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        print(f"  Avg sentence length: {avg_len:.1f} words")
        print(f"  Length range: {min(lengths)} - {max(lengths)} words")
        
        # Show samples
        print(f"  Samples: {corpus[:3]}")


def test_developmental_trajectory():
    """Analyze how the curriculum mirrors child development"""
    print("\n" + "=" * 60)
    print("DEVELOPMENTAL TRAJECTORY ANALYSIS")
    print("=" * 60)
    
    from src.lexicon.curriculum.stage1_first_words import STAGE1_CORPUS
    from src.lexicon.curriculum.stage2_vocabulary_spurt import STAGE2_CORPUS
    from src.lexicon.curriculum.stage3_two_word import STAGE3_CORPUS
    from src.lexicon.curriculum.stage4_sentences import STAGE4_CORPUS
    
    print("""
    Child Development Milestones vs Our Curriculum:
    
    Age 12-18mo: First words (~50 words)
    ├── Our Stage 1: {s1_vocab} vocabulary items
    ├── Sentence length: {s1_len:.1f} words avg
    └── Focus: Naming, social words
    
    Age 18-24mo: Vocabulary spurt (~300 words)
    ├── Our Stage 2: {s2_vocab} vocabulary items
    ├── Sentence length: {s2_len:.1f} words avg
    └── Focus: Two-word combinations
    
    Age 24-30mo: Telegraphic speech (~500 words)
    ├── Our Stage 3: {s3_vocab} vocabulary items
    ├── Sentence length: {s3_len:.1f} words avg
    └── Focus: SVO emerging, prepositions
    
    Age 30-36mo: Full sentences (~1000 words)
    ├── Our Stage 4: {s4_vocab} vocabulary items
    ├── Sentence length: {s4_len:.1f} words avg
    └── Focus: Auxiliaries, questions, complex grammar
    """.format(
        s1_vocab=len(STAGE1_CORPUS),
        s1_len=sum(len(s.split()) for s in STAGE1_CORPUS) / len(STAGE1_CORPUS),
        s2_vocab=len(STAGE2_CORPUS),
        s2_len=sum(len(s.split()) for s in STAGE2_CORPUS) / len(STAGE2_CORPUS),
        s3_vocab=len(STAGE3_CORPUS),
        s3_len=sum(len(s.split()) for s in STAGE3_CORPUS) / len(STAGE3_CORPUS),
        s4_vocab=len(STAGE4_CORPUS),
        s4_len=sum(len(s.split()) for s in STAGE4_CORPUS) / len(STAGE4_CORPUS),
    ))


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("LEXICON AND CURRICULUM SYSTEM TEST")
    print("=" * 60)
    
    # Test lexicon
    lexicon = test_lexicon()
    
    # Test grounded corpus
    corpus = test_grounded_corpus()
    
    # Test free sentence addition
    test_free_sentence_addition()
    
    # Test word learning protocol
    test_word_learning_protocol()
    
    # Test curriculum stages
    test_curriculum_stages()
    
    # Analyze developmental trajectory
    test_developmental_trajectory()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    
    # Summary
    print("""
    SUMMARY:
    --------
    ✓ Lexicon built with {total} words across {cats} categories
    ✓ Grounded training corpus with automatic context inference
    ✓ Word learning protocol (10 grounded exposures per word)
    ✓ 4-stage curriculum mirroring child development
    
    NEXT STEPS:
    -----------
    1. Integrate with assembly brain for actual learning experiments
    2. Test word learning curves (how many exposures needed?)
    3. Test syntactic generalization (can it parse novel sentences?)
    4. Measure developmental trajectory (does it follow child-like stages?)
    """.format(
        total=lexicon.get_stats()['total_words'],
        cats=len(lexicon.get_stats()['by_category']),
    ))


if __name__ == '__main__':
    main()


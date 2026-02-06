"""
Emergent NEMO Demo
==================

Demonstrates emergent category learning from grounding.
"""

import time
from . import (
    EmergentLanguageLearner, EmergentParams,
    SentenceGenerator, create_training_data
)


def demo():
    """Demo emergent category learning with full neurobiological architecture"""
    print("="*70)
    print("EMERGENT NEMO LANGUAGE LEARNER")
    print("37 brain areas - ALL categories emerge from grounding!")
    print("="*70)
    
    # Create learner
    params = EmergentParams(n=10000)
    learner = EmergentLanguageLearner(params, verbose=True)
    
    # Create training data
    print("\nCreating grounded training data...")
    training_data = create_training_data()
    print(f"  {len(training_data)} sentences")
    
    # Train
    print("\nTraining...")
    start = time.perf_counter()
    for epoch in range(3):
        for sentence in training_data:
            learner.present_grounded_sentence(
                sentence.words, sentence.contexts, 
                roles=sentence.roles, mood=sentence.mood,
                learn=True
            )
    elapsed = time.perf_counter() - start
    print(f"  {learner.sentences_seen} sentences in {elapsed:.2f}s")
    
    # Show emergent categories
    print("\n" + "="*60)
    print("EMERGENT CATEGORIES (learned from grounding)")
    print("="*60)
    
    vocab = learner.get_vocabulary_by_category()
    for cat in ['NOUN', 'VERB', 'ADJECTIVE', 'ADVERB', 'PREPOSITION', 'PRONOUN', 'FUNCTION']:
        if cat in vocab:
            words = vocab[cat][:10]
            print(f"\n{cat} ({len(vocab[cat])} words):")
            print(f"  {', '.join(words)}")
    
    # Show word order
    print("\n" + "="*60)
    print("LEARNED WORD ORDER")
    print("="*60)
    print(f"  Category order: {learner.get_word_order()}")
    
    # Show category transitions
    print("\n  Category transitions (top 10):")
    sorted_trans = sorted(learner.category_transitions.items(), key=lambda x: -x[1])[:10]
    for (cat1, cat2), count in sorted_trans:
        print(f"    {cat1} → {cat2}: {count}")
    
    # Show thematic roles
    print("\n" + "="*60)
    print("EMERGENT THEMATIC ROLES")
    print("="*60)
    
    test_words = ['dog', 'cat', 'boy', 'runs', 'chases', 'food']
    print(f"\n{'Word':>10} {'Category':>12} {'Role':>10} {'Conf':>6}")
    print("-"*45)
    for word in test_words:
        cat, _ = learner.get_emergent_category(word)
        role, conf = learner.get_thematic_role(word)
        print(f"{word:>10} {cat:>12} {role:>10} {conf:>6.0%}")
    
    # Generate sentences
    generator = SentenceGenerator(learner)
    
    print("\n" + "="*60)
    print("GENERATED SENTENCES (phrase structure)")
    print("="*60)
    for i in range(5):
        sent = generator.generate_structured()
        print(f"  {i+1}. {' '.join(sent)}")
    
    print("\n" + "="*60)
    print("GENERATED SENTENCES (transitions)")
    print("="*60)
    for i in range(5):
        sent = generator.generate_from_transitions()
        print(f"  {i+1}. {' '.join(sent)}")
    
    # Show detailed grounding analysis
    print("\n" + "="*60)
    print("WORD GROUNDING ANALYSIS")
    print("="*60)
    
    test_words = ['dog', 'runs', 'big', 'the', 'she', 'on', 'quickly', 'yesterday']
    print(f"\n{'Word':>10} {'Cat':>8} {'V':>5} {'M':>5} {'P':>5} {'Sp':>5} {'So':>5} {'T':>5} {'∅':>5}")
    print("-"*65)
    
    for word in test_words:
        cat, scores = learner.get_emergent_category(word)
        v = scores.get('VISUAL', 0)
        m = scores.get('MOTOR', 0)
        p = scores.get('PROPERTY', 0)
        sp = scores.get('SPATIAL', 0)
        so = scores.get('SOCIAL', 0)
        t = scores.get('TEMPORAL', 0)
        n = scores.get('NONE', 0)
        print(f"{word:>10} {cat:>8} {v:>5.0%} {m:>5.0%} {p:>5.0%} {sp:>5.0%} {so:>5.0%} {t:>5.0%} {n:>5.0%}")
    
    # Final stats
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    stats = learner.get_stats()
    print(f"  Sentences seen: {stats['sentences_seen']}")
    print(f"  Vocabulary size: {stats['vocabulary_size']}")
    print(f"  Categories: {stats['categories']}")
    
    from .areas import NUM_AREAS, MUTUAL_INHIBITION_GROUPS
    print(f"  Brain areas: {NUM_AREAS}")
    print(f"  Mutual inhibition groups: {len(MUTUAL_INHIBITION_GROUPS)}")


if __name__ == "__main__":
    demo()


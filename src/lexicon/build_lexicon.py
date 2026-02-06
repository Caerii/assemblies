"""
Build the Full Lexicon
======================

Loads all word data and creates a complete lexicon with statistics.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.lexicon.lexicon_manager import LexiconManager, Word, WordCategory, SemanticDomain
from src.lexicon.statistics import WordStatistics, WordStats

# Import word data
from src.lexicon.data.nouns import NOUNS
from src.lexicon.data.verbs import VERBS
from src.lexicon.data.adjectives import ADJECTIVES
from src.lexicon.data.adverbs import ADVERBS
from src.lexicon.data.determiners import DETERMINERS
from src.lexicon.data.prepositions import PREPOSITIONS
from src.lexicon.data.pronouns import PRONOUNS
from src.lexicon.data.conjunctions import CONJUNCTIONS
from src.lexicon.data.auxiliaries import AUXILIARIES


def domain_from_string(s: str) -> SemanticDomain:
    """Convert string to SemanticDomain enum"""
    mapping = {
        'ANIMAL': SemanticDomain.ANIMAL,
        'PERSON': SemanticDomain.PERSON,
        'BODY_PART': SemanticDomain.BODY_PART,
        'FOOD': SemanticDomain.FOOD,
        'PLANT': SemanticDomain.PLANT,
        'OBJECT': SemanticDomain.OBJECT,
        'BUILDING': SemanticDomain.BUILDING,
        'VEHICLE': SemanticDomain.VEHICLE,
        'CLOTHING': SemanticDomain.CLOTHING,
        'TOOL': SemanticDomain.TOOL,
        'EMOTION': SemanticDomain.EMOTION,
        'COGNITION': SemanticDomain.COGNITION,
        'TIME': SemanticDomain.TIME,
        'SPACE': SemanticDomain.SPACE,
        'QUANTITY': SemanticDomain.QUANTITY,
        'QUALITY': SemanticDomain.QUALITY,
        'SOCIAL': SemanticDomain.SOCIAL,
        'MOTION': SemanticDomain.MOTION,
        'PERCEPTION': SemanticDomain.PERCEPTION,
        'COMMUNICATION': SemanticDomain.COMMUNICATION,
        'CONSUMPTION': SemanticDomain.CONSUMPTION,
        'CREATION': SemanticDomain.CREATION,
        'DESTRUCTION': SemanticDomain.DESTRUCTION,
        'POSSESSION': SemanticDomain.POSSESSION,
        'FUNCTION_WORD': SemanticDomain.FUNCTION_WORD,
    }
    return mapping.get(s, SemanticDomain.QUALITY)


def build_lexicon() -> LexiconManager:
    """Build the complete lexicon from all word data"""
    lexicon = LexiconManager()
    
    # Add nouns
    for data in NOUNS:
        domains = [domain_from_string(d) for d in data.get('domains', ['OBJECT'])]
        word = Word(
            lemma=data['lemma'],
            category=WordCategory.NOUN,
            forms=data.get('forms', {}),
            semantic_domains=domains,
            features=data.get('features', {}),
            frequency=data.get('freq', 0.0),
            age_of_acquisition=data.get('aoa', 5.0),
        )
        lexicon.add_word(word)
    
    # Add verbs
    for data in VERBS:
        domains = [domain_from_string(d) for d in data.get('domains', ['QUALITY'])]
        word = Word(
            lemma=data['lemma'],
            category=WordCategory.VERB,
            forms=data.get('forms', {}),
            semantic_domains=domains,
            features=data.get('features', {}),
            arguments=data.get('args', []),
            frequency=data.get('freq', 0.0),
            age_of_acquisition=data.get('aoa', 5.0),
        )
        lexicon.add_word(word)
    
    # Add adjectives
    for data in ADJECTIVES:
        domains = [domain_from_string(d) for d in data.get('domains', ['QUALITY'])]
        word = Word(
            lemma=data['lemma'],
            category=WordCategory.ADJECTIVE,
            forms=data.get('forms', {}),
            semantic_domains=domains,
            features=data.get('features', {}),
            frequency=data.get('freq', 0.0),
            age_of_acquisition=data.get('aoa', 5.0),
        )
        lexicon.add_word(word)
    
    # Add adverbs
    for data in ADVERBS:
        domains = [domain_from_string(d) for d in data.get('domains', ['QUALITY'])]
        word = Word(
            lemma=data['lemma'],
            category=WordCategory.ADVERB,
            forms=data.get('forms', {}),
            semantic_domains=domains,
            features=data.get('features', {}),
            frequency=data.get('freq', 0.0),
            age_of_acquisition=data.get('aoa', 5.0),
        )
        lexicon.add_word(word)
    
    # Add determiners
    for data in DETERMINERS:
        word = Word(
            lemma=data['lemma'],
            category=WordCategory.DETERMINER,
            forms=data.get('forms', {}),
            semantic_domains=[SemanticDomain.FUNCTION_WORD],
            features=data.get('features', {}),
            frequency=data.get('freq', 0.0),
            age_of_acquisition=data.get('aoa', 5.0),
        )
        lexicon.add_word(word)
    
    # Add prepositions
    for data in PREPOSITIONS:
        domains = [domain_from_string(d) for d in data.get('domains', ['FUNCTION_WORD'])]
        word = Word(
            lemma=data['lemma'],
            category=WordCategory.PREPOSITION,
            forms=data.get('forms', {}),
            semantic_domains=domains,
            features=data.get('features', {}),
            frequency=data.get('freq', 0.0),
            age_of_acquisition=data.get('aoa', 5.0),
        )
        lexicon.add_word(word)
    
    # Add pronouns
    for data in PRONOUNS:
        word = Word(
            lemma=data['lemma'],
            category=WordCategory.PRONOUN,
            forms=data.get('forms', {}),
            semantic_domains=[SemanticDomain.FUNCTION_WORD],
            features=data.get('features', {}),
            frequency=data.get('freq', 0.0),
            age_of_acquisition=data.get('aoa', 5.0),
        )
        lexicon.add_word(word)
    
    # Add conjunctions
    for data in CONJUNCTIONS:
        word = Word(
            lemma=data['lemma'],
            category=WordCategory.CONJUNCTION,
            forms=data.get('forms', {}),
            semantic_domains=[SemanticDomain.FUNCTION_WORD],
            features=data.get('features', {}),
            frequency=data.get('freq', 0.0),
            age_of_acquisition=data.get('aoa', 5.0),
        )
        lexicon.add_word(word)
    
    # Add auxiliaries
    for data in AUXILIARIES:
        cat = WordCategory.MODAL if data.get('features', {}).get('modal') else WordCategory.AUXILIARY
        word = Word(
            lemma=data['lemma'],
            category=cat,
            forms=data.get('forms', {}),
            semantic_domains=[SemanticDomain.FUNCTION_WORD],
            features=data.get('features', {}),
            frequency=data.get('freq', 0.0),
            age_of_acquisition=data.get('aoa', 5.0),
        )
        lexicon.add_word(word)
    
    return lexicon


def build_statistics(lexicon: LexiconManager) -> WordStatistics:
    """Build word statistics from the lexicon"""
    stats = WordStatistics()
    
    for lemma, word in lexicon.words.items():
        word_stats = WordStats(
            lemma=lemma,
            frequency=word.frequency,
            age_of_acquisition=word.age_of_acquisition,
        )
        stats.add_word_stats(word_stats)
    
    return stats


if __name__ == '__main__':
    print("Building lexicon...")
    lexicon = build_lexicon()
    
    print("\nLexicon Statistics:")
    stats = lexicon.get_stats()
    print(f"  Total words: {stats['total_words']}")
    print("\n  By category:")
    for cat, count in sorted(stats['by_category'].items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")
    
    print("\n  By semantic domain:")
    for dom, count in sorted(stats['by_domain'].items(), key=lambda x: -x[1])[:10]:
        print(f"    {dom}: {count}")
    
    # Show some examples
    print("\n  Sample words by age of acquisition:")
    for aoa in [2.0, 3.0, 4.0, 5.0]:
        words = lexicon.get_by_aoa(aoa)
        print(f"    AoA <= {aoa}: {len(words)} words")
        samples = [w.lemma for w in words[:5]]
        print(f"      Examples: {', '.join(samples)}")
    
    # Save lexicon
    print("\nSaving lexicon...")
    lexicon.save('full_lexicon.json')
    
    # Build and save statistics
    print("Building statistics...")
    word_stats = build_statistics(lexicon)
    word_stats.save('word_statistics.json')
    
    print("\nDone!")


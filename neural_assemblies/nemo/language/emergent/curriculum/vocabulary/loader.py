"""
Vocabulary Loader
=================

Loads vocabulary from the lexicon module or falls back to basic vocabulary.
"""

from typing import Dict

from ...params import GroundingContext
from .types import WordInfo
from .grounding import (
    create_grounding_from_domains,
    create_pronoun_grounding,
    create_question_word_grounding,
)


def load_from_lexicon() -> Dict[str, WordInfo]:
    """
    Load vocabulary from the lexicon/data module.
    
    Returns:
        Dictionary mapping lemmas to WordInfo
    """
    vocab = {}
    
    try:
        from lexicon.data.nouns import NOUNS
        from lexicon.data.verbs import VERBS
        from lexicon.data.adjectives import ADJECTIVES
        from lexicon.data.adverbs import ADVERBS
        from lexicon.data.pronouns import PRONOUNS
        from lexicon.data.prepositions import PREPOSITIONS
        from lexicon.data.determiners import DETERMINERS
        
        # Process nouns
        for noun_data in NOUNS:
            grounding = create_grounding_from_domains(
                noun_data.get('domains', []),
                noun_data.get('features', {})
            )
            vocab[noun_data['lemma'].lower()] = WordInfo(
                lemma=noun_data['lemma'].lower(),
                forms=noun_data.get('forms', {}),
                grounding=grounding,
                domains=noun_data.get('domains', []),
                features=noun_data.get('features', {}),
                frequency=noun_data.get('freq', 1.0),
                aoa=noun_data.get('aoa', 3.0),
            )
        
        # Process verbs
        for verb_data in VERBS:
            grounding = create_grounding_from_domains(
                verb_data.get('domains', []),
                verb_data.get('features', {})
            )
            vocab[verb_data['lemma'].lower()] = WordInfo(
                lemma=verb_data['lemma'].lower(),
                forms=verb_data.get('forms', {}),
                grounding=grounding,
                domains=verb_data.get('domains', []),
                features=verb_data.get('features', {}),
                frequency=verb_data.get('freq', 1.0),
                aoa=verb_data.get('aoa', 3.0),
            )
        
        # Process adjectives
        for adj_data in ADJECTIVES:
            lemma = adj_data['lemma'].lower()
            grounding = GroundingContext(properties=[lemma.upper()])
            vocab[lemma] = WordInfo(
                lemma=lemma,
                forms=adj_data.get('forms', {}),
                grounding=grounding,
                domains=adj_data.get('domains', ['QUALITY']),
                features=adj_data.get('features', {}),
                frequency=adj_data.get('freq', 1.0),
                aoa=adj_data.get('aoa', 3.0),
            )
        
        # Process adverbs
        for adv_data in ADVERBS:
            lemma = adv_data['lemma'].lower()
            grounding = GroundingContext(temporal=[lemma.upper()])
            vocab[lemma] = WordInfo(
                lemma=lemma,
                forms=adv_data.get('forms', {}),
                grounding=grounding,
                domains=adv_data.get('domains', ['TEMPORAL']),
                features=adv_data.get('features', {}),
                frequency=adv_data.get('freq', 1.0),
                aoa=adv_data.get('aoa', 3.0),
            )
        
        # Process pronouns
        for pron_data in PRONOUNS:
            lemma = pron_data['lemma'].lower()
            grounding = create_pronoun_grounding(lemma)
            vocab[lemma] = WordInfo(
                lemma=lemma,
                forms=pron_data.get('forms', {}),
                grounding=grounding,
                domains=['SOCIAL'],
                features=pron_data.get('features', {}),
                frequency=pron_data.get('freq', 1.0),
                aoa=pron_data.get('aoa', 2.0),
            )
        
        # Process prepositions
        for prep_data in PREPOSITIONS:
            lemma = prep_data['lemma'].lower()
            grounding = GroundingContext(spatial=[lemma.upper()])
            vocab[lemma] = WordInfo(
                lemma=lemma,
                forms={},
                grounding=grounding,
                domains=['SPACE'],
                features=prep_data.get('features', {}),
                frequency=prep_data.get('freq', 1.0),
                aoa=prep_data.get('aoa', 3.0),
            )
        
        # Process determiners (function words - no grounding)
        for det_data in DETERMINERS:
            lemma = det_data['lemma'].lower()
            vocab[lemma] = WordInfo(
                lemma=lemma,
                forms={},
                grounding=GroundingContext(),
                domains=[],
                features=det_data.get('features', {}),
                frequency=det_data.get('freq', 5.0),
                aoa=det_data.get('aoa', 2.0),
            )
        
        # Add question words with proper grounding
        for qword in ['what', 'who', 'where', 'when', 'why', 'how']:
            if qword not in vocab:
                vocab[qword] = WordInfo(
                    lemma=qword,
                    grounding=create_question_word_grounding(qword),
                    domains=['QUESTION'],
                )
        
        return vocab
        
    except ImportError as e:
        print(f"Warning: Could not import lexicon module: {e}")
        return {}


def load_basic_vocabulary() -> Dict[str, WordInfo]:
    """
    Load basic fallback vocabulary.
    
    Used when lexicon module is not available.
    """
    vocab = {}
    
    # Basic nouns
    basic_nouns = [
        ('dog', ['ANIMAL'], {'animate': True}),
        ('cat', ['ANIMAL'], {'animate': True}),
        ('bird', ['ANIMAL'], {'animate': True}),
        ('boy', ['PERSON'], {'animate': True, 'human': True}),
        ('girl', ['PERSON'], {'animate': True, 'human': True}),
        ('man', ['PERSON'], {'animate': True, 'human': True}),
        ('woman', ['PERSON'], {'animate': True, 'human': True}),
        ('ball', ['OBJECT'], {}),
        ('book', ['OBJECT'], {}),
        ('food', ['FOOD'], {}),
        ('table', ['FURNITURE'], {}),
        ('car', ['OBJECT'], {}),
        ('house', ['BUILDING'], {}),
        ('tree', ['PLANT'], {}),
    ]
    for lemma, domains, features in basic_nouns:
        grounding = create_grounding_from_domains(domains, features)
        vocab[lemma] = WordInfo(
            lemma=lemma, 
            grounding=grounding, 
            domains=domains, 
            features=features
        )
    
    # Basic verbs
    basic_verbs = [
        ('run', {'3sg': 'runs', 'past': 'ran', 'prog': 'running'}, ['MOTION'], {'intransitive': True}),
        ('walk', {'3sg': 'walks', 'past': 'walked', 'prog': 'walking'}, ['MOTION'], {'intransitive': True}),
        ('jump', {'3sg': 'jumps', 'past': 'jumped', 'prog': 'jumping'}, ['MOTION'], {'intransitive': True}),
        ('see', {'3sg': 'sees', 'past': 'saw', 'prog': 'seeing'}, ['PERCEPTION'], {'transitive': True}),
        ('eat', {'3sg': 'eats', 'past': 'ate', 'prog': 'eating'}, ['CONSUMPTION'], {'transitive': True}),
        ('sleep', {'3sg': 'sleeps', 'past': 'slept', 'prog': 'sleeping'}, ['CONSUMPTION'], {'intransitive': True}),
        ('chase', {'3sg': 'chases', 'past': 'chased', 'prog': 'chasing'}, ['MOTION'], {'transitive': True}),
        ('find', {'3sg': 'finds', 'past': 'found', 'prog': 'finding'}, ['PERCEPTION'], {'transitive': True}),
        ('know', {'3sg': 'knows', 'past': 'knew', 'prog': 'knowing'}, ['COGNITION'], {'transitive': True}),
        ('think', {'3sg': 'thinks', 'past': 'thought', 'prog': 'thinking'}, ['COGNITION'], {'transitive': True}),
        ('understand', {'3sg': 'understands', 'past': 'understood', 'prog': 'understanding'}, ['COGNITION'], {'transitive': True}),
        ('learn', {'3sg': 'learns', 'past': 'learned', 'prog': 'learning'}, ['COGNITION'], {'transitive': True}),
        ('say', {'3sg': 'says', 'past': 'said', 'prog': 'saying'}, ['COMMUNICATION'], {'transitive': True}),
    ]
    for lemma, forms, domains, features in basic_verbs:
        grounding = create_grounding_from_domains(domains, features)
        vocab[lemma] = WordInfo(
            lemma=lemma, 
            forms=forms, 
            grounding=grounding, 
            domains=domains,
            features=features
        )
    
    # Basic adjectives
    basic_adj = ['big', 'small', 'red', 'blue', 'fast', 'slow', 'good', 'bad', 'happy', 'sad']
    for adj in basic_adj:
        vocab[adj] = WordInfo(
            lemma=adj,
            grounding=GroundingContext(properties=[adj.upper()]),
            domains=['QUALITY']
        )
    
    # Pronouns
    for lemma in ['i', 'you', 'he', 'she', 'it', 'we', 'they']:
        vocab[lemma] = WordInfo(
            lemma=lemma,
            grounding=create_pronoun_grounding(lemma),
            domains=['SOCIAL']
        )
    
    # Function words (no grounding)
    for word in ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'and', 'or', 'but']:
        vocab[word] = WordInfo(lemma=word, grounding=GroundingContext())
    
    # Question words
    for qword in ['what', 'who', 'where', 'when', 'why', 'how']:
        vocab[qword] = WordInfo(
            lemma=qword,
            grounding=create_question_word_grounding(qword),
            domains=['QUESTION']
        )
    
    # Response words
    vocab['yes'] = WordInfo(lemma='yes', grounding=GroundingContext(emotional=['AFFIRM']))
    vocab['no'] = WordInfo(lemma='no', grounding=GroundingContext(emotional=['NEGATE']))
    
    return vocab


__all__ = ['load_from_lexicon', 'load_basic_vocabulary']



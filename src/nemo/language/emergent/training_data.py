"""
Grounded Training Data
======================

Version: 2.0.0
Date: 2025-12-01

Training data with grounding contexts for emergent category learning.
"""

from typing import List
from .params import GroundingContext, GroundedSentence

__all__ = ['create_training_data', 'create_simple_training_data']


def create_simple_training_data() -> List[GroundedSentence]:
    """Create simple training sentences."""
    data = []
    
    # Basic SVO sentences
    nouns = ['dog', 'cat', 'bird', 'ball', 'book', 'boy', 'girl', 'food', 'table', 'car']
    verbs = ['runs', 'sees', 'eats', 'plays', 'sleeps', 'jumps', 'walks', 'reads', 'chases', 'finds']
    
    for noun in nouns:
        for verb in ['runs', 'sleeps', 'jumps', 'walks', 'plays']:
            data.append(GroundedSentence(
                words=['the', noun, verb],
                contexts=[
                    GroundingContext(),
                    GroundingContext(visual=[noun.upper(), 'OBJECT']),
                    GroundingContext(motor=[verb.upper(), 'ACTION']),
                ],
                roles=[None, 'agent', 'action'],
                mood='declarative'
            ))
    
    # Transitive sentences
    for noun1 in nouns[:5]:
        for verb in ['sees', 'chases', 'finds', 'eats']:
            for noun2 in nouns[5:]:
                data.append(GroundedSentence(
                    words=['the', noun1, verb, 'the', noun2],
                    contexts=[
                        GroundingContext(),
                        GroundingContext(visual=[noun1.upper(), 'OBJECT']),
                        GroundingContext(motor=[verb.upper(), 'ACTION']),
                        GroundingContext(),
                        GroundingContext(visual=[noun2.upper(), 'OBJECT']),
                    ],
                    roles=[None, 'agent', 'action', None, 'patient'],
                    mood='declarative'
                ))
    
    return data


def create_training_data() -> List[GroundedSentence]:
    """Create comprehensive training data with all word types."""
    data = []
    
    # === DECLARATIVE SENTENCES ===
    
    data.append(GroundedSentence(
        words=['the', 'dog', 'runs'],
        contexts=[
            GroundingContext(),
            GroundingContext(visual=['DOG', 'ANIMAL']),
            GroundingContext(motor=['RUNNING', 'MOTION']),
        ],
        roles=[None, 'agent', 'action'],
        mood='declarative'
    ))
    
    data.append(GroundedSentence(
        words=['the', 'cat', 'chases', 'the', 'bird'],
        contexts=[
            GroundingContext(),
            GroundingContext(visual=['CAT', 'ANIMAL']),
            GroundingContext(motor=['CHASING', 'PURSUIT']),
            GroundingContext(),
            GroundingContext(visual=['BIRD', 'ANIMAL']),
        ],
        roles=[None, 'agent', 'action', None, 'patient'],
        mood='declarative'
    ))
    
    data.append(GroundedSentence(
        words=['a', 'big', 'cat', 'sleeps'],
        contexts=[
            GroundingContext(),
            GroundingContext(properties=['SIZE', 'BIG']),
            GroundingContext(visual=['CAT', 'ANIMAL']),
            GroundingContext(motor=['SLEEPING', 'REST']),
        ],
        roles=[None, None, 'agent', 'action'],
        mood='declarative'
    ))
    
    data.append(GroundedSentence(
        words=['the', 'ball', 'is', 'red'],
        contexts=[
            GroundingContext(),
            GroundingContext(visual=['BALL', 'OBJECT']),
            GroundingContext(),
            GroundingContext(properties=['COLOR', 'RED']),
        ],
        roles=[None, 'agent', 'action', None],
        mood='declarative'
    ))
    
    data.append(GroundedSentence(
        words=['she', 'sees', 'the', 'bird'],
        contexts=[
            GroundingContext(social=['PERSON', 'FEMALE']),
            GroundingContext(motor=['SEEING', 'PERCEPTION']),
            GroundingContext(),
            GroundingContext(visual=['BIRD', 'ANIMAL']),
        ],
        roles=['agent', 'action', None, 'patient'],
        mood='declarative'
    ))
    
    data.append(GroundedSentence(
        words=['the', 'boy', 'eats', 'food', 'quickly'],
        contexts=[
            GroundingContext(),
            GroundingContext(visual=['BOY', 'PERSON'], social=['PERSON', 'CHILD']),
            GroundingContext(motor=['EATING', 'CONSUMPTION']),
            GroundingContext(visual=['FOOD', 'OBJECT']),
            GroundingContext(temporal=['QUICK', 'MANNER']),
        ],
        roles=[None, 'agent', 'action', 'patient', None],
        mood='declarative'
    ))
    
    data.append(GroundedSentence(
        words=['the', 'cat', 'is', 'on', 'the', 'table'],
        contexts=[
            GroundingContext(),
            GroundingContext(visual=['CAT', 'ANIMAL']),
            GroundingContext(),
            GroundingContext(spatial=['ON', 'ABOVE']),
            GroundingContext(),
            GroundingContext(visual=['TABLE', 'FURNITURE']),
        ],
        roles=[None, 'agent', 'action', None, None, None],
        mood='declarative'
    ))
    
    data.append(GroundedSentence(
        words=['he', 'and', 'she', 'play'],
        contexts=[
            GroundingContext(social=['PERSON', 'MALE']),
            GroundingContext(),
            GroundingContext(social=['PERSON', 'FEMALE']),
            GroundingContext(motor=['PLAYING', 'ACTION']),
        ],
        roles=['agent', None, 'agent', 'action'],
        mood='declarative'
    ))
    
    data.append(GroundedSentence(
        words=['yesterday', 'the', 'dog', 'ran'],
        contexts=[
            GroundingContext(temporal=['PAST', 'TIME']),
            GroundingContext(),
            GroundingContext(visual=['DOG', 'ANIMAL']),
            GroundingContext(motor=['RUNNING', 'MOTION']),
        ],
        roles=[None, None, 'agent', 'action'],
        mood='declarative'
    ))
    
    # === INTERROGATIVE ===
    
    data.append(GroundedSentence(
        words=['does', 'the', 'dog', 'run'],
        contexts=[
            GroundingContext(),
            GroundingContext(),
            GroundingContext(visual=['DOG', 'ANIMAL']),
            GroundingContext(motor=['RUNNING', 'MOTION']),
        ],
        roles=[None, None, 'agent', 'action'],
        mood='interrogative'
    ))
    
    data.append(GroundedSentence(
        words=['what', 'does', 'the', 'cat', 'see'],
        contexts=[
            GroundingContext(),
            GroundingContext(),
            GroundingContext(),
            GroundingContext(visual=['CAT', 'ANIMAL']),
            GroundingContext(motor=['SEEING', 'PERCEPTION']),
        ],
        roles=['patient', None, None, 'agent', 'action'],
        mood='interrogative'
    ))
    
    # === IMPERATIVE ===
    
    data.append(GroundedSentence(
        words=['run'],
        contexts=[GroundingContext(motor=['RUNNING', 'MOTION'])],
        roles=['action'],
        mood='imperative'
    ))
    
    data.append(GroundedSentence(
        words=['eat', 'the', 'food'],
        contexts=[
            GroundingContext(motor=['EATING', 'CONSUMPTION']),
            GroundingContext(),
            GroundingContext(visual=['FOOD', 'OBJECT']),
        ],
        roles=['action', None, 'patient'],
        mood='imperative'
    ))
    
    # === GENERATED EXAMPLES ===
    
    # IMPORTANT: Separate animate from inanimate nouns
    # Only animate things can be AGENTS of action verbs
    animate_nouns = ['dog', 'cat', 'bird', 'boy', 'girl']  # Can do actions
    inanimate_nouns = ['ball', 'book', 'food', 'table', 'car']  # Cannot do actions
    all_nouns = animate_nouns + inanimate_nouns
    
    action_verbs = ['runs', 'sleeps', 'jumps', 'walks', 'plays', 'eats', 'chases', 'finds']
    perception_verbs = ['sees', 'reads']  # These can have animate subjects
    
    adjectives = ['big', 'small', 'red', 'blue', 'fast', 'slow', 'good', 'bad', 'happy', 'sad']
    prepositions = ['on', 'in', 'under', 'near', 'behind']
    adverbs = ['quickly', 'slowly', 'happily', 'sadly']
    
    # Intransitive with ANIMATE subjects only
    for noun in animate_nouns:
        for verb in ['runs', 'sleeps', 'jumps', 'walks', 'plays']:
            data.append(GroundedSentence(
                words=['the', noun, verb],
                contexts=[
                    GroundingContext(),
                    GroundingContext(visual=[noun.upper(), 'ANIMAL']),  # Animate
                    GroundingContext(motor=[verb.upper(), 'ACTION']),
                ],
                roles=[None, 'agent', 'action'],
                mood='declarative'
            ))
    
    # Transitive: ANIMATE subject, any object
    for noun1 in animate_nouns:
        for verb in ['sees', 'chases', 'finds', 'eats']:
            for noun2 in all_nouns:
                if noun1 != noun2:
                    data.append(GroundedSentence(
                        words=['the', noun1, verb, 'the', noun2],
                        contexts=[
                            GroundingContext(),
                            GroundingContext(visual=[noun1.upper(), 'ANIMAL']),  # Animate agent
                            GroundingContext(motor=[verb.upper(), 'ACTION']),
                            GroundingContext(),
                            GroundingContext(visual=[noun2.upper(), 'OBJECT']),  # Any object
                        ],
                        roles=[None, 'agent', 'action', None, 'patient'],
                        mood='declarative'
                    ))
    
    # Adjective + noun (any noun can have adjectives)
    for adj in adjectives:
        for noun in all_nouns[:5]:
            data.append(GroundedSentence(
                words=['the', adj, noun],
                contexts=[
                    GroundingContext(),
                    GroundingContext(properties=[adj.upper(), 'PROPERTY']),
                    GroundingContext(visual=[noun.upper(), 'OBJECT']),
                ],
                roles=[None, None, None],
                mood='declarative'
            ))
    
    # Prepositional phrases (any noun can be in a location)
    for noun1 in all_nouns[:3]:
        for prep in prepositions:
            for noun2 in all_nouns[3:6]:
                data.append(GroundedSentence(
                    words=['the', noun1, 'is', prep, 'the', noun2],
                    contexts=[
                        GroundingContext(),
                        GroundingContext(visual=[noun1.upper(), 'OBJECT']),
                        GroundingContext(),
                        GroundingContext(spatial=[prep.upper(), 'LOCATION']),
                        GroundingContext(),
                        GroundingContext(visual=[noun2.upper(), 'OBJECT']),
                    ],
                    roles=[None, 'agent', 'action', None, None, None],
                    mood='declarative'
                ))
    
    # Adverb sentences - only ANIMATE subjects with action verbs
    for noun in animate_nouns[:3]:
        for verb in ['runs', 'walks', 'eats']:
            for adv in adverbs:
                data.append(GroundedSentence(
                    words=['the', noun, verb, adv],
                    contexts=[
                        GroundingContext(),
                        GroundingContext(visual=[noun.upper(), 'ANIMAL']),  # Animate
                        GroundingContext(motor=[verb.upper(), 'ACTION']),
                        GroundingContext(temporal=[adv.upper(), 'MANNER']),
                    ],
                    roles=[None, 'agent', 'action', None],
                    mood='declarative'
                ))
    
    return data


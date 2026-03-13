"""
Stage 2: Vocabulary Spurt (18-24 months equivalent)
===================================================

~200-300 words with:
- More nouns (objects, animals, people)
- Basic verbs (action words)
- Simple adjectives (big, little, hot, cold)
- Two-word combinations emerging

Training focuses on:
- Rapid noun learning
- Verb-noun combinations
- Simple property words
"""

STAGE2_VOCABULARY = [
    # Expanded animals
    'horse', 'cow', 'pig', 'chicken', 'duck', 'rabbit', 'bear', 'lion',
    'elephant', 'monkey', 'frog', 'butterfly', 'bee',
    
    # More body parts
    'head', 'arm', 'leg', 'finger', 'toe', 'hair', 'face', 'ear',
    
    # Clothing
    'shirt', 'pants', 'dress', 'hat', 'coat', 'sock',
    
    # More food
    'bread', 'cheese', 'egg', 'meat', 'rice', 'soup', 'cake',
    'orange', 'grape', 'strawberry',
    
    # Vehicles
    'car', 'truck', 'bus', 'train', 'plane', 'boat', 'bike',
    
    # Nature
    'tree', 'flower', 'grass', 'sun', 'moon', 'star', 'rain', 'snow',
    
    # Furniture
    'table', 'chair', 'couch', 'lamp',
    
    # More actions
    'sit', 'stand', 'jump', 'throw', 'catch', 'push', 'pull',
    'open', 'close', 'wash', 'brush', 'kick', 'hug', 'kiss',
    
    # More descriptors
    'happy', 'sad', 'angry', 'tired', 'hungry', 'thirsty',
    'fast', 'slow', 'loud', 'quiet', 'clean', 'dirty',
    'wet', 'dry', 'hard', 'soft', 'new', 'old',
    
    # Colors
    'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
    'black', 'white', 'brown',
    
    # More determiners/pronouns
    'that', 'these', 'those', 'some', 'all',
    'he', 'she', 'we', 'they',
    
    # Question words
    'what', 'where', 'who',
]

# Two-word combinations (characteristic of this stage)
STAGE2_CORPUS = [
    # Noun + Noun (compound-like)
    'baby shoe',
    'dog food',
    'apple juice',
    
    # Adjective + Noun
    'big truck',
    'little bird',
    'red ball',
    'blue car',
    'hot soup',
    'cold milk',
    'dirty shoe',
    'clean hand',
    
    # Agent + Action
    'dog run',
    'cat sleep',
    'baby cry',
    'bird fly',
    'mommy go',
    'daddy work',
    
    # Action + Object
    'eat cookie',
    'drink juice',
    'throw ball',
    'read book',
    'wash hand',
    'brush teeth',
    
    # Negation
    'no sleep',
    'no eat',
    'not hot',
    'not dirty',
    
    # Possession
    'my ball',
    'your shoe',
    'mommy book',
    'baby toy',
    
    # Location (emerging)
    'ball there',
    'dog here',
    'up there',
    'in box',
    
    # Recurrence
    'more milk',
    'more cookie',
    'again play',
    
    # Questions
    'what that',
    'where ball',
    'where mommy',
    'who that',
]

STAGE2_PATTERNS = {
    'single_word': 0.2,
    'two_word': 0.6,
    'three_word': 0.2,
}

STAGE2_REPETITIONS = 15


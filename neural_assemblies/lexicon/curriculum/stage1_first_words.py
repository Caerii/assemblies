"""
Stage 1: First Words (12-18 months equivalent)
==============================================

~50 core words focused on:
- Social words (hi, bye, no, yes)
- Family (mama, dada)
- Common objects (ball, book)
- Animals (dog, cat)
- Food (milk, juice)
- Actions (go, eat, sleep)

Training is single-word focused with high repetition.
"""

# Core vocabulary for Stage 1
STAGE1_VOCABULARY = [
    # Social/Interjections
    'hi', 'bye', 'no', 'yes', 'please', 'more', 'up', 'down',
    
    # People
    'mom', 'dad', 'baby',
    
    # Animals  
    'dog', 'cat', 'bird', 'fish',
    
    # Body parts
    'eye', 'nose', 'mouth', 'hand', 'foot',
    
    # Food/Drink
    'milk', 'juice', 'water', 'cookie', 'apple', 'banana',
    
    # Objects
    'ball', 'book', 'toy', 'cup', 'shoe',
    
    # Places
    'home', 'bed',
    
    # Actions (simple)
    'go', 'eat', 'drink', 'sleep', 'play', 'run', 'walk',
    
    # Descriptors
    'big', 'little', 'hot', 'cold', 'good', 'bad',
    
    # Determiners
    'the', 'a', 'my',
]

# Training corpus - single words and simple two-word combos
STAGE1_CORPUS = [
    # Single word utterances (most common at this stage)
    'ball',
    'dog',
    'cat',
    'milk',
    'mom',
    'dad',
    'up',
    'down',
    'more',
    'no',
    'yes',
    'hi',
    'bye',
    'baby',
    'book',
    'water',
    'juice',
    'cookie',
    'apple',
    'shoe',
    'eye',
    'nose',
    
    # Simple two-word combinations
    'more milk',
    'more cookie',
    'my ball',
    'my book',
    'big dog',
    'little cat',
    'go home',
    'eat cookie',
    'drink milk',
    'drink juice',
    'play ball',
    'good dog',
    'bad dog',
    'hot water',
    'cold milk',
    'bye bye',
    'hi mom',
    'hi dad',
    'up up',
    'no no',
    
    # Naming/Pointing
    'the dog',
    'the cat',
    'the ball',
    'a book',
    'a cookie',
    'my mom',
    'my dad',
    'my toy',
]

# Training patterns for this stage
STAGE1_PATTERNS = {
    'single_word': 0.6,      # 60% single words
    'two_word': 0.3,         # 30% two-word combinations
    'three_word': 0.1,       # 10% three-word (rare)
}

# Repetition schedule (how many times each word should be seen)
STAGE1_REPETITIONS = 20


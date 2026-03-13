"""
Stage 4: Sentence Formation (30-36 months equivalent)
=====================================================

~1000 words with:
- Full SVO sentences
- Determiners used consistently
- Auxiliaries emerging (is, are, can, will)
- Past tense emerging (-ed)
- Plurals used correctly
- Questions with inversion

Characteristic patterns:
- "The dog is running"
- "I want to go"
- "Where is my ball?"
- "The cat eated the food" (overgeneralization errors)
"""

STAGE4_VOCABULARY = [
    # More abstract concepts
    'idea', 'problem', 'question', 'answer', 'reason', 'truth',
    'life', 'world', 'country', 'city', 'school', 'home',
    
    # Emotions
    'happy', 'sad', 'angry', 'scared', 'excited', 'surprised',
    'worried', 'proud', 'sorry', 'glad',
    
    # More verbs
    'become', 'seem', 'appear', 'remain', 'stay',
    'begin', 'end', 'continue', 'happen', 'change',
    'remember', 'forget', 'understand', 'believe', 'decide',
    'learn', 'teach', 'explain', 'describe',
    
    # Auxiliaries
    'is', 'are', 'was', 'were', 'am',
    'has', 'have', 'had',
    'do', 'does', 'did',
    'can', 'could', 'will', 'would', 'should', 'must',
    
    # Conjunctions
    'and', 'but', 'or', 'because', 'so', 'if', 'when', 'while',
    
    # More prepositions
    'through', 'across', 'along', 'around', 'against',
    'during', 'until', 'since', 'before', 'after',
]

# Full sentences with proper grammar
STAGE4_CORPUS = [
    # Simple present
    'the dog runs fast',
    'the cat sleeps on the bed',
    'the bird flies in the sky',
    'the baby plays with toys',
    'mommy reads a book',
    'daddy drives the car',
    
    # Present progressive
    'the dog is running',
    'the cat is sleeping',
    'i am eating lunch',
    'she is reading a book',
    'they are playing outside',
    'we are going home',
    
    # Simple past
    'the dog ran away',
    'the cat ate the food',
    'i saw a bird',
    'she gave me a cookie',
    'he went to school',
    'we played in the park',
    
    # Past progressive
    'the dog was running',
    'i was eating dinner',
    'they were playing games',
    
    # Future
    'i will go to school',
    'she will come tomorrow',
    'we will play later',
    'the dog will eat soon',
    
    # Modal verbs
    'i can run fast',
    'she can swim',
    'he can read books',
    'we can play together',
    'you should eat vegetables',
    'i must go now',
    
    # Questions (yes/no)
    'is the dog sleeping',
    'are you hungry',
    'can i have a cookie',
    'will you help me',
    'do you like apples',
    'does she have a dog',
    
    # Questions (wh-)
    'where is my ball',
    'what is that',
    'who is at the door',
    'why is the baby crying',
    'when will daddy come home',
    'how does this work',
    
    # Negation
    'the dog is not sleeping',
    'i do not want that',
    'she cannot come today',
    'we will not go there',
    'he did not eat lunch',
    
    # Conjunctions
    'i like apples and oranges',
    'she is happy but tired',
    'do you want milk or juice',
    'i am sad because my toy broke',
    'we will play if it stops raining',
    
    # Complex sentences
    'i want to go to the park',
    'she likes to read books',
    'he needs to eat dinner',
    'they want to play outside',
    
    # Relative clauses (simple)
    'the dog that is brown is mine',
    'the book that i read was good',
    'the man who is tall is my dad',
]

# Characteristic errors (overgeneralization)
STAGE4_ERRORS = [
    ('the dog goed home', 'the dog went home'),  # Past tense overgeneralization
    ('i eated my food', 'i ate my food'),
    ('she runned fast', 'she ran fast'),
    ('the mouses are small', 'the mice are small'),  # Plural overgeneralization
    ('two foots', 'two feet'),
    ('me want cookie', 'i want a cookie'),  # Case errors
    ('him is nice', 'he is nice'),
]

STAGE4_PATTERNS = {
    'three_word': 0.2,
    'four_word': 0.3,
    'five_word': 0.3,
    'six_plus': 0.2,
}

STAGE4_REPETITIONS = 10


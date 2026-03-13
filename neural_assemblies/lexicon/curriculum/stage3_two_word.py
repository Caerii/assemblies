"""
Stage 3: Two-Word Stage (24-30 months equivalent)
=================================================

~500 words with:
- Telegraphic speech (content words, few function words)
- Simple SVO emerging
- More verbs and prepositions
- Basic pronouns

Characteristic patterns:
- "Mommy go" (agent-action)
- "Eat cookie" (action-object)
- "Big doggie" (modifier-noun)
- "More juice" (recurrence)
- "No bed" (negation)
"""

STAGE3_VOCABULARY = [
    # More abstract nouns
    'time', 'day', 'night', 'morning', 'thing', 'way', 'place',
    'name', 'word', 'story', 'song', 'game', 'picture',
    
    # More verbs
    'want', 'need', 'like', 'love', 'hate', 'know', 'think',
    'see', 'look', 'hear', 'listen', 'feel', 'touch',
    'give', 'take', 'bring', 'put', 'hold', 'carry',
    'make', 'build', 'draw', 'paint', 'cut',
    'find', 'hide', 'show', 'help', 'try',
    'wait', 'stop', 'start', 'finish',
    
    # Prepositions
    'in', 'on', 'under', 'over', 'behind', 'beside', 'between',
    'to', 'from', 'with', 'for', 'about',
    
    # More pronouns
    'me', 'him', 'her', 'us', 'them',
    'mine', 'yours', 'his', 'hers',
    
    # Adverbs
    'now', 'then', 'here', 'there', 'too', 'very',
    'again', 'always', 'never', 'sometimes',
    
    # Question words
    'why', 'how', 'when',
]

# Telegraphic speech patterns
STAGE3_CORPUS = [
    # Agent-Action-Object (SVO emerging)
    'mommy read book',
    'daddy throw ball',
    'baby drink milk',
    'doggie eat food',
    'kitty catch mouse',
    'i want cookie',
    'i need help',
    'you give me',
    'he push me',
    'she take toy',
    
    # Action-Object with pronouns
    'want that',
    'see it',
    'give me',
    'help me',
    'show me',
    
    # Modifier-Noun-Verb
    'big dog run',
    'little cat sleep',
    'red ball bounce',
    
    # Location phrases
    'ball on table',
    'cat under bed',
    'dog in house',
    'toy in box',
    'book on floor',
    'mommy in kitchen',
    
    # Possession
    'my toy',
    'your ball',
    'mommy shoe',
    'daddy car',
    'baby bottle',
    
    # Negation (more complex)
    'no want that',
    'not my toy',
    'doggie no bite',
    'baby not sleep',
    
    # Questions
    'where mommy go',
    'what that noise',
    'who that man',
    'why baby cry',
    'how do that',
    
    # Time expressions
    'now eat',
    'later play',
    'again do',
    
    # Conjunctions emerging
    'mommy and daddy',
    'ball and book',
    'eat and drink',
]

STAGE3_PATTERNS = {
    'single_word': 0.1,
    'two_word': 0.4,
    'three_word': 0.4,
    'four_word': 0.1,
}

STAGE3_REPETITIONS = 12


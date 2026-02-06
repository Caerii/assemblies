# Neurobiologically Plausible Brain Areas for NEMO

Based on the papers (Mitropolsky & Papadimitriou 2025, parser.py, learner.py) and neuroscience.

## Complete Area List

### 1. INPUT/SENSORY AREAS (Grounding)

| Area | Purpose | Neurobiological Basis |
|------|---------|----------------------|
| **PHON** | Phonological input | Auditory cortex, Wernicke's area |
| **VISUAL** | Visual grounding (objects) | Visual cortex, inferotemporal cortex |
| **MOTOR** | Motor grounding (actions) | Motor cortex, mirror neurons |
| **AUDITORY** | Non-speech sounds | Auditory cortex |
| **TACTILE** | Touch sensations | Somatosensory cortex |
| **PROPRIOCEPTIVE** | Body position | Parietal cortex |
| **OLFACTORY** | Smell | Olfactory cortex |
| **GUSTATORY** | Taste | Insular cortex |

### 2. SEMANTIC/CONCEPTUAL AREAS

| Area | Purpose | Neurobiological Basis |
|------|---------|----------------------|
| **PROPERTY** | Properties (size, color) | Temporal-parietal junction |
| **SPATIAL** | Spatial relations | Parietal cortex |
| **TEMPORAL** | Time concepts | Prefrontal cortex |
| **QUANTITY** | Numbers, amounts | Intraparietal sulcus |
| **SOCIAL** | Social concepts, people | Temporal pole, medial PFC |
| **EMOTION** | Emotional concepts | Amygdala, insula |
| **ABSTRACT** | Abstract concepts | Angular gyrus |

### 3. LEXICAL AREAS

| Area | Purpose | Neurobiological Basis |
|------|---------|----------------------|
| **LEX_CONTENT** | Content words (N, V, Adj) | Middle temporal gyrus |
| **LEX_FUNCTION** | Function words (det, conj) | Inferior frontal gyrus |
| **LEX1** | Nouns (strong → Visual) | Anterior temporal lobe |
| **LEX2** | Verbs (strong → Motor) | Posterior temporal, frontal |

### 4. CATEGORY/CORE AREAS (Grammatical Categories)

| Area | Purpose | Emergent From |
|------|---------|---------------|
| **NOUN_CORE** | Noun category | Visual grounding consistency |
| **VERB_CORE** | Verb category | Motor grounding consistency |
| **ADJ_CORE** | Adjective category | Property grounding |
| **ADV_CORE** | Adverb category | Manner/degree grounding |
| **DET_CORE** | Determiner category | High freq + no grounding |
| **PREP_CORE** | Preposition category | Spatial grounding |
| **PRON_CORE** | Pronoun category | Social grounding |
| **CONJ_CORE** | Conjunction category | No grounding + linking |
| **AUX_CORE** | Auxiliary category | Tense/aspect grounding |

### 5. SYNTACTIC ROLE AREAS

| Area | Purpose | Mutual Inhibition |
|------|---------|-------------------|
| **SUBJ** | Subject NP | Yes (with OBJ, IOBJ) |
| **OBJ** | Direct object NP | Yes |
| **IOBJ** | Indirect object NP | Yes |
| **PRED** | Predicate | No |

### 6. PHRASE STRUCTURE AREAS

| Area | Purpose | Combines |
|------|---------|----------|
| **NP** | Noun phrase | DET + ADJ + N |
| **VP** | Verb phrase | V + NP/PP |
| **PP** | Prepositional phrase | PREP + NP |
| **ADJP** | Adjective phrase | ADV + ADJ |
| **ADVP** | Adverb phrase | ADV + ADV |
| **SENT** | Full sentence | NP + VP |
| **COMP** | Complement clause | COMP + SENT |

### 7. MODIFIER AREAS

| Area | Purpose | Attaches To |
|------|---------|-------------|
| **DET** | Determiners | NP |
| **ADJ** | Adjectives | NP |
| **ADV** | Adverbs | VP, ADJ, ADV |
| **PREP** | Prepositions | NP → PP |
| **PREP_P** | Prep phrase | VP, NP |

### 8. THEMATIC ROLE AREAS

| Area | Purpose | Semantic Role |
|------|---------|---------------|
| **ROLE_AGENT** | Agent/doer | Who does action |
| **ROLE_PATIENT** | Patient/undergoer | Who is affected |
| **ROLE_THEME** | Theme/moved | What is moved/changed |
| **ROLE_EXPERIENCER** | Experiencer | Who feels/perceives |
| **ROLE_GOAL** | Goal/destination | Where to |
| **ROLE_SOURCE** | Source/origin | Where from |
| **ROLE_LOCATION** | Location | Where at |
| **ROLE_INSTRUMENT** | Instrument | With what |
| **ROLE_BENEFICIARY** | Beneficiary | For whom |

### 9. SEQUENCE/CONTROL AREAS

| Area | Purpose | Function |
|------|---------|----------|
| **SEQ** | Sequence memory | Word order learning |
| **MOOD** | Sentence mood | Declarative, interrogative, imperative |
| **TENSE** | Tense marking | Past, present, future |
| **ASPECT** | Aspect marking | Perfective, progressive |
| **POLARITY** | Polarity | Affirmative, negative |
| **FOCUS** | Information focus | What's emphasized |
| **TOPIC** | Topic marking | What's being talked about |

### 10. ERROR/CONTROL AREAS

| Area | Purpose | Function |
|------|---------|----------|
| **ERROR** | Parse error detection | Wobbly assembly = error |
| **INHIBIT** | Global inhibition | Competition control |

## Fiber Types

### Strong Fibers (Higher p, β)
- PHON ↔ LEX
- VISUAL ↔ LEX1 (nouns)
- MOTOR ↔ LEX2 (verbs)
- LEX ↔ CORE areas

### Regular Fibers
- LEX → Syntactic areas
- Syntactic areas ↔ each other
- CORE → SEQ
- MOOD → SEQ

### Weak Fibers (Lower p, β)
- Cross-modal (VISUAL ↔ LEX2)
- Long-distance dependencies

## Mutual Inhibition Groups

1. **Role areas**: SUBJ, OBJ, IOBJ (only one active)
2. **Thematic roles**: AGENT, PATIENT, THEME (compete)
3. **Mood**: Declarative, Interrogative, Imperative (exclusive)
4. **Polarity**: Affirmative, Negative (exclusive)

## What's Currently Missing in emergent_learner.py

1. **CORE areas** - For grammatical category abstraction
2. **Thematic role areas** - AGENT, PATIENT, etc.
3. **Phrase structure** - NP, VP, PP, SENT
4. **Tense/Aspect/Mood** - TENSE, ASPECT, MOOD
5. **Multiple semantic areas** - TEMPORAL, QUANTITY, etc.
6. **Error detection** - ERROR area
7. **Inhibition control** - Fiber/area states

## Recommended Minimal Set for Emergent Learning

```python
class Area(Enum):
    # Input (8 areas)
    PHON = 0
    VISUAL = 1
    MOTOR = 2
    PROPERTY = 3
    SPATIAL = 4
    TEMPORAL = 5
    SOCIAL = 6
    EMOTION = 7
    
    # Lexical (2 areas)
    LEX_CONTENT = 8
    LEX_FUNCTION = 9
    
    # Core/Category (8 areas) - EMERGE from grounding
    NOUN_CORE = 10
    VERB_CORE = 11
    ADJ_CORE = 12
    ADV_CORE = 13
    PREP_CORE = 14
    DET_CORE = 15
    PRON_CORE = 16
    CONJ_CORE = 17
    
    # Thematic Roles (6 areas) - Mutual inhibition
    ROLE_AGENT = 18
    ROLE_PATIENT = 19
    ROLE_THEME = 20
    ROLE_GOAL = 21
    ROLE_SOURCE = 22
    ROLE_LOCATION = 23
    
    # Phrase Structure (5 areas)
    NP = 24
    VP = 25
    PP = 26
    ADJP = 27
    SENT = 28
    
    # Syntactic Roles (3 areas) - Mutual inhibition
    SUBJ = 29
    OBJ = 30
    IOBJ = 31
    
    # Sequence/Control (4 areas)
    SEQ = 32
    MOOD = 33
    TENSE = 34
    POLARITY = 35
    
    # Error (1 area)
    ERROR = 36
```

**Total: 37 areas** (vs current 13)

## Key Principles

1. **Categories EMERGE from grounding** - No pre-labels
2. **Thematic roles learned from argument structure** - Not hardcoded
3. **Phrase structure from co-occurrence** - NP = things that appear with DET
4. **Mood/Tense from context** - Question intonation, temporal markers
5. **Mutual inhibition for competition** - Only one role active at a time
6. **Stability-based classification** - Wobbly = wrong category


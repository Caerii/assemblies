# Grammatical Structure Learning with Assembly Calculus

## Overview

Based on analysis of `parser.py`, `recursive_parser.py`, and `learner.py`, here's how grammatical structure is properly learned using Assembly Calculus.

## Robust Implementation

We've created `src/text_generation/robust_grammatical_brain.py` with:

### Brain Areas

| Area | Type | Purpose |
|------|------|---------|
| **LEX** | Explicit | Lexicon (128 words) |
| **NOUN_CORE** | Explicit | Noun category representation |
| **VERB_CORE** | Explicit | Verb category representation |
| **ADJ_CORE** | Explicit | Adjective category representation |
| **DET_CORE** | Explicit | Determiner category representation |
| **PREP_CORE** | Explicit | Preposition category representation |
| **ADV_CORE** | Explicit | Adverb category representation |
| **SUBJ, OBJ, IOBJ** | Non-explicit | Subject, Object, Indirect Object |
| **VERB, COMP** | Non-explicit | Verb phrase, Complement |
| **DET, ADJ, ADV** | Non-explicit | Modifiers |
| **PREP, PREP_P** | Non-explicit | Preposition, Prepositional phrase |
| **SEQ** | Non-explicit | Sequence memory (word order) |
| **MOOD** | Explicit | Sentence mood (declarative, interrogative, etc.) |
| **TENSE** | Explicit | Tense marking |
| **VISUAL, MOTOR, ABSTRACT, EMOTION** | Explicit | Context/grounding |
| **ERROR** | Non-explicit | Parse error detection |

### Word Order Learning Results

Training on SVO sentences like "the dog chases the cat":

```
DET_CORE -> NOUN_CORE: 10  (determiners followed by nouns)
NOUN_CORE -> VERB_CORE: 6  (nouns followed by verbs)
VERB_CORE -> DET_CORE: 4   (verbs followed by determiners)
VERB_CORE -> ADV_CORE: 2   (verbs followed by adverbs)
```

### Error Detection via "Wobbly" Assemblies

The ERROR area uses low plasticity (β/2) to detect impossible parses:
- **Stable assembly** (overlap > 0.5 after recurrence) = valid parse
- **Unstable assembly** (overlap < 0.5) = parsing error

```python
def measure_assembly_stability(area, rounds=3):
    initial_winners = area.winners
    for _ in range(rounds):
        project({}, {area: [area]})  # Recurrent
    final_winners = area.winners
    return overlap(initial_winners, final_winners)
```

## Key Architecture Components

### 1. Brain Areas

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GRAMMATICAL BRAIN                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │    LEX      │────▶│    SUBJ     │────▶│    VERB     │           │
│  │  (Lexicon)  │     │  (Subject)  │     │   (Verb)    │           │
│  │  EXPLICIT   │     │ NON-EXPLICIT│     │ NON-EXPLICIT│           │
│  └─────────────┘     └─────────────┘     └─────────────┘           │
│        │                   │                   │                    │
│        │                   │                   ▼                    │
│        │             ┌─────────────┐     ┌─────────────┐           │
│        └────────────▶│    DET      │     │    OBJ      │           │
│                      │(Determiner) │     │  (Object)   │           │
│                      └─────────────┘     └─────────────┘           │
│                                                                      │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │    ADJ      │     │   PREP      │     │   PREP_P    │           │
│  │ (Adjective) │     │(Preposition)│     │(Prep Phrase)│           │
│  └─────────────┘     └─────────────┘     └─────────────┘           │
│                                                                      │
│  ┌─────────────┐     ┌─────────────┐                                │
│  │   ADVERB    │     │    MOOD     │                                │
│  │  (Adverb)   │     │ (Sentence   │                                │
│  │             │     │   Mood)     │                                │
│  └─────────────┘     └─────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 2. Explicit vs Non-Explicit Areas

| Area Type | Examples | Assemblies | Purpose |
|-----------|----------|------------|---------|
| **Explicit** | LEX, PHON | Pre-defined, fixed | Store vocabulary |
| **Non-Explicit** | SUBJ, OBJ, VERB | Emerge dynamically | Represent syntactic roles |

### 3. Fiber and Area States

The key innovation from `parser.py` is **inhibition control**:

```python
# Fiber states: control which areas can project to which
fiber_states[area1][area2] = {0}  # Inhibited (index 0 present)
fiber_states[area1][area2] = {}   # Disinhibited (empty set)

# Area states: control which areas can fire
area_states[area] = {0}  # Inhibited
area_states[area] = {}   # Disinhibited
```

### 4. PRE_RULES and POST_RULES

Each word type has rules that control the parsing:

```python
# Noun example
def generic_noun(index):
    return {
        "PRE_RULES": [
            # Before processing noun: enable LEX → SUBJ/OBJ fibers
            FiberRule(DISINHIBIT, LEX, SUBJ, 0),
            FiberRule(DISINHIBIT, LEX, OBJ, 0),
            # Enable DET → noun areas
            FiberRule(DISINHIBIT, DET, SUBJ, 0),
        ],
        "POST_RULES": [
            # After processing: close the fibers
            FiberRule(INHIBIT, LEX, SUBJ, 0),
            FiberRule(INHIBIT, LEX, OBJ, 0),
            # Inhibit modifier areas
            AreaRule(INHIBIT, DET, 0),
            AreaRule(INHIBIT, ADJ, 0),
        ],
    }
```

## Parsing Flow

### Example: "the dog chases the cat"

```
Step 1: "the" (determiner)
├── PRE_RULES: Disinhibit DET area, LEX→DET fiber
├── Activate "the" in LEX
├── Project: LEX → DET
├── POST_RULES: Inhibit LEX→DET fiber
└── DET now has assembly for "the"

Step 2: "dog" (noun)
├── PRE_RULES: Disinhibit LEX→SUBJ, DET→SUBJ fibers
├── Activate "dog" in LEX
├── Project: LEX → SUBJ, DET → SUBJ
├── POST_RULES: Inhibit fibers, inhibit DET area
└── SUBJ has merged assembly: "the" + "dog"

Step 3: "chases" (verb)
├── PRE_RULES: Disinhibit LEX→VERB, VERB→SUBJ fibers
├── Activate "chases" in LEX
├── Project: LEX → VERB, VERB ↔ SUBJ
├── POST_RULES: Disinhibit OBJ, inhibit SUBJ
└── VERB has assembly linked to SUBJ

Step 4-5: "the cat" (object)
├── Similar to steps 1-2 but into OBJ area
└── OBJ has merged assembly: "the" + "cat"
```

## Readout

After parsing, we can read out the structure:

```python
# Project VERB → LEX to get the verb word
project({}, {VERB: [LEX]})
verb_word = get_word(LEX)  # "chases"

# Project VERB → SUBJ → LEX to get subject
project({}, {VERB: [SUBJ]})
project({}, {SUBJ: [LEX]})
subj_word = get_word(LEX)  # "dog"

# Project VERB → OBJ → LEX to get object
project({}, {VERB: [OBJ]})
project({}, {OBJ: [LEX]})
obj_word = get_word(LEX)  # "cat"
```

## Word Order Learning (from learner.py)

The `SEQ` (sequence) area learns word order:

```python
# SEQ area remembers the order words appeared
# CORE areas represent grammatical categories (NOUN_CORE, VERB_CORE)

# Training: "dog runs" (Subject-Verb order)
1. Activate MOOD[declarative] → SEQ
2. Activate "dog" in LEX → NOUN_CORE → SEQ
3. Activate "runs" in LEX → VERB_CORE → SEQ
4. Hebbian learning strengthens: SEQ[t] → SEQ[t+1]

# After training, SEQ can predict:
# Given MOOD[declarative], first expect NOUN, then VERB
```

## Generation

To generate grammatically correct sentences:

```python
def generate(structure="SVO", mood="declarative"):
    # 1. Activate mood
    activate(MOOD, mood)
    project({}, {MOOD: [SEQ]})
    
    # 2. SEQ predicts first category (e.g., NOUN for SVO)
    project({}, {SEQ: [CORE]})
    # NOUN_CORE activates
    
    # 3. Sample a noun from NOUN_CORE → NOUN → LEX
    project({}, {NOUN_CORE: [NOUN]})
    project({}, {NOUN: [LEX]})
    first_word = get_word(LEX)
    
    # 4. Advance SEQ, predict next category
    project({}, {CORE: [SEQ]})
    project({}, {SEQ: [CORE]})
    # VERB_CORE activates
    
    # 5. Sample a verb
    # ... and so on
```

## Key Insights

### 1. Competition Through Inhibition
- Areas compete for activation
- Only disinhibited areas can fire
- Rules control which areas/fibers are active at each step

### 2. Merge Through Co-projection
- When DET and LEX both project to SUBJ
- The resulting SUBJ assembly encodes BOTH
- This is how "the dog" becomes a single noun phrase

### 3. Binding Through Reciprocal Projection
- VERB ↔ SUBJ creates bidirectional link
- Later, projecting VERB → SUBJ retrieves the subject
- This is how grammatical relations are stored

### 4. Sequence Learning
- SEQ area learns temporal order
- CORE areas (NOUN_CORE, VERB_CORE) represent categories
- Together they encode word order statistics

## What We Need for Text Generation

1. **Vocabulary (LEX)**: Explicit area with word assemblies ✅
2. **Syntactic Areas**: SUBJ, OBJ, VERB, etc. ✅
3. **Inhibition Control**: Fiber and area states ✅
4. **Word Type Rules**: PRE_RULES and POST_RULES ✅
5. **Sequence Memory (SEQ)**: For word order ⚠️ (partial)
6. **Category Cores**: NOUN_CORE, VERB_CORE 🚧 (not implemented)
7. **Mood Control**: For different sentence types 🚧 (not implemented)

## Files Created

- `src/text_generation/grammatical_assembly_brain.py` - Main implementation
- `src/text_generation/assembly_text_generator.py` - Simple sequence-based generator

## Next Steps

1. **Implement SEQ area** for word order learning
2. **Add CORE areas** for grammatical categories
3. **Train on corpus** to learn word order statistics
4. **Implement generation** using learned structures
5. **Add MOOD control** for questions, commands, etc.


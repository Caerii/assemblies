# NEMO Neurobiological Patterns

Summary of the key patterns from papers and our implementation.

## 1. Architecture (from Mitropolsky & Papadimitriou 2025)

```
    Phon ─────────┬──────────┐
                  ▼          ▼
    Visual ──→ Lex1 ──→ NP ──┬──→ Sent
    Motor ───→ Lex2 ──→ VP ──┘
                  │
                  ▼
    Role_agent ←─┼─→ Role_action ←─┼─→ Role_patient
         (mutual inhibition)
                  │
                  ▼
                 SEQ (word order)
```

### Key Areas

| Area | Purpose | Grounding |
|------|---------|-----------|
| Phon | Phonological input | Sound patterns |
| Visual | Visual grounding | Objects, scenes |
| Motor | Motor grounding | Actions, movements |
| Lex1 | Noun lexicon | Strong → Visual |
| Lex2 | Verb lexicon | Strong → Motor |
| NP | Noun phrase | Combines Det+Adj+N |
| VP | Verb phrase | Combines V+NP |
| Sent | Full sentence | Combines NP+VP |
| Role_agent | Agent/Subject | Thematic role |
| Role_action | Action/Verb | Thematic role |
| Role_patient | Patient/Object | Thematic role |
| SEQ | Sequence | Word order |

## 2. Key Principles

### Grounded Learning
- Words are ALWAYS presented with sensory context
- Grounding fires CONTINUOUSLY while word is presented
- This creates differential association:
  - Nouns: Phon + Visual → Lex1
  - Verbs: Phon + Motor → Lex2

### Stability-Based Classification
- Stable assembly = word learned in correct area
- Wobbly assembly = word not learned
- Classification: Check which Lex area responds more strongly

### Strong vs Regular Fibers
- Strong fibers: p=0.1, β=0.15 (Phon→Lex, Lex→Grounding)
- Regular fibers: p=0.05, β=0.1 (everything else)

### Role Areas with Mutual Inhibition
- Only ONE role active at a time
- Prevents ambiguity in sentence parsing
- Learned through competition

## 3. Curriculum Stages (Child Language Acquisition)

| Stage | Age | Words | Key Features |
|-------|-----|-------|--------------|
| 1 | 12-18mo | ~50 | Single words, naming |
| 2 | 18-24mo | ~300 | Vocabulary spurt, two-word combos |
| 3 | 24-30mo | ~500 | Telegraphic speech, SVO emerging |
| 4 | 30-36mo | ~1000 | Full sentences, auxiliaries |

### Stage 1 Patterns
- 60% single words
- 30% two-word combinations
- 10% three-word (rare)

### Stage 2 Patterns
- Adjective + Noun: "big dog"
- Agent + Action: "dog run"
- Action + Object: "eat cookie"

### Stage 3 Patterns
- SVO emerging: "mommy read book"
- Location phrases: "ball on table"
- Questions: "where mommy go"

### Stage 4 Patterns
- Full sentences: "the dog is running"
- Modal verbs: "i can run fast"
- Complex sentences: "i want to go to the park"

## 4. Grounding Types

```python
class GroundingType(Enum):
    VISUAL = auto()      # Objects, scenes
    MOTOR = auto()       # Actions, movements
    AUDITORY = auto()    # Sounds
    TACTILE = auto()     # Touch sensations
    EMOTIONAL = auto()   # Feelings
    SPATIAL = auto()     # Locations
```

## 5. Speech Acts (Pragmatic Function)

```python
class SpeechAct(Enum):
    NAMING = auto()         # "That's a dog"
    DESCRIBING = auto()     # "The dog is big"
    REQUESTING = auto()     # "Give me the ball"
    COMMANDING = auto()     # "Sit down"
    QUESTIONING = auto()    # "What's that?"
```

## 6. Word Features (from lexicon)

Each word has:
- **Semantic domains**: ANIMAL, PERSON, FOOD, MOTION, etc.
- **Features**: animate, human, transitive, stative, etc.
- **Argument structure**: agent, theme, patient, experiencer
- **Age of acquisition (AoA)**: When typically learned
- **Frequency**: Log frequency

## 7. Results

### Noun/Verb Classification
- 96.4% mean accuracy (5 runs)
- Uses activation-based classification
- Tests which Lex area responds to word's phon assembly

### Word Order Learning
- SVO correctly learned from training
- Transitions tracked: SUBJ→VERB, VERB→OBJ
- Can generalize to new sentences

## 8. Implementation Notes

### k = sqrt(n)
- Assembly size follows theoretical predictions
- n=10,000 → k=100

### Implicit Random Connectivity
- No explicit weight matrices stored
- Hash-based connectivity (O(1) memory)
- Only learned weights stored (sparse)

### Hebbian Learning with Saturation
- delta = β * (1 - w/w_max)
- Prevents weight explosion
- Converges to stable assemblies


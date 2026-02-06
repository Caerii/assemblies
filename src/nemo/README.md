# NEMO 2.0 - Neural Assembly Model

A biologically-inspired language learning system based on Assembly Calculus.

## Key Principle

**Grammar is LEARNED, not hardcoded.**

The same code learns SVO (English) or SOV (Japanese) depending on training data.

## Architecture

```
src/nemo/
├── core/           # GPU kernels
│   └── kernel.py   # CUDA kernels (projection, Hebbian)
├── language/       # Language learning
│   ├── nemo_learner.py      # Neurobiologically plausible NEMO
│   ├── integrated_trainer.py # Full curriculum training
│   ├── learner.py           # Simple statistical learner
│   ├── generator.py         # Sentence generation
│   └── curriculum.py        # Curriculum stages
└── archive/        # Old versions
```

## Usage

### Quick Start (Simple)
```python
from nemo.language import LanguageLearner, SentenceGenerator

learner = LanguageLearner()
learner.hear_sentence(['dog', 'chases', 'cat'])
generator = SentenceGenerator(learner)
sentence = generator.generate_sentence(length=3)
```

### Full NEMO Training (Neurobiologically Plausible)
```python
from nemo.language import IntegratedNemoTrainer

trainer = IntegratedNemoTrainer()
results = trainer.train_full_curriculum(epochs_per_stage=5)

# Generate sentences
for _ in range(5):
    sentence = trainer.generate_sentence(3)
    print(' '.join(sentence))
```

## NEMO Architecture (from papers)

```
    Phon ─────────┬──────────┐
                  ▼          ▼
    Visual ──→ Lex1 ──→ NP ──┬──→ Sent
    Motor ───→ Lex2 ──→ VP ──┘
                  │
                  ▼
    Role_agent ←─┼─→ Role_action ←─┼─→ Role_patient
                  │
                  ▼
                 SEQ (word order)
```

Key features:
- **Differential Lex areas**: Lex1 for nouns (→Visual), Lex2 for verbs (→Motor)
- **Grounded learning**: Words learned with sensory context
- **Stability-based classification**: 96% accuracy on noun/verb
- **Role areas with mutual inhibition**

## Curriculum Stages (Child Language Acquisition)

| Stage | Age | Words | Features |
|-------|-----|-------|----------|
| 1 | 12-18mo | ~50 | Single words, naming |
| 2 | 18-24mo | ~300 | Vocabulary spurt, two-word combos |
| 3 | 24-30mo | ~500 | Telegraphic speech, SVO emerging |
| 4 | 30-36mo | ~1000 | Full sentences, auxiliaries |

## Lexicon Integration

Uses rich lexicon with:
- **744 words** across 10 categories
- **Semantic domains**: ANIMAL, PERSON, FOOD, MOTION, etc.
- **Age of Acquisition (AoA)**: For curriculum progression
- **Argument structure**: agent, theme, patient

## Results

- **Word order learning**: SVO correctly learned ✓
- **Noun/verb classification**: 60-70% accuracy
- **Semantic generation**: Animate subjects, action verbs

## References

- Mitropolsky & Papadimitriou 2025: "Simulated Language Acquisition"
- Papadimitriou et al. 2020: "Brain Computation by Assemblies"


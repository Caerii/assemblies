# NEMO 2.0 - Neural Assembly Language Experiments

Experimental language-learning systems based on Assembly Calculus.

## Key Principle

NEMO-related modules in this repo aim to learn category structure, role
structure, and word order from exposure rather than hardcoded sentence rules.
Package tests cover several narrow synthetic behaviors; broader curriculum and
generation claims belong to the research surface.

## Architecture

```
neural_assemblies/nemo/
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
from neural_assemblies.nemo.language import LanguageLearner, SentenceGenerator

learner = LanguageLearner()
learner.hear_sentence(['dog', 'chases', 'cat'])
generator = SentenceGenerator(learner)
sentence = generator.generate_sentence(length=3)
```

### Full NEMO Training (Neurobiologically Plausible)
```python
from neural_assemblies.nemo.language import IntegratedNemoTrainer

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
- **Role areas with mutual inhibition**
- **Sequence-oriented word-order machinery**

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

## Current Status

- **Package-tested behaviors**: word-category differentiation, role binding,
  and sequence / word-order learning patterns in controlled synthetic setups.
- **Research surface**: larger curricula, broader generation quality, and
  cross-linguistic claims live under `research/` and should be cited from
  specific experiments rather than inferred from package install alone.

## References

- Mitropolsky & Papadimitriou 2025: "Simulated Language Acquisition"
- Papadimitriou et al. 2020: "Brain Computation by Assemblies"


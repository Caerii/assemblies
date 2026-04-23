# NEMO

Experimental language-learning systems built on assembly-style computation.

This package should be read as a research-oriented surface inside
`neural_assemblies`, not as the same stability tier as the core runtime.

## Scope

NEMO-related code in this repository aims to learn category structure, role
structure, and word order from exposure rather than fixed sentence templates.

What is package-backed here:

- narrow synthetic tests for category differentiation
- role binding behavior
- sequence and word-order related patterns in controlled settings

What is not a package guarantee:

- broad curriculum success
- general language acquisition
- cross-linguistic competence claims
- generation quality claims beyond specific experiments

For that boundary, see [../../docs/scientific_status.md](../../docs/scientific_status.md).

## Layout

```text
neural_assemblies/nemo/
|-- core/        # NEMO-specific brain and kernels
|-- language/    # Learners, generators, curriculum, integrated trainer
|-- archive/     # Older experiments and historical code
`-- README.md
```

## Installation Notes

Many NEMO paths assume optional GPU-oriented dependencies.

From a checkout:

```bash
uv sync --group gpu
```

From the published package:

```bash
pip install "neural-assemblies[gpu]"
```

## Quick Start

### Simple Learner Surface

```python
from neural_assemblies.nemo.language import LanguageLearner, SentenceGenerator

learner = LanguageLearner()
learner.hear_sentence(["dog", "chases", "cat"])

generator = SentenceGenerator(learner)
sentence = generator.generate_sentence(length=3)
print(sentence)
```

### Integrated Trainer Surface

```python
from neural_assemblies.nemo.language import IntegratedNemoTrainer

trainer = IntegratedNemoTrainer()
results = trainer.train_full_curriculum(epochs_per_stage=1)
sentence = trainer.generate_sentence(3)
print(sentence)
```

The second path is a research-oriented training workflow, not the default
package contract.

## Conceptual Architecture

The NEMO code organizes around a few recurring ideas from the papers:

- differentiated lexical pathways for nouns and verbs
- grounding-oriented learning rather than pure text-only templates
- role areas and inhibitory structure
- sequence-sensitive machinery for word order

This repository contains implementations and experiments inspired by those
ideas; it should not be read as a claim that every paper-level result has been
fully reproduced as a package guarantee.

## Related Surfaces

- [../language/README.md](../language/README.md) for rule-based parsing
- [../lexicon/README.md](../lexicon/README.md) for vocabulary and curriculum
- [../../research/README.md](../../research/README.md) for experiment and claim
  tracking

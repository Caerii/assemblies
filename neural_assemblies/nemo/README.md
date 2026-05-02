# NEMO

`neural_assemblies.nemo` contains experimental language-learning code inspired
by NEMO and the language-organ papers.

Use it for category, role, and word-order experiments. Do not treat it as the
same maturity level as the core runtime.

## What The Tests Cover

- category differentiation in narrow synthetic settings
- role binding behavior
- sequence and word-order patterns in controlled examples

## What Still Needs Research Evidence

- broad curriculum success
- general language acquisition
- cross-linguistic competence
- generation quality beyond named experiments

For the claim boundary, read
[Scientific status](../../docs/scientific_status.md).

## Layout

```text
neural_assemblies/nemo/
|-- core/        # NEMO-specific brain and kernels
|-- language/    # Learners, generators, curriculum, trainer
|-- archive/     # Older experiments and historical code
`-- README.md
```

## Install GPU Dependencies

From a checkout:

```bash
uv sync --group gpu
```

From the published package:

```bash
pip install "neural-assemblies[gpu]"
```

## Simple Learner

```python
from neural_assemblies.nemo.language import LanguageLearner, SentenceGenerator

learner = LanguageLearner()
learner.hear_sentence(["dog", "chases", "cat"])

generator = SentenceGenerator(learner)
sentence = generator.generate_sentence(length=3)
print(sentence)
```

## Integrated Trainer

```python
from neural_assemblies.nemo.language import IntegratedNemoTrainer

trainer = IntegratedNemoTrainer()
results = trainer.train_full_curriculum(epochs_per_stage=1)
sentence = trainer.generate_sentence(3)
print(sentence)
```

The trainer is a research workflow. Interpret its output through the relevant
experiment or claim document.

## Conceptual Roots

The implementation centers on ideas from the language-organ and simulated
language-acquisition papers:

- differentiated lexical pathways for nouns and verbs
- grounding-oriented learning instead of pure text templates
- role areas and inhibition
- sequence-sensitive machinery for word order

The code implements and extends those ideas. It does not claim that every
paper-level result has been reproduced as a package guarantee.

## See Also

- [Language](../language/README.md) for rule-based parsing
- [Lexicon](../lexicon/README.md) for vocabulary and curriculum data
- [Research guide](../../research/README.md) for experiment and claim tracking

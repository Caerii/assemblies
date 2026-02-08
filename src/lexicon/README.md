# Lexicon — Words, curriculum, learners

The **lexicon** module provides a large, structured word set and curriculum for language learning: categories (nouns, verbs, etc.), frequency data, curriculum stages, and assembly-based learners (including GPU).

## What’s here

| Component | File / folder | Role |
|-----------|----------------|------|
| **LexiconManager** | `lexicon_manager.py` | Load and query words; `Word`, `WordCategory` |
| **WordStatistics** | `statistics.py` | Frequency and co-occurrence stats |
| **data/** | `data/*.py` | Word lists by category (nouns, verbs, adjectives, etc.) |
| **curriculum/** | `curriculum/` | Stages: first words, vocabulary spurt, two-word, sentences |
| **Assembly learners** | `assembly_language_learner.py`, `true_assembly_learner.py` | Assembly-based word/sentence learning |
| **GPU learners** | `gpu_assembly_learner.py`, `gpu_language_learner.py` | GPU-accelerated lexicon/curriculum learning |

## Quick use

```python
from src.lexicon import LexiconManager, Word, WordCategory

manager = LexiconManager()
# Load lexicon, query by category, get curriculum order
words = manager.get_words_by_category(WordCategory.NOUN)
```

For full curriculum training with assemblies (often with NEMO), see the integrated trainer in [src/nemo](nemo/README.md) and the curriculum modules in `lexicon/curriculum/`.

## Curriculum stages

Roughly aligned with child language acquisition:

- **Stage 1** — First words (~50), single-word naming.
- **Stage 2** — Vocabulary spurt (~300), two-word combinations.
- **Stage 3** — Telegraphic (~500), simple sentences.
- **Stage 4** — Full sentences (~1000), auxiliaries, etc.

Details and data live in `curriculum/` and `data/`.

## See also

- [src/nemo](nemo/README.md) — NEMO language learning and curriculum integration.
- [research/](../../research/README.md) — Experiments and results using the lexicon and curriculum.

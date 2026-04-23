# Lexicon

The `neural_assemblies.lexicon` package provides structured vocabulary,
curriculum data, and learner support for the language experiments.

## Main Components

| Component | File | Role |
|-----------|------|------|
| `LexiconManager` | `lexicon_manager.py` | Load and query words and categories. |
| `Word`, `WordCategory` | `lexicon_manager.py` | Core lexicon data types. |
| `WordStatistics` | `statistics.py` | Frequency and co-occurrence helpers. |
| `data/` | `data/` | Word lists grouped by category. |
| `curriculum/` | `curriculum/` | Staged curriculum data and helpers. |
| assembly learners | `assembly_language_learner.py`, `true_assembly_learner.py` | Assembly-based learner variants. |
| GPU learners | `gpu_assembly_learner.py`, `gpu_language_learner.py` | Optional GPU-oriented learner variants. |

## Example

```python
from neural_assemblies.lexicon import LexiconManager, WordCategory

manager = LexiconManager()
nouns = manager.get_words_by_category(WordCategory.NOUN)
print(len(nouns))
```

The lexicon package is infrastructure for broader language experiments. It is
useful on its own, but stronger curriculum or acquisition claims should still
be tied to the research artifacts.

## See Also

- [../nemo/README.md](../nemo/README.md)
- [../../research/README.md](../../research/README.md)

# Lexicon

`neural_assemblies.lexicon` provides vocabulary data, category labels,
curriculum helpers, and learner variants for language experiments.

## Objects

| Object | File | Use it for |
|--------|------|------------|
| `LexiconManager` | `lexicon_manager.py` | Loading and querying words and categories. |
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

The lexicon is useful infrastructure. Claims about curriculum learning or
language acquisition still need specific research artifacts.

## See Also

- [NEMO](../nemo/README.md)
- [Research guide](../../research/README.md)

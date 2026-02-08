# Language — Parsing and grammar

The **language** module implements **rule-based** sentence parsing with the assembly calculus: predefined grammar rules, language areas (e.g. NP, VP, LEX), and readout methods. For **learned** grammar and word order (NEMO), see [src/nemo](nemo/README.md).

## What’s here

| Component | File | Role |
|-----------|------|------|
| **ParserBrain** | `parser.py` | Base parser brain; subclasses for English/Russian |
| **EnglishParserBrain** | `parser.py` | English areas and projection map |
| **RussianParserBrain** | `parser.py` | Russian areas and projection map |
| **Grammar rules** | `grammar_rules.py` | `LEXEME_DICT`, `RUSSIAN_LEXEME_DICT` (pre/post rules per word) |
| **Language areas** | `language_areas.py` | Area names, explicit areas, readout rules |
| **Readout** | `readout_methods.py` | `fixed_map_readout`, `fiber_readout` |
| **parse()** | `__init__.py` | One-shot: `parse("cats chase mice", language="English")` |

## Quick use

```python
from src.language import parse, EnglishParserBrain, ParserBrain

# One-liner (uses default brain and readout)
result = parse("cats chase mice", language="English")

# Or build a parser brain and run your own loop
from src.language import EnglishParserBrain
b = EnglishParserBrain(p=0.1, LEX_k=20)
b.activateWord("LEX", "cats")
# ... apply rules, project, readout
```

Root-level convenience (same behavior):

```python
from parser import parse  # re-exports src.language
parse("cats chase mice", language="English")
```

## Language support

- **English** — SVO; areas and lexeme dict in `grammar_rules.py` / `language_areas.py`.
- **Russian** — SOV and case; `RussianParserBrain`, `RUSSIAN_LEXEME_DICT`, Russian areas and readout.

## See also

- [src/nemo](nemo/README.md) — Learned grammar and word order (no hardcoded rules).
- [DOCUMENTATION.md](../../DOCUMENTATION.md) — Parser simulation and Turing details.

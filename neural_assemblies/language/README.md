# Language

`neural_assemblies.language` implements rule-based parsing with explicit
grammar rules and hand-specified language areas.

Use `neural_assemblies.nemo` for learned-language experiments.

## Objects

| Object | File | Use it for |
|--------|------|------------|
| `ParserBrain` | `parser.py` | Base parser brain. |
| `EnglishParserBrain` | `parser.py` | English parser configuration. |
| `RussianParserBrain` | `parser.py` | Russian parser configuration. |
| `LEXEME_DICT` | `grammar_rules.py` | English lexical rules. |
| `RUSSIAN_LEXEME_DICT` | `grammar_rules.py` | Russian lexical rules. |
| `ReadoutMethod` | `readout_methods.py` | Readout mode selection. |
| `fixed_map_readout`, `fiber_readout` | `readout_methods.py` | Readout helpers. |
| `ParserDebugger` | `debugger.py` | Debugging support. |
| `parse(...)` | `__init__.py` | One-shot parser helper. |

## Example

```python
from neural_assemblies.language import parse

result = parse("cats chase mice", language="English")
print(result)
```

From a checkout, `from parser import parse` still works through the root
compatibility shim.

## See Also

- [NEMO](../nemo/README.md)
- [API guide](../../docs/api.md)

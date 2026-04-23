# Language

The `neural_assemblies.language` package is the rule-based parsing surface.

Use this package when you want explicit grammar rules and hand-specified
language areas. Use `neural_assemblies.nemo` when you want the experimental
learned-language surfaces.

## Main Components

| Component | File | Role |
|-----------|------|------|
| `ParserBrain` | `parser.py` | Base parser brain. |
| `EnglishParserBrain` | `parser.py` | English parser configuration. |
| `RussianParserBrain` | `parser.py` | Russian parser configuration. |
| `LEXEME_DICT` | `grammar_rules.py` | English lexical rules. |
| `RUSSIAN_LEXEME_DICT` | `grammar_rules.py` | Russian lexical rules. |
| `ReadoutMethod` | `readout_methods.py` | Readout mode selection. |
| `fixed_map_readout`, `fiber_readout` | `readout_methods.py` | Readout helpers. |
| `ParserDebugger` | `debugger.py` | Debugging support. |
| `parse(...)` | `__init__.py` | One-shot parsing helper. |

## Example

```python
from neural_assemblies.language import parse

result = parse("cats chase mice", language="English")
print(result)
```

From a repo checkout, `from parser import parse` still works through the
root compatibility shim.

## See Also

- [../nemo/README.md](../nemo/README.md)
- [../../docs/api.md](../../docs/api.md)
